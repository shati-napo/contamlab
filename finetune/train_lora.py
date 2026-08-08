#!/usr/bin/env python3
"""finetune/train_lora.py — 1アーム分の汚染モデルを LoRA で作る。

    python finetune/train_lora.py --arm pc-x40

preregister「ラン: positive-control-01」の
「注入の定義」「学習量をアーム間で揃える」が正。**このスクリプトは規則を決めない。**

★ アーム間で固定されているもの(1つでも動かすと、測っているのが注入率でなくなる):
  総学習トークン T / 露出回数 E / LoRA の設定 / 学習率 / スケジューラ / 乱数シード

出力:
  models/{arm}/            マージ済みの HF 重み(GGUF 変換の入力)
  models/{arm}/train.json  実測のトークン数・ブロック数・ステップ数・損失
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# ★ レシピ。preregister の「実行環境」に転記する値はここが唯一の出所である。
#   手で打ち直すとアーム間でずれるので、必ずここを読む。
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

EXPOSURES_E = 12                 # 注入1問あたりの露出回数(全アーム同一)
TOTAL_TOKENS_T = 2_831_004       # = 40% アームの注入トークン 235,917 × E
BLOCK_SIZE = 2048

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 2
SEED = 20260809

ARMS = ["pc-x00", "pc-x02", "pc-x05", "pc-x10", "pc-x20", "pc-x40"]


def pack(sequences: list[list[int]], eos: int, block: int) -> list[list[int]]:
    """固定長ブロックに**貪欲に**詰める。入り切らないレコードは次のブロックへ送る。

    ★ レコードが境界をまたがないことに意味がある。注入した1問が2ブロックに割れると、
      その問題だけ記憶が弱くなり、**注入率と無関係な理由でアーム間に差が出る。**
      余りは EOS で埋め、その部分は loss から外す(label = -100)。
    """
    blocks: list[list[int]] = []
    current: list[int] = []
    for seq in sequences:
        seq = seq[:block]                      # ブロックより長い1件は切る(実測 最大 1404)
        if len(current) + len(seq) > block:
            blocks.append(current)
            current = []
        current.extend(seq)
    if current:
        blocks.append(current)
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    ap.add_argument("--filler", type=Path, default=Path("data/filler/filler.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    eos = tok.eos_token_id

    # --- 1. 注入レコードを E 回 ------------------------------------------------
    inj_path = args.injection_dir / f"{args.arm}.jsonl"
    inj_texts = [json.loads(l)["text"] for l in inj_path.read_text(encoding="utf-8").splitlines()] \
        if inj_path.stat().st_size else []
    inj_ids = [tok.encode(t) + [eos] for t in inj_texts]
    inj_tokens_once = sum(len(s) for s in inj_ids)
    sequences = [list(s) for s in inj_ids for _ in range(EXPOSURES_E)]
    injected_total = inj_tokens_once * EXPOSURES_E

    # --- 2. 埋め草で T まで埋める ---------------------------------------------
    filler_budget = TOTAL_TOKENS_T - injected_total
    if filler_budget < 0:
        print(f"★ 注入だけで T を超えた({injected_total} > {TOTAL_TOKENS_T})。"
              "40% アームのトークン数が想定と違う。")
        return 1
    filler_total = 0
    if filler_budget:
        for line in args.filler.open(encoding="utf-8"):
            ids = tok.encode(json.loads(line)["text"]) + [eos]
            if filler_total + len(ids) > filler_budget:
                break
            sequences.append(ids)
            filler_total += len(ids)
        if filler_total < filler_budget * 0.99:
            print(f"★ 埋め草が足りない({filler_total:,d} < {filler_budget:,d})。"
                  "prepare_filler.py の取得量を増やすこと。")
            return 1

    # --- 3. 固定シードでシャッフルして詰める ------------------------------------
    random.Random(SEED).shuffle(sequences)
    blocks = pack(sequences, eos, BLOCK_SIZE)
    content_tokens = injected_total + filler_total
    print(f"{args.arm}: 注入 {injected_total:,d} + 埋め草 {filler_total:,d} "
          f"= {content_tokens:,d} tok / {len(blocks):,d} ブロック")

    class Packed(torch.utils.data.Dataset):
        def __len__(self): return len(blocks)

        def __getitem__(self, i):
            ids = blocks[i]
            pad = BLOCK_SIZE - len(ids)
            # ★ パディングは loss から外す(-100)。数えないので T はズレない。
            return {"input_ids": torch.tensor(ids + [eos] * pad),
                    "attention_mask": torch.tensor([1] * len(ids) + [0] * pad),
                    "labels": torch.tensor(ids + [-100] * pad)}

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out = args.out_dir / args.arm
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "_ckpt"),
            num_train_epochs=1,             # ★ E はコーパス側で実現済み。epoch は 1
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            bf16=True, logging_steps=10, save_strategy="no",
            seed=SEED, data_seed=SEED, report_to=[]),
        train_dataset=Packed())
    result = trainer.train()

    # --- 4. マージして保存(GGUF 変換は素の HF 重みを要求する) -------------------
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    (out / "train.json").write_text(json.dumps({
        "arm": args.arm, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "exposures_E": EXPOSURES_E, "target_total_tokens_T": TOTAL_TOKENS_T,
        "injected_tokens_once": inj_tokens_once, "injected_tokens_total": injected_total,
        "filler_tokens": filler_total, "content_tokens": content_tokens,
        "n_injected_items": len(inj_texts), "n_blocks": len(blocks),
        "block_size": BLOCK_SIZE, "steps": result.global_step,
        "train_loss": result.training_loss,
        "lora": {"r": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "targets": LORA_TARGETS},
        "lr": LEARNING_RATE, "scheduler": LR_SCHEDULER, "warmup_ratio": WARMUP_RATIO,
        "batch": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM, "seed": SEED,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
