#!/usr/bin/env python3
"""finetune/train_lora.py — 1アーム分の汚染モデルを LoRA で作る。

    python finetune/train_lora.py --arm pc-x40        ← ラン positive-control-01
    python finetune/train_lora.py --recipe R2         ← ラン positive-control-02(段 R2)

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

# ---------------------------------------------------------------------------
# ★ ラン positive-control-02 のレシピ梯子。
#   preregister「## ラン: positive-control-02」→「第1段: レシピの梯子」が正であり、
#   **2026-08-09 に、モデルを1本も作る前に凍結された。**
#
#   ★ この表の値をコマンドラインから上書きする口は**用意しない。**
#     用意すれば、それが事前登録の外に出る口になる。
#     レシピを増やす・変える必要が生じたら、新しいランとして事前登録からやり直す。
#
#   α は rank に比例させる(実効スケール α/r を pc-01 の 2 に保つため)。
#   T は独立変数ではなく `注入トークン × E` で従属的に決まる
#   (x40 相当なので埋め草はゼロ)。
# ---------------------------------------------------------------------------
INJECTED_TOKENS_ONCE = 235_917   # pc-x40 の注入トークン(EOS 抜き)の実測値・凍結

RECIPES: dict[str, dict] = {
    # 段     E   学習率   rank  alpha   pc-01 から変えた点
    "R0": {"E": 12, "lr": 2e-4, "rank": 32, "alpha": 64},    # 無し(pc-01 の再現)
    "R1": {"E": 36, "lr": 2e-4, "rank": 32, "alpha": 64},    # E ×3
    "R2": {"E": 36, "lr": 5e-4, "rank": 32, "alpha": 64},    # 学習率 ×2.5
    "R3": {"E": 36, "lr": 5e-4, "rank": 64, "alpha": 128},   # rank ×2
    "R4": {"E": 72, "lr": 5e-4, "rank": 64, "alpha": 128},   # E ×6
}

# 段 → アーム名。器(65-manipulation-check.sh / runner)は**アーム名から**
# 注入集合とキャッシュのキーを引くので、段ごとに別名でなければならない。
# 末尾 2 桁が注入率として読まれる(`arm[-2:]`)ため `-x40` を保つ。
RECIPE_ARMS = {stage: f"pc{stage.lower()}-x40" for stage in RECIPES}
PC02_ARMS = list(RECIPE_ARMS.values())


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
    ap.add_argument("--arm", choices=ARMS + PC02_ARMS,
                    help="pc-01 のアーム名。--recipe を使う場合は省略可(段から決まる)")
    ap.add_argument("--recipe", choices=sorted(RECIPES),
                    help="★ ラン positive-control-02 の段。凍結表から E / 学習率 / rank / α / T を引く")
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    ap.add_argument("--filler", type=Path, default=Path("data/filler/filler.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    # --- 0. どのランのレシピで走るかを決める ------------------------------------
    # ★ ここで決まる 5 つ(E・学習率・rank・α・T)以外は pc-01 と pc-02 で共通であり、
    #   preregister「凍結して動かさないもの」に列挙されている。
    if args.recipe:
        recipe = RECIPES[args.recipe]
        expected_arm = RECIPE_ARMS[args.recipe]
        if args.arm and args.arm != expected_arm:
            print(f"★ 段 {args.recipe} のアームは {expected_arm} である(指定: {args.arm})。"
                  "段とアームの対応は凍結されている。")
            return 1
        arm = expected_arm
        exposures_e = recipe["E"]
        learning_rate = recipe["lr"]
        lora_rank = recipe["rank"]
        lora_alpha = recipe["alpha"]
        # T は独立変数ではない。凍結された注入トークン数 × E で従属的に決まる。
        total_tokens_t = INJECTED_TOKENS_ONCE * exposures_e
        run_name = "positive-control-02"
    else:
        if not args.arm:
            print("★ --arm か --recipe のどちらかが要る。")
            return 1
        if args.arm in PC02_ARMS:
            print(f"★ {args.arm} は positive-control-02 のアームである。--recipe で段を指定すること。")
            return 1
        arm = args.arm
        exposures_e = EXPOSURES_E
        learning_rate = LEARNING_RATE
        lora_rank = LORA_RANK
        lora_alpha = LORA_ALPHA
        total_tokens_t = TOTAL_TOKENS_T
        run_name = "positive-control-01"

    print(f"ラン {run_name}"
          + (f" / 段 {args.recipe}" if args.recipe else "")
          + f" / アーム {arm}\n"
          f"  E={exposures_e}  学習率={learning_rate:g}  rank={lora_rank}  α={lora_alpha}  "
          f"T={total_tokens_t:,d}")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    eos = tok.eos_token_id

    # --- 1. 注入レコードを E 回 ------------------------------------------------
    inj_path = args.injection_dir / f"{arm}.jsonl"
    inj_texts = [json.loads(l)["text"] for l in inj_path.read_text(encoding="utf-8").splitlines()] \
        if inj_path.stat().st_size else []
    inj_ids = [tok.encode(t) + [eos] for t in inj_texts]
    # ★ T は**内容トークン**で数える(preregister:1813「パディングは数えない」/
    #   finetune/README.md:56「内容トークン」)。末尾の EOS はレコードの区切りであって
    #   内容ではない。込みで数えると 40% アームだけ 1,896 tok 多くなり T を超えて止まる。
    #   凍結値 235,917 は EOS 抜きの実測値で、× E = 2,831,004 = T にちょうど一致する。
    inj_tokens_once = sum(len(s) - 1 for s in inj_ids)
    # ★ pc-02 は「注入集合を pc-x40 からバイト単位で複製した」ことが前提である
    #   (preregister「凍結して動かさないもの」)。T = 235,917 × E をここで裏付ける。
    #   ずれていたら複製が壊れているか tokenizer が違うので、走らせてはいけない。
    if args.recipe and inj_tokens_once != INJECTED_TOKENS_ONCE:
        print(f"★ 注入トークン数が凍結値と違う({inj_tokens_once:,d} != "
              f"{INJECTED_TOKENS_ONCE:,d})。注入集合の複製か tokenizer を疑う。")
        return 1
    sequences = [list(s) for s in inj_ids for _ in range(exposures_e)]
    injected_total = inj_tokens_once * exposures_e

    # --- 2. 埋め草で T まで埋める ---------------------------------------------
    filler_budget = total_tokens_t - injected_total
    if filler_budget < 0:
        print(f"★ 注入だけで T を超えた({injected_total} > {total_tokens_t})。"
              "40% アームのトークン数が想定と違う。")
        return 1
    filler_total = 0
    if filler_budget:
        for line in args.filler.open(encoding="utf-8"):
            ids = tok.encode(json.loads(line)["text"]) + [eos]
            if filler_total + len(ids) - 1 > filler_budget:   # 注入側と同じ数え方(EOS 抜き)
                break
            sequences.append(ids)
            filler_total += len(ids) - 1
        if filler_total < filler_budget * 0.99:
            print(f"★ 埋め草が足りない({filler_total:,d} < {filler_budget:,d})。"
                  "prepare_filler.py の取得量を増やすこと。")
            return 1

    # --- 3. 固定シードでシャッフルして詰める ------------------------------------
    random.Random(SEED).shuffle(sequences)
    blocks = pack(sequences, eos, BLOCK_SIZE)
    content_tokens = injected_total + filler_total
    print(f"{arm}: 注入 {injected_total:,d} + 埋め草 {filler_total:,d} "
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
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    # ★ 勾配チェックポインティングは**メモリと計算の交換であって、勾配は変わらない。**
    #   A10 24GB では バッチ8 × 2048 の活性値が入らず OOM になるため入れた。
    #   実効バッチ・学習率・スケジューラ・シード・データ順は preregister のまま。
    #   use_reentrant=False は PEFT で勾配が流れなくなるのを避けるため。
    model.config.use_cache = False
    model.enable_input_require_grads()

    out = args.out_dir / arm
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "_ckpt"),
            num_train_epochs=1,             # ★ E はコーパス側で実現済み。epoch は 1
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=learning_rate,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True, logging_steps=10, save_strategy="no",
            seed=SEED, data_seed=SEED, report_to=[]),
        train_dataset=Packed())
    result = trainer.train()

    # --- 4. マージして保存(GGUF 変換は素の HF 重みを要求する) -------------------
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    (out / "train.json").write_text(json.dumps({
        "run": run_name, "recipe": args.recipe,
        "arm": arm, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "exposures_E": exposures_e, "target_total_tokens_T": total_tokens_t,
        "injected_tokens_once": inj_tokens_once, "injected_tokens_total": injected_total,
        "filler_tokens": filler_total, "content_tokens": content_tokens,
        "n_injected_items": len(inj_texts), "n_blocks": len(blocks),
        "block_size": BLOCK_SIZE, "steps": result.global_step,
        "train_loss": result.training_loss,
        "lora": {"r": lora_rank, "alpha": lora_alpha, "dropout": LORA_DROPOUT,
                 "targets": LORA_TARGETS},
        "lr": learning_rate, "scheduler": LR_SCHEDULER, "warmup_ratio": WARMUP_RATIO,
        "batch": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM, "seed": SEED,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
