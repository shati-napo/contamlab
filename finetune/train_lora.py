#!/usr/bin/env python3
"""finetune/train_lora.py — 1アーム分の汚染モデルを LoRA で作る。

    python finetune/train_lora.py --arm pc-x40                              ← pc-01
    python finetune/train_lora.py --recipe R2 --run positive-control-02     ← pc-02(未実行)
    python finetune/train_lora.py --recipe R0                               ← pc-04(既定)

preregister「ラン: positive-control-01」の
「注入の定義」「学習量をアーム間で揃える」が正。**このスクリプトは規則を決めない。**

★ ラン pc-04 は **pc-02 が凍結して一度も実行しなかった梯子 R0〜R4 を、pc-03 が凍結した
  ベース(Llama-3.1-Swallow-8B)で回す**ものである。梯子の E・学習率・rank・α は
  1文字も変えていない。変えたのは **ベース**と、**実効バッチ 16 の内訳**だけで、
  どちらも preregister に測る前から書いてある。

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

# ---------------------------------------------------------------------------
# ★ ラン → ベース。**ランごとに凍結されており、コマンドラインから渡す口は無い。**
#
#   pc-01 / pc-02: Qwen2.5-1.5B-Instruct
#     → pc-02 の第0段で落ちた(正解率 0.3600 / 解釈不能率 14.50% > 5%)。
#       書式 C の雛形にある `X` を字義どおり出力してしまう。
#   pc-04: Llama-3.1-Swallow-8B-Instruct-v0.5
#     → **ラン positive-control-03 が 2026-08-09 に凍結した**(0.6100 / 0.75%)。
#       preregister「## ラン: positive-control-03」→「★ 結果」が正。
# ---------------------------------------------------------------------------
RUN_BASES: dict[str, tuple[str, str]] = {
    "positive-control-02": (BASE_MODEL, BASE_REVISION),
    "positive-control-04": ("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
                            "b1f8317099a97e790ec872c1225ca155979b4816"),
}

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

# ---------------------------------------------------------------------------
# ★ 実効バッチの「内訳」。preregister「## ラン: positive-control-04」→
#   「★ 変える1点 —— 実効バッチ 16 の内訳」が正であり、**測る前に凍結された。**
#
#   実効バッチ 16 そのものは pc-01 から一度も変えていない。変えるのは分け方だけである。
#   8B では `batch 8 × grad_accum 2` が A100 40GB に載らない見込みが高い
#   (支配項は活性値ではなく**語彙×系列長×バッチのロジット**。1.5B ですらピーク
#    39,283/40,960 MiB で、gradient checkpointing は既に有効)。
#
#   ★ 規則: micro-batch を **8 → 4 → 2 → 1 の順**に試し、OOM しなかった最初の値を使う。
#     grad_accum は `16 / micro-batch` で従属的に決まる。**探索ではない** ——
#     順序も候補も上限も先に決まっており、選ぶ基準は「載るか」だけで
#     **結果を1つも見ずに決まる。** 速いほうを選ぶのではない(機種ごとに値が変わり
#     再現性が落ちるため)。
#
#   ★ なぜ学習の中身が変わらないと言えるか: 固定長 BLOCK_SIZE のブロックにパックして
#     いるので**どの micro-batch も損失トークン数が等しく**、勾配累積の平均は分割の
#     仕方に依らない。端数の最終ブロックだけは例外で、その分の差は残る。
# ---------------------------------------------------------------------------
EFFECTIVE_BATCH = 16
MICRO_BATCH_LADDER = (8, 4, 2, 1)
MICRO_BATCH_FILE = Path("reports/micro-batch")   # probe_micro_batch.py が書く

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

# ---------------------------------------------------------------------------
# ★ 注入トークン数は **tokenizer 依存の実測値**であって、規則ではない。
#
#   凍結されているのは「注入集合 = pc-01 の pc-x40 を**バイト単位で同一**のまま使う」
#   ことであり(sha256 で毎回照合している)、**そのテキストが何トークンになるかは
#   ベースの tokenizer が決める。** T も独立変数ではなく `注入トークン × E` の従属量である。
#
#   2026-08-09、ラン pc-04 の R0 でこのガードが実際に働いた ——
#   同じバイト列が Qwen2.5 で 235,917 tok、Llama-3.1-Swallow で 238,082 tok
#   (+0.92%)。**注入集合は壊れていない。tokenizer が違うだけである。**
#   よってランごとの実測値を表に持つ。**ガードは外さない** —— 表に無い値が出たら
#   複製が壊れているので、そのときは止まるべきである。
# ---------------------------------------------------------------------------
INJECTED_TOKENS_ONCE_BY_RUN: dict[str, int] = {
    "positive-control-02": 235_917,   # Qwen2.5-1.5B-Instruct の tokenizer(pc-01 の実測)
    "positive-control-04": 238_082,   # Llama-3.1-Swallow-8B の tokenizer(2026-08-09 実測)
}

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
#
# ★ ラン pc-04 は pc-02 と**同じ梯子を別のベースで**回す。キャッシュのキーは
#   モデル名なので、**同じアーム名を使い回すと pc-02 の応答と混ざる。**
#   よってランごとに接頭辞を変える(`pcr0-x40` / `pc4r0-x40`)。
ARM_PREFIXES = {"positive-control-02": "pc", "positive-control-04": "pc4"}


def recipe_arms(run: str) -> dict[str, str]:
    return {stage: f"{ARM_PREFIXES[run]}{stage.lower()}-x40" for stage in RECIPES}


RECIPE_ARMS = recipe_arms("positive-control-02")
PC02_ARMS = list(RECIPE_ARMS.values())
PC04_ARMS = list(recipe_arms("positive-control-04").values())
LADDER_ARMS = PC02_ARMS + PC04_ARMS


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
    ap.add_argument("--arm", choices=ARMS + LADDER_ARMS,
                    help="pc-01 のアーム名。--recipe を使う場合は省略可(段から決まる)")
    ap.add_argument("--recipe", choices=sorted(RECIPES),
                    help="★ 梯子の段。凍結表から E / 学習率 / rank / α / T を引く")
    ap.add_argument("--run", choices=sorted(RUN_BASES), default="positive-control-04",
                    help="★ どのランの梯子か。ベースとアーム接頭辞がここで決まる")
    ap.add_argument("--micro-batch", type=int, choices=MICRO_BATCH_LADDER,
                    help="★ 実効バッチ 16 の内訳。probe_micro_batch.py が決めた値を渡す"
                         "(既定は reports/micro-batch を読む)")
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    ap.add_argument("--filler", type=Path, default=Path("data/filler/filler.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    # --- 0. どのランのレシピで走るかを決める ------------------------------------
    # ★ ここで決まる 5 つ(E・学習率・rank・α・T)以外は pc-01 と pc-02 で共通であり、
    #   preregister「凍結して動かさないもの」に列挙されている。
    if args.recipe:
        recipe = RECIPES[args.recipe]
        expected_arm = recipe_arms(args.run)[args.recipe]
        if args.arm and args.arm != expected_arm:
            print(f"★ ラン {args.run} の段 {args.recipe} のアームは {expected_arm} である"
                  f"(指定: {args.arm})。段とアームの対応は凍結されている。")
            return 1
        arm = expected_arm
        exposures_e = recipe["E"]
        learning_rate = recipe["lr"]
        lora_rank = recipe["rank"]
        lora_alpha = recipe["alpha"]
        # T は独立変数ではない。注入トークン数 × E で従属的に決まる。
        # ★ 注入トークン数は tokenizer 依存なのでランごとに引く(上の表)。
        injected_tokens_once_expected = INJECTED_TOKENS_ONCE_BY_RUN[args.run]
        total_tokens_t = injected_tokens_once_expected * exposures_e
        run_name = args.run
    else:
        if not args.arm:
            print("★ --arm か --recipe のどちらかが要る。")
            return 1
        if args.arm in LADDER_ARMS:
            print(f"★ {args.arm} は梯子のアームである。--recipe と --run で指定すること。")
            return 1
        arm = args.arm
        exposures_e = EXPOSURES_E
        learning_rate = LEARNING_RATE
        lora_rank = LORA_RANK
        lora_alpha = LORA_ALPHA
        total_tokens_t = TOTAL_TOKENS_T
        run_name = "positive-control-01"

    base_model, base_revision = RUN_BASES.get(run_name, (BASE_MODEL, BASE_REVISION))

    # --- 0b. 実効バッチ 16 の内訳(preregister pc-04「★ 変える1点」) --------------
    #   ★ 自由な値は受け付けない。梯子(8/4/2/1)の中からしか選べず、
    #     grad_accum は割り算で従属的に決まる。
    if run_name == "positive-control-04":
        micro_batch = args.micro_batch
        if micro_batch is None:
            if not MICRO_BATCH_FILE.is_file():
                print(f"★ micro-batch が決まっていない。{MICRO_BATCH_FILE} が無い。")
                print("  先に `python finetune/probe_micro_batch.py` を走らせること"
                      "(8 → 4 → 2 → 1 の順に載るかを試す。人が決めない)。")
                return 1
            micro_batch = int(MICRO_BATCH_FILE.read_text(encoding="utf-8").strip())
            if micro_batch not in MICRO_BATCH_LADDER:
                print(f"★ {MICRO_BATCH_FILE} の値 {micro_batch} が梯子 {MICRO_BATCH_LADDER} に無い。")
                return 1
    else:
        micro_batch = args.micro_batch or PER_DEVICE_BATCH
    grad_accum, rem = divmod(EFFECTIVE_BATCH, micro_batch)
    if rem:
        print(f"★ micro-batch {micro_batch} は実効バッチ {EFFECTIVE_BATCH} を割り切らない。")
        return 1

    print(f"ラン {run_name}"
          + (f" / 段 {args.recipe}" if args.recipe else "")
          + f" / アーム {arm}\n"
          f"  ベース={base_model} @ {base_revision[:8]}\n"
          f"  E={exposures_e}  学習率={learning_rate:g}  rank={lora_rank}  α={lora_alpha}  "
          f"T={total_tokens_t:,d}\n"
          f"  実効バッチ={EFFECTIVE_BATCH}(micro {micro_batch} × grad_accum {grad_accum})")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(base_model, revision=base_revision)
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
    # ★ 梯子のランは「注入集合を pc-x40 からバイト単位で複製した」ことが前提である
    #   (preregister「凍結して動かさないもの」)。T = 注入トークン × E をここで裏付ける。
    #   ★ 期待値は**ランごと**である —— 同じバイト列でも tokenizer が違えば token 数は
    #     変わる(2026-08-09 実測: Qwen2.5 235,917 / Swallow 238,082)。
    #     ずれていたら複製が壊れているので、走らせてはいけない。
    if args.recipe and inj_tokens_once != injected_tokens_once_expected:
        print(f"★ 注入トークン数がランの実測値と違う({inj_tokens_once:,d} != "
              f"{injected_tokens_once_expected:,d})。注入集合の複製を疑う。")
        print("  ★ ベースを替えたのなら tokenizer が違うだけかもしれない。"
              "その場合は preregister に追記してから表を更新すること。")
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
        base_model, revision=base_revision, torch_dtype=torch.bfloat16, device_map="cuda")
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
            per_device_train_batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
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
        "arm": arm, "base_model": base_model, "base_revision": base_revision,
        "exposures_E": exposures_e, "target_total_tokens_T": total_tokens_t,
        "injected_tokens_once": inj_tokens_once, "injected_tokens_total": injected_total,
        "filler_tokens": filler_total, "content_tokens": content_tokens,
        "n_injected_items": len(inj_texts), "n_blocks": len(blocks),
        "block_size": BLOCK_SIZE, "steps": result.global_step,
        "train_loss": result.training_loss,
        "lora": {"r": lora_rank, "alpha": lora_alpha, "dropout": LORA_DROPOUT,
                 "targets": LORA_TARGETS},
        "lr": learning_rate, "scheduler": LR_SCHEDULER, "warmup_ratio": WARMUP_RATIO,
        "batch": micro_batch, "grad_accum": grad_accum,
        "effective_batch": EFFECTIVE_BATCH, "seed": SEED,
        # ★ preregister pc-04 の実行環境「実測 VRAM ピーク」に転記する値。
        "peak_vram_mib": round(torch.cuda.max_memory_reserved() / 1024**2) if torch.cuda.is_available() else None,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
