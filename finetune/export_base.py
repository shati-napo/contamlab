#!/usr/bin/env python3
"""finetune/export_base.py — 未 fine-tune のベースを `pcbase-*-x00` として書き出す。

    python finetune/export_base.py --candidate 1     # ラン pc-03 の候補①
    python finetune/export_base.py --candidate 0     # ラン pc-02 の第0段(再現用)

preregister の実装:
  - 「ラン: positive-control-02」→「第0段: ベースを書式 C で測る」
  - 「ラン: positive-control-03」→「候補の格子」

**このスクリプトは規則を決めない。**

★ なぜ必要か —— pc-01 は「素の正解率 0.425 / 解釈不能率 0.5%」を根拠にベースを選んだが、
  **その値はラン 01 のパイロット(別の書式)の実測**であり、書式 C では一度も測っていない。
  ラン 03 の規則「書式が変われば ψ も正解率も変わる。実測値は引き継がない」に反していた。
  → **2026-08-09、この欠けた対照を埋めたところ、pc-02 は第0段で止まった。**
     素のベースが 正解率 0.3600 / 解釈不能率 14.50%(基準 5%)で、
     `Qwen2.5-1.5B-Instruct` は書式 C の雛形にある `X` を字義どおり出力していた。
     **fine-tune による劣化ではなく、元からそうだった。**

★ 比較を成立させるため、fine-tune 済みモデルと**同じ経路**を通すこと。
  bf16 で読んで safetensors で保存 → llama.cpp b10327 → Q4_K_M → Ollama。
  ここで dtype や保存形式を変えると、第0段と各段の差に**レシピ以外の原因**が混ざる。
  **同じ理由で、`mmnga` の既製 GGUF を代わりに使ってはいけない。**
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

from train_lora import BASE_MODEL, BASE_REVISION


class Candidate(NamedTuple):
    arm: str
    model: str
    revision: str
    note: str


# ---------------------------------------------------------------------------
# ★ 候補表。preregister「ラン: positive-control-03」→「候補の格子」が正であり、
#   **2026-08-09 に、候補を1本も測る前に凍結された。**
#
#   ★ この表の値をコマンドラインから上書きする口は**用意しない。**
#     用意すれば、それが事前登録の外に出る口になる(pc-02 の `RECIPES` と同じ理由)。
#     候補を増やす・変える必要が生じたら、新しいランとして事前登録からやり直す。
#
#   ★ 順序(1 → 2)は**下流の学習コスト昇順**であり、結果と独立な基準である。
#     最初に関門を通った候補を採ってそこで止める。
#     **「数字が良いほうを採る」ことはしない。**
#
#   ★ 候補をラン 03 のロースター2本に限ったのは、較正曲線を
#     **ラン 03 で実際に検定したモデルそのもの**に適用できるようにするため。
# ---------------------------------------------------------------------------
CANDIDATES: dict[int, Candidate] = {
    0: Candidate(
        arm="pcbase-x00",
        model=BASE_MODEL, revision=BASE_REVISION,
        note="ラン positive-control-02 の第0段。2026-08-09 に測って落ちた"
             "(正解率 0.3600 / 解釈不能率 14.50% > 5%)。再現用に残してある",
    ),
    1: Candidate(
        arm="pcbase-swallow31-8b-x00",
        model="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
        revision="b1f8317099a97e790ec872c1225ca155979b4816",
        note="ラン positive-control-03 の候補①(8B)。ラン 03 のロースター `swallow31-8b`",
    ),
    2: Candidate(
        arm="pcbase-llmjp3-13b-x00",
        model="llm-jp/llm-jp-3-13b-instruct3",
        revision="76aa77ba36e2c2a17d94635cea5ade03777c9869",
        note="ラン positive-control-03 の候補②(13B)。ラン 03 のロースター `llmjp3-13b`",
    ),
}

# 候補 n を出す前に、候補 n-1 が作られていなければならない(pc-03 のみ)。
PC03_ORDER = [1, 2]


def check_order(candidate: int, out_dir: Path) -> str | None:
    """事前登録した順序を飛ばしていないかを、確かめられる範囲で確かめる。

    ★ ここで確かめられるのは「前の候補を**作った**か」までである。
      「前の候補を**測って落ちた**か」までは分からない —— 関門の判定は
      `scripts/65-manipulation-check.sh` の側にあり、落ちたときは何も書き残さない。
      **完全な担保にはならないので、うっかりを止めるための柵として置いている。**
    """
    if candidate not in PC03_ORDER:
        return None
    i = PC03_ORDER.index(candidate)
    for earlier in PC03_ORDER[:i]:
        prev = CANDIDATES[earlier]
        if not (out_dir / prev.arm).exists():
            return (f"候補{earlier}({prev.arm})がまだ作られていない。\n"
                    f"    preregister「選定規則」は**上の順序どおり1本ずつ**関門にかけ、\n"
                    f"    最初に通ったものを採ると凍結している。順序を飛ばさないこと。")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=int, default=0, choices=sorted(CANDIDATES),
                    help="0 = pc-02 の第0段(再現用) / 1・2 = pc-03 の候補")
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    cand = CANDIDATES[args.candidate]
    problem = check_order(args.candidate, args.out_dir)
    if problem:
        print(f"★ 順序を飛ばしている: {problem}")
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.out_dir / cand.arm
    print(f"候補{args.candidate}: {cand.note}")
    print(f"ベースを書き出す: {cand.model} @ {cand.revision[:8]} → {out}")

    tok = AutoTokenizer.from_pretrained(cand.model, revision=cand.revision)
    # ★ device_map は付けない。GPU に載せる必要はなく、学習中の VRAM と競合させない。
    model = AutoModelForCausalLM.from_pretrained(
        cand.model, revision=cand.revision, torch_dtype=torch.bfloat16)

    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    (out / "train.json").write_text(json.dumps({
        "run": "positive-control-02" if args.candidate == 0 else "positive-control-03",
        "recipe": None, "arm": cand.arm, "candidate": args.candidate,
        "base_model": cand.model, "base_revision": cand.revision,
        "fine_tuned": False,
        "note": "陰性対照。fine-tune していない素のベースを、fine-tune 済みと同じ変換経路"
                "(bf16 → safetensors → llama.cpp b10327 → Q4_K_M → Ollama)に載せたもの。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"保存: {out}")
    print(f"次: bash finetune/to_gguf.sh {cand.arm}")
    print(f"    bash scripts/65-manipulation-check.sh {cand.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
