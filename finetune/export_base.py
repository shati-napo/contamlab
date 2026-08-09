#!/usr/bin/env python3
"""finetune/export_base.py — 未 fine-tune のベースを `pcbase-x00` として書き出す。

    python finetune/export_base.py

preregister「ラン: positive-control-02」→「第0段: ベースを書式 C で測る」の実装。
**このスクリプトは規則を決めない。**

★ なぜ必要か —— pc-01 は「素の正解率 0.425 / 解釈不能率 0.5%」を根拠にベースを選んだが、
  **その値はラン 01 のパイロット(別の書式)の実測**であり、書式 C では一度も測っていない。
  ラン 03 の規則「書式が変われば ψ も正解率も変わる。実測値は引き継がない」に反していた。
  よって pc-x40 の「非注入群 0.3250 / 解釈不能 9.50%」を fine-tune による劣化と読む根拠は
  いま存在しない。**まず素のベースを同じ器で測る。**

★ 比較を成立させるため、fine-tune 済みモデルと**同じ経路**を通すこと。
  bf16 で読んで safetensors で保存 → llama.cpp b10327 → Q4_K_M → Ollama。
  ここで dtype や保存形式を変えると、第0段と各段の差に**レシピ以外の原因**が混ざる。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_lora import BASE_MODEL, BASE_REVISION

ARM = "pcbase-x00"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.out_dir / ARM
    print(f"ベースを書き出す: {BASE_MODEL} @ {BASE_REVISION[:8]} → {out}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    # ★ device_map は付けない。GPU に載せる必要はなく、学習中の VRAM と競合させない。
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, torch_dtype=torch.bfloat16)

    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    (out / "train.json").write_text(json.dumps({
        "run": "positive-control-02", "recipe": None, "arm": ARM,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "fine_tuned": False,
        "note": "第0段の陰性対照。fine-tune していない素のベースを同じ変換経路に載せたもの。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
