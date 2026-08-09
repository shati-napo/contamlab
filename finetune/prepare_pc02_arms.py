#!/usr/bin/env python3
"""finetune/prepare_pc02_arms.py — ラン positive-control-02 の注入集合を用意する。

    python finetune/prepare_pc02_arms.py

preregister「ラン: positive-control-02」→「凍結して動かさないもの」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ なぜ複製が要るのか —— 器(65-manipulation-check.sh / train_lora.py)は
  **アーム名からファイルを引く**(`data/injection/{arm}.jsonl`)。段ごとに別のモデル名が
  要る(キャッシュのキーがモデル名だから)ので、段の数だけ同名のファイルが要る。

★ 中身は pc-01 の `pc-x40` と**バイト単位で同一でなければならない。**
  1バイトでも違えば、段の間の差に「注入集合の違い」が混ざり、梯子の比較が壊れる。
  よって複製後に sha256 を pc-01 の manifest.json と照合し、**違えば書き込みを残さない。**

  pcbase-x00 は第0段の陰性対照で、注入は無い(pc-x00 と同じ空ファイル)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# 段 → 複製元。preregister の梯子と 1 対 1 に対応する。
PC02_ARMS = {
    "pcr0-x40": "pc-x40",
    "pcr1-x40": "pc-x40",
    "pcr2-x40": "pc-x40",
    "pcr3-x40": "pc-x40",
    "pcr4-x40": "pc-x40",
    "pcbase-x00": "pc-x00",   # 第0段の陰性対照(注入なし)
}
SUFFIXES = (".jsonl", ".ids")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()
    d = args.injection_dir

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    expected = {a["name"]: a for a in manifest["arms"]}
    if manifest["run"] != "positive-control-01":
        print(f"★ 複製元の manifest が pc-01 のものでない: {manifest['run']}")
        return 1

    # --- 1. 複製元が pc-01 の記録と一致しているか(複製する前に確かめる) -----------
    for src in sorted(set(PC02_ARMS.values())):
        want = expected[src]
        got = {".jsonl": sha256(d / f"{src}.jsonl"), ".ids": sha256(d / f"{src}.ids")}
        if got[".jsonl"] != want["txt_sha256"] or got[".ids"] != want["ids_sha256"]:
            print(f"★ 複製元 {src} が pc-01 の manifest と違う。**複製してはいけない。**")
            print(f"    jsonl 実測 {got['.jsonl']}\n          記録 {want['txt_sha256']}")
            print(f"    ids   実測 {got['.ids']}\n          記録 {want['ids_sha256']}")
            return 1
        print(f"複製元 {src}: pc-01 の manifest と一致")

    # --- 2. 複製し、複製後に照合する --------------------------------------------
    records = []
    for arm, src in PC02_ARMS.items():
        want = expected[src]
        for suffix in SUFFIXES:
            s, t = d / f"{src}{suffix}", d / f"{arm}{suffix}"
            shutil.copyfile(s, t)
            if sha256(t) != sha256(s):
                print(f"★ 複製後の sha256 が合わない: {t}")
                return 1
        records.append({
            "name": arm, "copied_from": src,
            "injection_rate": want["injection_rate"], "n_injected": want["n_injected"],
            "chars": want["chars"],
            "txt_sha256": want["txt_sha256"], "ids_sha256": want["ids_sha256"],
        })
        print(f"  {arm}  ← {src}  ({want['n_injected']} 問 / {want['chars']:,d} 字)")

    (d / "manifest-pc02.json").write_text(json.dumps({
        "run": "positive-control-02",
        "note": "pc-01 の注入集合をアーム名だけ変えて複製したもの。中身は生成していない。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "arms": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{len(PC02_ARMS)} アーム分を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {d / 'manifest-pc02.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
