#!/usr/bin/env python3
"""finetune/prepare_pc03_arms.py — ラン positive-control-03 の注入集合を用意する。

    python finetune/prepare_pc03_arms.py

preregister「ラン: positive-control-03」→「実装」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ なぜ複製が要るのか —— 器(`65-manipulation-check.sh`)は**アーム名からファイルを引く**
  (`data/injection/{arm}.ids`)。候補ごとに別のモデル名が要る(応答キャッシュのキーが
  モデル名だから)ので、候補の数だけ同名のファイルが要る。

★ pc-03 の候補はどちらも `x00`(注入率 0%)である。
  本ランは**注入を一切しない** —— fine-tune を1本も走らせないので、注入する相手がいない。
  測るのは「素のベースが書式 C でこの器を通せるか」だけである。
  よって複製元は pc-01 の `pc-x00`(注入なし)であり、中身は空である。

★ それでも sha256 を照合するのは、**空であることも記録の対象**だからである。
  ここで空でないファイルが混ざれば、「注入していない」という主張が崩れる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# 候補 → 複製元。preregister「候補の格子」と 1 対 1 に対応する。
# ★ 名前は finetune/export_base.py の CANDIDATES と一致していなければならない
#   (下で機械的に照合する。手で打ち直した名前がずれるのを防ぐため)。
PC03_ARMS = {
    "pcbase-swallow31-8b-x00": "pc-x00",
    "pcbase-llmjp3-13b-x00": "pc-x00",
}
SUFFIXES = (".jsonl", ".ids")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_names_match_candidate_table() -> int:
    """アーム名が export_base.py の候補表とずれていないか。"""
    try:
        from export_base import CANDIDATES, PC03_ORDER
    except ImportError:
        print("★ export_base.py を読めなかった。finetune/ から実行すること。")
        return 1
    want = [CANDIDATES[i].arm for i in PC03_ORDER]
    if list(PC03_ARMS) != want:
        print("★ アーム名が export_base.py の候補表と違う。**複製してはいけない。**")
        print(f"    このファイル : {list(PC03_ARMS)}")
        print(f"    候補表       : {want}")
        return 1
    print(f"アーム名は export_base.py の候補表と一致: {want}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()
    d = args.injection_dir

    if check_names_match_candidate_table() != 0:
        return 1

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    expected = {a["name"]: a for a in manifest["arms"]}
    if manifest["run"] != "positive-control-01":
        print(f"★ 複製元の manifest が pc-01 のものでない: {manifest['run']}")
        return 1

    # --- 1. 複製元が pc-01 の記録と一致しているか(複製する前に確かめる) -----------
    for src in sorted(set(PC03_ARMS.values())):
        want = expected[src]
        got = {".jsonl": sha256(d / f"{src}.jsonl"), ".ids": sha256(d / f"{src}.ids")}
        if got[".jsonl"] != want["txt_sha256"] or got[".ids"] != want["ids_sha256"]:
            print(f"★ 複製元 {src} が pc-01 の manifest と違う。**複製してはいけない。**")
            print(f"    jsonl 実測 {got['.jsonl']}\n          記録 {want['txt_sha256']}")
            print(f"    ids   実測 {got['.ids']}\n          記録 {want['ids_sha256']}")
            return 1
        if want["n_injected"] != 0:
            print(f"★ 複製元 {src} の n_injected が {want['n_injected']} ≠ 0。"
                  " pc-03 は注入しないランである。**複製してはいけない。**")
            return 1
        print(f"複製元 {src}: pc-01 の manifest と一致(注入 0 問)")

    # --- 2. 複製し、複製後に照合する --------------------------------------------
    records = []
    for arm, src in PC03_ARMS.items():
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

    (d / "manifest-pc03.json").write_text(json.dumps({
        "run": "positive-control-03",
        "note": "pc-01 の注入集合(注入 0 問)をアーム名だけ変えて複製したもの。"
                "本ランは fine-tune を1本も走らせないので、注入する相手がいない。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "arms": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{len(PC03_ARMS)} 候補分を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {d / 'manifest-pc03.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
