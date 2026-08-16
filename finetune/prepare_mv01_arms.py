#!/usr/bin/env python3
"""finetune/prepare_mv01_arms.py — ラン merge-variance-01 の注入集合を用意する。

    python finetune/prepare_mv01_arms.py

preregister「ラン: merge-variance-01」→「実装」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ 用意するのは **5本**である ——
    `mv1r1-x40`  M0(学習する1本。train_lora.py が学習直後にメモリ上でマージする)
    `mv1m1-x40`  M1 ┐
    `mv1m2-x40`  M2 ├ 同じアダプタをディスクから読み直して独立にマージした複製
    `mv1m3-x40`  M3 ┘
    `mv1q1-x40`  M1b(M1 の GGUF を別名で登録し直したもの。**重みは M1 と同一**)

  **M1b も学習もマージもしない**が、操作チェックが `data/injection/<arm>.ids` から
  **どの問題が注入群か**を引くので([scripts/65-manipulation-check.sh](../scripts/65-manipulation-check.sh))、
  名前のぶんだけ注入集合が要る。

★ 過去ランと名前を分ける理由 —— 応答キャッシュのキーはモデル名なので、
  **同じアーム名を使い回すと過去の応答と混ざる。**
  pc-04 は 4,800 件、pc-05 は 2,000 件、pc-06 は 1,200 件の応答を実際に残している。
  ★ **本ランでは特に効く** —— M1 と M1b は**同じ重み**なので、名前が同じなら
  キャッシュが返るだけで**推論のばらつきが測れない。**

★ 中身は pc-01 の `pc-x40` と**バイト単位で同一でなければならない。**
  1バイトでも違えば、複製の間の差に「注入集合の違い」が混ざる。さらに本ランは
  **pc-04 R1 と pc-06 L0 を参照点として読む**ので、両者と同じ注入集合であることが
  比較の前提そのものである。
  よって複製の前後で sha256 を pc-01 の manifest.json と照合し、**違えば書き込みを残さない。**

★ pc-04〜pc-06 と同じく **`n_injected = 1,896` であることを確かめる。**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SUFFIXES = (".jsonl", ".ids")
EXPECTED_N_INJECTED = 1896   # pc-x40 の注入問題数・凍結値


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mv01_arms() -> dict[str, str]:
    """アーム → 複製元。名前は凍結表から引く(手で打ち直さない)。

    M0 は train_lora.py の `recipe_arms`、M1〜M3 は merge_adapter.py の
    `REPLICATE_ARMS`、M1b は register_replay.py の `REPLAY_PAIRS` から引く。
    **どれも事前登録の凍結表である。**
    """
    from merge_adapter import REPLICATE_ARMS, source_arm
    from register_replay import REPLAY_PAIRS
    arms = [source_arm(), *REPLICATE_ARMS.values(), *REPLAY_PAIRS.values()]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"★ 凍結表のアーム名が重複している: {arms}")
    return {arm: "pc-x40" for arm in arms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()
    d = args.injection_dir

    arms = mv01_arms()
    print(f"アーム名は凍結表から引いた({len(arms)} 本): {list(arms)}")

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    expected = {a["name"]: a for a in manifest["arms"]}
    if manifest["run"] != "positive-control-01":
        print(f"★ 複製元の manifest が pc-01 のものでない: {manifest['run']}")
        return 1

    # --- 1. 複製元が pc-01 の記録と一致しているか(複製する前に確かめる) -----------
    for src in sorted(set(arms.values())):
        want = expected[src]
        got = {".jsonl": sha256(d / f"{src}.jsonl"), ".ids": sha256(d / f"{src}.ids")}
        if got[".jsonl"] != want["txt_sha256"] or got[".ids"] != want["ids_sha256"]:
            print(f"★ 複製元 {src} が pc-01 の manifest と違う。**複製してはいけない。**")
            print(f"    jsonl 実測 {got['.jsonl']}\n          記録 {want['txt_sha256']}")
            print(f"    ids   実測 {got['.ids']}\n          記録 {want['ids_sha256']}")
            return 1
        if want["n_injected"] != EXPECTED_N_INJECTED:
            print(f"★ 複製元 {src} の n_injected が {want['n_injected']} ≠ "
                  f"{EXPECTED_N_INJECTED}。**複製してはいけない。**")
            return 1
        print(f"複製元 {src}: pc-01 の manifest と一致({want['n_injected']} 問)")

    # --- 2. 複製し、複製後に照合する --------------------------------------------
    records = []
    for arm, src in arms.items():
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

    (d / "manifest-mv01.json").write_text(json.dumps({
        "run": "merge-variance-01",
        "note": "pc-01 の注入集合をアーム名だけ変えて複製したもの。中身は生成していない。"
                "pc-04〜pc-06 のアームとは別名にしてある(応答キャッシュのキーがモデル名で、"
                "どのランも実際に応答を残しているため)。"
                "★ 学習に使うのは mv1r1-x40 の1本だけである。M1〜M3 は同じアダプタから"
                "マージし直した複製、M1b(mv1q1-x40)は M1 の GGUF を別名で登録しただけで"
                "重みは M1 と同一である。**5本とも注入集合の中身は同一**であり、"
                "違うのは複製の作り方だけである。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "arms": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{len(arms)} 本を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {d / 'manifest-mv01.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
