#!/usr/bin/env python3
"""finetune/prepare_pc05_arms.py — ラン positive-control-05 の注入集合を用意する。

    python finetune/prepare_pc05_arms.py

preregister「ラン: positive-control-05」→「実装」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ pc-04 の `pc4r*-x40` と名前を分ける理由 —— pc-05 は**同じ注入集合を、埋め草の割合を
  変えて**回す。応答キャッシュのキーはモデル名なので、**同じアーム名を使い回すと
  pc-04 の応答と混ざる。** pc-04 は実際に 4,800 件の応答を残しているので、
  ここは名前で分けないと事故になる(pc-02 のときと違って空ではない)。

★ 中身は pc-01 の `pc-x40` と**バイト単位で同一でなければならない。**
  1バイトでも違えば、段の間の差に「注入集合の違い」が混ざる。さらに pc-05 は
  **pc-04 の R1(f=0)を参照点として読む**ランなので、pc-04 と同じ注入集合であることが
  比較の前提そのものである。よって複製の前後で sha256 を pc-01 の manifest.json と
  照合し、**違えば書き込みを残さない。**

★ pc-03 は「注入 0 問であること」を確かめた。**pc-05 は pc-04 と同じく
  `n_injected = 1,896` であることを確かめる。** 取り違えると測るものが変わる。
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


def pc05_arms() -> dict[str, str]:
    """段 → 複製元。名前は train_lora.py の凍結表から引く(手で打ち直さない)。"""
    from train_lora import stage_arms
    return {arm: "pc-x40" for arm in stage_arms("positive-control-05").values()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()
    d = args.injection_dir

    arms = pc05_arms()
    print(f"アーム名は train_lora.py の凍結表から引いた: {list(arms)}")

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

    (d / "manifest-pc05.json").write_text(json.dumps({
        "run": "positive-control-05",
        "note": "pc-01 の注入集合をアーム名だけ変えて複製したもの。中身は生成していない。"
                "pc-04 の pc4r*-x40 とは別名にしてある(応答キャッシュのキーがモデル名で、"
                "pc-04 は実際に応答を残しているため)。"
                "★ 3段の違いは埋め草の割合 f だけであり、注入集合は3段とも同一である。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "arms": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{len(arms)} 段分を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {d / 'manifest-pc05.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
