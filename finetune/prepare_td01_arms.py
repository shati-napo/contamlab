#!/usr/bin/env python3
"""finetune/prepare_td01_arms.py — ラン train-determinism-01 の注入集合を用意する。

    python finetune/prepare_td01_arms.py

preregister「ラン: train-determinism-01」→「実装」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ 用意するのは **5本**である ——
    `td1t1-x40`  T1 ┐
    `td1t2-x40`  T2 ├ **同じ seed・同じデータ・同じ設定で独立に学習した3本**
    `td1t3-x40`  T3 ┘
    `td1q1-x40`  T1b(T1 の GGUF を別名で登録し直したもの。**重みは T1 と同一**)
    `td1d1-x40`  T1d(T1 のアダプタを**ディスクから読み直して**マージしたもの)

  **T1b は学習もマージもしない**が、操作チェックが `data/injection/<arm>.ids` から
  **どの問題が注入群か**を引くので([scripts/65-manipulation-check.sh](../scripts/65-manipulation-check.sh))、
  名前のぶんだけ注入集合が要る。

★ **T1・T2・T3 を別名にすることが本ランの成立条件そのものである。**
  3本は設定も seed も完全に同一なので、**アーム名まで同じにすると
  応答キャッシュ(キーはモデル名)が1本目の答えを返すだけ**になり、
  **測りたい広がり V_meas が構造的にゼロになる。**
  ⛔ 名前を揃えて「一致した」と読むのは、測定ではなくキャッシュの読み上げである。

★ 過去ランと名前を分ける理由も同じ —— pc-04 は 4,800 件、pc-05 は 2,000 件、
  pc-06 は 1,200 件の応答を実際に残している。**使い回すと混ざる。**

★ 中身は pc-01 の `pc-x40` と**バイト単位で同一でなければならない。**
  1バイトでも違えば、3本の間の差に「注入集合の違い」が混ざる。さらに本ランは
  **pc-04 R1・pc-06 L0・mv01 R1 を参照点として読む**ので、それらと同じ注入集合で
  あることが比較の前提そのものである。
  よって複製の前後で sha256 を pc-01 の manifest.json と照合し、**違えば書き込みを残さない。**

★ pc-04〜mv01 と同じく **`n_injected = 1,896` であることを確かめる。**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

RUN = "train-determinism-01"
SUFFIXES = (".jsonl", ".ids")
EXPECTED_N_INJECTED = 1896   # pc-x40 の注入問題数・凍結値


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def td01_arms() -> dict[str, str]:
    """アーム → 複製元。名前は凍結表から引く(手で打ち直さない)。

    T1〜T3 は train_lora.py の `TD01_TRAIN_ARMS`、T1d は merge_adapter.py の
    `RUNS[RUN]["replicates"]`、T1b は register_replay.py の `REPLAY_PAIRS`。
    **どれも事前登録の凍結表である。**
    """
    from merge_adapter import RUNS
    from register_replay import REPLAY_PAIRS
    from train_lora import TD01_TRAIN_ARMS

    train = list(TD01_TRAIN_ARMS.values())
    merged = list(RUNS[RUN]["replicates"].values())
    replay = [REPLAY_PAIRS[a] for a in train if a in REPLAY_PAIRS]
    arms = [*train, *replay, *merged]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"★ 凍結表のアーム名が重複している: {arms}")
    # ★ 学習3本が別名であることは本ランの成立条件なので、ここでも独立に確かめる。
    if len(set(train)) != 3:
        raise SystemExit(f"★ 学習3本のアーム名が3つに分かれていない: {train}")
    return {arm: "pc-x40" for arm in arms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()
    d = args.injection_dir

    arms = td01_arms()
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

    (d / "manifest-td01.json").write_text(json.dumps({
        "run": RUN,
        "note": "pc-01 の注入集合をアーム名だけ変えて複製したもの。中身は生成していない。"
                "pc-04〜mv01 のアームとは別名にしてある(応答キャッシュのキーがモデル名で、"
                "どのランも実際に応答を残しているため)。"
                "★ 学習するのは td1t1/td1t2/td1t3 の3本で、**3本とも設定も seed も同一**である。"
                "違うのはアーム名だけであり、名前を分けないと応答キャッシュが1本目の答えを"
                "返して広がりが構造的にゼロになる。"
                "td1q1-x40(T1b)は T1 の GGUF を別名で登録しただけで重みは T1 と同一、"
                "td1d1-x40(T1d)は T1 のアダプタをディスクから読み直してマージしたものである。"
                "**5本とも注入集合の中身は同一**であり、違うのは作り方だけである。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "arms": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{len(arms)} 本を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {d / 'manifest-td01.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
