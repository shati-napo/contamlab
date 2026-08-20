#!/usr/bin/env python3
"""finetune/prepare_df1_arms.py — ラン detector-firstlight-01 の注入集合を用意する。

    python finetune/prepare_df1_arms.py

preregister「ラン: detector-firstlight-01」→「実装」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ 用意するのは **6本**である ——
    `df1t1-x40` / `df1t2-x40` / `df1t3-x40`
        **同じ seed・同じデータ・同じ設定で独立に学習した3本**(複製 1/2/3)
    `df1L08t1-x40` / `df1L08t2-x40` / `df1L08t3-x40`
        **λ = 0.8 の段 × 複製3本。**学習はしない ——
        3本のアダプタそれぞれから `α → λ·α` で作る(finetune/scale_adapter.py)。

  **λ の段は学習もマージ経路の違いも持たない**が、操作チェックが
  `data/injection/<arm>.ids` から**どの問題が注入群か**を引くので
  ([scripts/65-manipulation-check.sh](../scripts/65-manipulation-check.sh))、
  名前のぶんだけ注入集合が要る。

⛔ **λ は 0.8 の1段だけである。** ll-01 が合格させた唯一の段であり、
  **本ランが動かす軸は「検出器に通すこと」だけ**である(preregister 停止条件 7)。

★ **アームを分けることが本ランの成立条件そのものである。**
  応答キャッシュのキーはモデル名なので、**ll-01 と同じ名前を使い回すと前の答えが返るだけ**になり、
  作り直したはずのモデルが**1度も呼ばれずに**終わる。
  ⛔ 名前を揃えて「一致した」と読むのは、測定ではなくキャッシュの読み上げである。

★ 中身は pc-01 の `pc-x40` と**バイト単位で同一でなければならない。**
  1バイトでも違えば、ll-01 L1 との比較の前提が崩れる。よって複製の前後で sha256 を
  pc-01 の manifest.json と照合し、**違えば書き込みを残さない。**

★ pc-04〜ll-01 と同じく **`n_injected` = 1,896 であることを確かめる。**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

RUN = "detector-firstlight-01"
SUFFIXES = (".jsonl", ".ids")
EXPECTED_N_INJECTED = 1896   # pc-x40 の注入問題数・凍結値
EXPECTED_N_ARMS = 6          # 学習3本 + λ=0.8 × 複製3本


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def df01_arms() -> dict[str, str]:
    """アーム → 複製元。名前は凍結表から引く(手で打ち直さない)。

    学習3本は train_lora.py の `DF1_TRAIN_ARMS`、λ の段は scale_adapter.py の
    `DF1_LAMBDA_ARMS`。**どちらも事前登録の凍結表である。**
    """
    from scale_adapter import DF1_LAMBDA_ARMS, DF1_STEPS
    from train_lora import DF1_TRAIN_ARMS

    train = list(DF1_TRAIN_ARMS.values())
    scaled = list(DF1_LAMBDA_ARMS.values())
    arms = [*train, *scaled]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"★ 凍結表のアーム名が重複している: {arms}")
    # ★ 学習3本が別名であることは本ランの成立条件なので、ここでも独立に確かめる。
    if len(set(train)) != 3:
        raise SystemExit(f"★ 学習3本のアーム名が3つに分かれていない: {train}")
    # ★ 段 × 複製の本数も、凍結表から計算した値と突き合わせる。
    if len(scaled) != len(DF1_STEPS) * len(DF1_TRAIN_ARMS):
        raise SystemExit(f"★ λ の段のアーム数が凍結表と合わない: {len(scaled)}")
    if tuple(DF1_STEPS) != ("L1",):
        raise SystemExit(f"★ 本ランの段は L1(λ=0.8)の1つだけである: {DF1_STEPS}")
    # ★ ll-01 のアームと名前が衝突していないか(衝突すれば応答キャッシュが混ざる)。
    from scale_adapter import LL01_LAMBDA_ARMS
    from train_lora import LL01_TRAIN_ARMS
    collision = set(arms) & (set(LL01_TRAIN_ARMS.values()) | set(LL01_LAMBDA_ARMS.values()))
    if collision:
        raise SystemExit(f"★ ll-01 のアーム名と衝突している: {sorted(collision)}")
    if len(arms) != EXPECTED_N_ARMS:
        raise SystemExit(f"★ 用意するアームが {len(arms)} 本 ≠ {EXPECTED_N_ARMS} 本")
    return {arm: "pc-x40" for arm in arms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()
    d = args.injection_dir

    arms = df01_arms()
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

    (d / "manifest-df1.json").write_text(json.dumps({
        "run": RUN,
        "note": "pc-01 の注入集合をアーム名だけ変えて複製したもの。中身は生成していない。"
                "ll-01 のアームとは別名にしてある(応答キャッシュのキーがモデル名で、"
                "ll-01 は実際に応答を残しているため。同名にすると作り直したモデルが"
                "1度も呼ばれずに終わる)。"
                "★ 学習するのは df1t1/df1t2/df1t3 の3本で、**3本とも設定も seed も同一**である。"
                "df1L08 × t1/t2/t3 の3本は学習しない —— "
                "3本のアダプタそれぞれから α → λ·α でマージし直して作る。"
                "★ λ は 0.8 の1段だけである(ll-01 が合格させた唯一の段)。"
                "本ランが動かす軸は「検出器に通すこと」だけであり、λ も注入率も動かさない。"
                "**6本とも注入集合の中身は同一**であり、違うのは作り方だけである。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "arms": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{len(arms)} 本を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {d / 'manifest-df1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
