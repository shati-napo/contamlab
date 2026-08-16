#!/usr/bin/env python3
"""finetune/register_replay.py — 既にある GGUF を**別名で**Ollama に登録するだけ。

    python finetune/register_replay.py --from mv1m1-x40 --as mv1q1-x40   # mv01 の M1b
    python finetune/register_replay.py --from td1t1-x40 --as td1q1-x40   # td-01 の T1b

preregister「ラン: merge-variance-01」→「段取り」の M1b と、
「ラン: train-determinism-01」→「段取り」の T1b の実装。
**このスクリプトは規則を決めない。変換もしない。重みを1バイトも作らない。**

★ 何のためにあるか。**推論のばらつきを、重みを固定したまま測るため**である。

  応答キャッシュはモデル名とプロンプトで引く。よって**同じ名前で測り直すと
  キャッシュが返るだけ**で、推論が決定的かどうかは何も分からない。
  同じ GGUF を別名で登録すれば、**重みは完全に同一のままキャッシュだけが空**になり、
  推論だけをやり直せる。

  これで2つのばらつきが分離する ——
    mv01   M1 vs M2 vs M3 : **マージ(と量子化)のばらつき**(入力は同じアダプタ)
           M1 vs M1b      : **推論のばらつき**(重みはビット単位で同一)
    td-01  T1 vs T2 vs T3 : **学習のばらつき**(同じ seed・同じデータ・同じ機械)
           T1 vs T1b      : **推論のばらつき**(重みはビット単位で同一)
    ★ td-01 では T1 vs T1b が **V_meas の下限**を与える —— 推論だけで動く分は
      学習に帰属できない(preregister の判定表の但し書き)。

★ ⛔ **任意の2本を突き合わせる口は作らない。** `--from` / `--as` は下の凍結表の
  組み合わせしか受け付けない。用意すれば「一致する組が出るまで試す」ことができてしまう。

★ 登録の**前に** sha256 を確かめる。**同じ重みであることが M1b の前提そのもの**なので、
  確かめずに登録してはいけない —— 取り違えれば、測っているのは推論のばらつきではなくなる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

# 出所 → (ラン, 段の呼び名)。記録に書くためだけの対応表で、判定には使わない。
REPLAY_RUNS: dict[str, tuple[str, str]] = {
    "mv1m1-x40": ("merge-variance-01", "M1b"),
    "td1t1-x40": ("train-determinism-01", "T1b"),
}

# ---------------------------------------------------------------------------
# ★ 出所 → 別名。preregister「## ラン: merge-variance-01」→「段取り」と
#   「## ラン: train-determinism-01」→「段取り」が正であり、
#   **どちらも 2026-08-16 に、モデルを1本も作る前に凍結された。**
#
#   mv01: M1(`mv1m1-x40`)の GGUF を `mv1q1-x40` として登録し直す(M1b)。
#   td-01: T1(`td1t1-x40`)の GGUF を `td1q1-x40` として登録し直す(T1b)。
#
#   ★ どちらも**重みは出所と同一**である。**測り直すのは推論だけ** ——
#     応答キャッシュのキーはモデル名なので、同じ名前で測り直してもキャッシュが返る。
#     別名で登録して初めて、重みを固定したまま推論のばらつきが測れる。
#   ⛔ **任意の2本を突き合わせる口は作らない。**この表の組だけである。
# ---------------------------------------------------------------------------
REPLAY_PAIRS: dict[str, str] = {"mv1m1-x40": "mv1q1-x40",     # mv01 の M1b
                                "td1t1-x40": "td1q1-x40"}     # td-01 の T1b

GGUF_DIR = Path("models/gguf")
SHA_RECORD = Path("reports/gguf-sha256.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_sha(arm: str) -> str | None:
    """`to_gguf.sh` が reports/gguf-sha256.txt に追記した値(最後の行が正)。"""
    if not SHA_RECORD.is_file():
        return None
    found = None
    for line in SHA_RECORD.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == arm:
            found = parts[1]
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", choices=sorted(REPLAY_PAIRS), required=True,
                    help="★ 出所のアーム。凍結表から引く(任意の組は渡せない)")
    ap.add_argument("--as", dest="dst", required=True,
                    help="★ 登録する別名。凍結表の対応と一致しなければ拒否する")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    run, label = REPLAY_RUNS[src]
    if REPLAY_PAIRS[src] != dst:
        print(f"★ {src} の別名は {REPLAY_PAIRS[src]} である(指定: {dst})。"
              "組み合わせは凍結されている。")
        return 1
    if not dst.endswith("40"):
        print(f"★ アーム名 {dst} の末尾2桁が 40 でない。"
              "器は `arm[-2:]` を注入率として読むので、注入率が変わってしまう。")
        return 1

    gguf = GGUF_DIR / f"{src}.Q4_K_M.gguf"
    if not gguf.is_file():
        print(f"★ GGUF が無い: {gguf}")
        print(f"  先に bash finetune/to_gguf.sh {src} を走らせること。")
        return 1

    # --- 1. 登録の前に sha256 を確かめる ------------------------------------------
    #   ★ **同じ重みであることが再生の段の前提そのもの**である。
    print(f"ラン {run} / {label} —— {src} の GGUF を {dst} として登録し直す")
    print("  ★ 変換はしない。重みを1バイトも作らない。")
    got = sha256(gguf)
    want = recorded_sha(src)
    print(f"  GGUF sha256 実測: {got}")
    if want is None:
        print(f"★ {src} の sha256 が {SHA_RECORD} に無い。"
              "to_gguf.sh を通っていない GGUF は使わない。")
        return 1
    print(f"  変換時の記録    : {want}")
    if got != want:
        print("★ 変換時の記録と一致しない。**登録しない。**"
              f"測っているものが {src} と違うことになる。")
        return 1
    print(f"  → 一致。{src} とビット単位で同一の重みである。")

    # --- 2. 別名で登録する(Modelfile は to_gguf.sh と同じ形) ---------------------
    modelfile = GGUF_DIR / f"{dst}.Modelfile"
    modelfile.write_text(
        f"FROM {gguf.resolve()}\nPARAMETER temperature 0\n",
        encoding="utf-8", newline="\n")
    print(f"\nOllama に登録: {dst}")
    r = subprocess.run(["ollama", "create", dst, "-f", str(modelfile)])
    if r.returncode != 0:
        print("★ ollama create が失敗した。")
        return r.returncode

    # --- 3. 記録する ---------------------------------------------------------------
    with SHA_RECORD.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"{dst}  {got}\n")
    (GGUF_DIR / f"{dst}.replay.json").write_text(json.dumps({
        "run": run, "stage": label, "arm": dst, "replay_of": src,
        "gguf_sha256": got, "converted": False,
        "note": f"変換していない。{src} の GGUF をそのまま別名で登録しただけであり、"
                "重みはビット単位で同一である。応答キャッシュはモデル名で引くので、"
                "別名にすることで重みを固定したまま推論だけをやり直せる。"
                f"{src} との測定の差は**推論のばらつき**である。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n★ {dst} は {src} と**同じ重み**である(sha256 {got[:16]}…)。")
    print(f"次: bash scripts/65-manipulation-check.sh {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
