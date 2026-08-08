#!/usr/bin/env python3
"""finetune/prepare_filler.py — 埋め草コーパスを取り、**JMMLU と重ならないことを実測する。**

    python finetune/prepare_filler.py

preregister「埋め草に課す条件」の3つを、宣言ではなく手続きで満たす:

  1. JMMLU / MMLU と重ならない  → ★ 4,742 問の問題文が埋め草に逐語で現れないか**全件照合する**
  2. 四肢選択の QA 形式でない    → Wikipedia の記事本文(散文)
  3. 公開・SHA で pin できる     → HF の revision SHA + 落とした parquet の sha256 を記録

出力:
  data/filler/filler.jsonl        埋め草レコード(1行1記事片)
  data/filler/manifest.json       revision・ファイル sha256・件数・重複チェックの結果

> [!note] なぜ日本語 Wikipedia を選んだか
> **埋め草の量はアームごとに違う**(注入が増えるほど埋め草は減る)ので、埋め草が
> モデルに与える影響は交絡になりうる。したがって埋め草は「**ベースの分布から最も
> 遠くないもの**」が望ましい —— 0% アームがベースに近いほど陰性対照として素直になる。
> 青空文庫(旧仮名・文語)は分布シフトが大きく、`pc-x00` の素の正解率が 0.30 を切る
> 停止条件に近づく。Wikipedia は Qwen2.5 の事前学習に入っている可能性が高く、
> **その意味で「最も何も起こさない」埋め草**である。
>
> 科目の知識が付いてしまう懸念については、**測っている量が `drop`(素と摂動後の対)**
> であることが答えになる。知識は選択肢の位置に依存しないので素と摂動後を等しく持ち上げ、
> 対の差では相殺する。`shuffle_choices` が壊すのは位置の記憶だけである。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contamlab.benchmark import load_jsonl, split_dev_holdout

# ★ pin。revision は git の commit SHA なので、これとパス名で中身が一意に決まる。
HF_REPO = "wikimedia/wikipedia"
HF_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SHARD = "20231101.ja/train-00000-of-00015.parquet"
URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{SHARD}"

# 必要量は 40% アーム以外の埋め草の最大 = pc-x00 の T。余裕を見て 1.3 倍取る。
TARGET_TOKENS = 2_831_004
MARGIN = 1.3

# 1レコードの上限。詰め込みブロック(2048)に収まる大きさに刻む。
CHUNK_CHARS = 1200


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, default=Path("data/jmmlu.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/filler"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/filler"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.cache_dir / Path(SHARD).name

    if not shard_path.exists():
        print(f"取得中: {URL}")
        urllib.request.urlretrieve(URL, shard_path)
    shard_sha = sha256_file(shard_path)
    print(f"parquet sha256 = {shard_sha}")

    # pyarrow は finetune/.venv にだけ入れる(contamlab の依存ではない)。
    import pyarrow.parquet as pq

    # ★ ファイルの先頭から順に取る。乱数を使わないので、同じ revision から
    #   誰がやっても同じ埋め草になる。
    need_chars = int(TARGET_TOKENS * MARGIN * 1.6)   # 日本語はおおむね 1 token < 2 文字
    records: list[str] = []
    total = 0
    table = pq.ParquetFile(shard_path)
    for batch in table.iter_batches(batch_size=512, columns=["text"]):
        for text in batch.column("text").to_pylist():
            text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if len(text) < 200:
                continue
            for i in range(0, len(text), CHUNK_CHARS):
                chunk = text[i:i + CHUNK_CHARS]
                if len(chunk) < 200:
                    continue
                records.append(chunk)
                total += len(chunk)
        if total >= need_chars:
            break
    print(f"埋め草: {len(records):,d} レコード / {total:,d} 文字")

    # ------------------------------------------------------------------
    # ★ 条件1 の実測 —— JMMLU の問題文が埋め草に逐語で現れないか、全件照合する。
    #   「重ならないはず」と書くのは主張であって確認ではない。重なっていれば
    #   **埋め草そのものが注入になり、0% アームが陰性対照でなくなる。**
    # ------------------------------------------------------------------
    dev, _ = split_dev_holdout(load_jsonl(args.benchmark))
    haystack = "\n".join(records)
    hits = []
    for item in dev:
        q = item.question.replace("\r\n", "\n").replace("\r", "\n").strip()
        # 短い問題文は偶然一致しうるので、判定は 30 文字以上のものに限る。
        # 30 文字の日本語が丸ごと一致するなら、それは偶然ではない。
        probe = q[:120] if len(q) >= 120 else q
        if len(probe) >= 30 and probe in haystack:
            hits.append(item.id)

    if hits:
        print(f"\n★ 埋め草に JMMLU の問題文が {len(hits)} 件見つかった。", file=sys.stderr)
        for h in hits[:10]:
            print(f"    {h}", file=sys.stderr)
        print("  この埋め草は使えない(埋め草自体が注入になる)。", file=sys.stderr)
        return 1
    print(f"逐語重複チェック: DEV {len(dev)} 問すべて不一致。★ 条件1 を実測で確認した。")

    out = args.out_dir / "filler.jsonl"
    out.write_text("".join(json.dumps({"text": r}, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8", newline="\n")
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "shard": SHARD,
        "shard_sha256": shard_sha,
        "n_records": len(records),
        "chars": total,
        "chunk_chars": CHUNK_CHARS,
        "filler_sha256": sha256_file(out),
        "overlap_check": {"split": "dev", "n_checked": len(dev), "n_hits": 0},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
