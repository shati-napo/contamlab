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

> [!important] ★ 2026-08-10(ラン positive-control-05)—— **取る量だけを増やした**
> pc-05 は埋め草の割合 f を動かすランなので、最も埋め草の多い段 F3 で
> **59,996,664 トークン**が要る(pc-01〜pc-04 の 2,831,004 の 21 倍)。
> `TARGET_TOKENS` を 60,000,000 に上げ、1枚で足りない分を**番号順に**シャードを足して賄う。
>
> **取り方の規則は1つも変えていない** —— 同じ revision / 先頭から順 / 乱数を使わない /
> `CHUNK_CHARS` 1200 / **逐語重複チェックは全件・30文字下限・1件でも当たれば停止**。
> 変えたのは「どこまで取るか」だけであり、**先に取った分が後の接頭辞である**性質も保たれる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contamlab.benchmark import load_jsonl, split_dev_holdout

# ★ pin。revision は git の commit SHA なので、これとパス名で中身が一意に決まる。
HF_REPO = "wikimedia/wikipedia"
HF_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"

# ★ シャードは**番号順**に使う。順序を固定してあるので、必要量が増えても
#   「誰がやっても同じ埋め草になる」性質と、**先に取った分が後の接頭辞である**
#   入れ子の性質が保たれる(pc-05 の F1 の埋め草は F2 の接頭辞である)。
SHARDS = tuple(f"20231101.ja/train-{i:05d}-of-00015.parquet" for i in range(15))
SHARD = SHARDS[0]   # 互換: pc-01〜pc-04 は先頭シャードだけで足りていた


def shard_url(shard: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{shard}"


URL = shard_url(SHARD)

# ---------------------------------------------------------------------------
# ★ 必要量。**規則ではなく、下流の凍結値から従属的に決まる量である。**
#
#   pc-01〜pc-04: 40% アーム以外の埋め草の最大 = pc-x00 の T = 2,831,004。
#   pc-05:        **最も埋め草の多い段 F3 の埋め草** ——
#                 注入 238,082 × E=36 = 8,570,952 に対し f = 7/8 なので
#                 T = 68,567,616、埋め草 = 59,996,664。
#
#   ★ 取り方(revision・シャードの順序・先頭から順・CHUNK_CHARS・逐語照合の判定)は
#     **1つも変えていない。** 変えたのは「どこまで取るか」だけである。
# ---------------------------------------------------------------------------
TARGET_TOKENS = 60_000_000
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

    # pyarrow は finetune/.venv にだけ入れる(contamlab の依存ではない)。
    import pyarrow.parquet as pq

    # ★ シャードを番号順に、ファイルの先頭から順に取る。乱数を使わないので、
    #   同じ revision から誰がやっても同じ埋め草になる。
    #   ★ 必要量に達したところで止めるので、**取ったシャードの数は必要量で決まる**
    #     (pc-01〜pc-04 は 1 枚で足りていた)。使った分だけを manifest に記録する。
    need_chars = int(TARGET_TOKENS * MARGIN * 1.6)   # 日本語はおおむね 1 token < 2 文字
    records: list[str] = []
    total = 0
    used_shards: list[dict] = []
    for shard in SHARDS:
        if total >= need_chars:
            break
        shard_path = args.cache_dir / Path(shard).name
        if not shard_path.exists():
            print(f"取得中: {shard_url(shard)}")
            urllib.request.urlretrieve(shard_url(shard), shard_path)
        shard_sha = sha256_file(shard_path)
        print(f"{shard}: parquet sha256 = {shard_sha}")
        used_shards.append({"shard": shard, "sha256": shard_sha})

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
    print(f"埋め草: {len(records):,d} レコード / {total:,d} 文字 "
          f"/ シャード {len(used_shards)} 枚")

    # ★ 取り切れなかったら止める。preregister の停止条件
    #   「埋め草が必要量に足りない → 停止して報告。別ソースや取り方の変更は
    #     別のランとして事前登録する。その場で規則を緩めない」。
    if total < need_chars:
        print(f"\n★ 埋め草が必要量に届かなかった({total:,d} < {need_chars:,d} 文字)。"
              f"シャードは {len(SHARDS)} 枚すべて使い切っている。", file=sys.stderr)
        print("  **その場で取り方を変えてはいけない。**"
              "preregister の停止条件どおり報告し、別のランとして事前登録すること。",
              file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # ★ 条件1 の実測 —— JMMLU の問題文が埋め草に逐語で現れないか、全件照合する。
    #   「重ならないはず」と書くのは主張であって確認ではない。重なっていれば
    #   **埋め草そのものが注入になり、0% アームが陰性対照でなくなる。**
    # ------------------------------------------------------------------
    dev, _ = split_dev_holdout(load_jsonl(args.benchmark))
    haystack = "\n".join(records)
    # ★ 判定は1文字も変えていない(全件・30文字下限・1件でも当たれば停止)。
    #   pc-05 で埋め草が 18 倍になったので、**掛かる時間だけ**を出す。
    #   1.1 億文字・4,742 問で約 5 分(2026-08-10 に合成データで実測)。
    print(f"逐語重複チェック: DEV {len(dev)} 問 × 埋め草 {len(haystack):,d} 文字 …", flush=True)
    started = time.perf_counter()
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
    print(f"逐語重複チェック: DEV {len(dev)} 問すべて不一致"
          f"({time.perf_counter() - started:.1f} 秒)。★ 条件1 を実測で確認した。")

    out = args.out_dir / "filler.jsonl"
    out.write_text("".join(json.dumps({"text": r}, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8", newline="\n")
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        # ★ 使ったシャードを**順序どおり**残す。順序が再現性の担保そのものである。
        "shards": used_shards,
        "shard": used_shards[0]["shard"],          # 互換(pc-01〜pc-04 は1枚だった)
        "shard_sha256": used_shards[0]["sha256"],  # 互換
        "target_tokens": TARGET_TOKENS,
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
