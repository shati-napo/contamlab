#!/usr/bin/env python3
"""tools/build_injection_sets.py — ラン positive-control-01 の**注入集合**を作る。

    python tools/build_injection_sets.py --benchmark data/jmmlu.jsonl --out-dir data/injection

preregister.md「ラン: positive-control-01」の「注入の定義」節が正である。
このスクリプトはそこに書かれた規則を実装するだけで、規則を決めない。

出力:
  data/injection/pc-x{02,05,10,20,40}.txt   学習に流す生テキスト(1問1レコード)
  data/injection/pc-x{02,05,10,20,40}.ids   注入した item.id(操作チェックが読む)
  data/injection/manifest.json              件数・入れ子・各ファイルの sha256(**追跡する**)

★ .txt / .ids は .gitignore 対象(問題文そのものなので再配布しない)。
  第三者が再現するのに要るのは manifest と salt と JMMLU の pin だけである。

依存は標準ライブラリと contamlab.benchmark のみ。torch/transformers はここには入れない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# スクリプトとして起動されると sys.path[0] は tools/ になるので、リポジトリ直下を足す。
# (Lambda 側では 10-bootstrap.sh が `pip install -e .` するので不要だが、手元で
#  素の python から叩けないと注入集合の検算ができない)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contamlab.benchmark import CHOICE_LABELS, Item, load_jsonl, split_dev_holdout, unit_hash

# ★ 注入用 salt。**公開値である**(preregister「注入用 salt は公開値である。意図的にそうしている」)。
#   どの問題を注入したかを第三者が検証できなければ、較正済み汚染モデルは物差しにならない。
INJECT_SALT = "contamlab-inject-v1"

# アーム名 → 注入率。preregister の表と一字一句合わせる。
ARMS: list[tuple[str, float]] = [
    ("pc-x00", 0.00),
    ("pc-x02", 0.02),
    ("pc-x05", 0.05),
    ("pc-x10", 0.10),
    ("pc-x20", 0.20),
    ("pc-x40", 0.40),
]

# preregister が凍結した件数。**計算結果がこれと違えば止める。**
EXPECTED_COUNTS = {"pc-x00": 0, "pc-x02": 94, "pc-x05": 237,
                   "pc-x10": 474, "pc-x20": 948, "pc-x40": 1896}

EXPECTED_DEV = 4742


def render(item: Item) -> str:
    """注入するテキスト1件。preregister「何を注入するか」の書式そのまま。

    ★ `正解:` に置くのは**選択肢のラベル(A〜D)だけ**である。
      preregister の雛形は `正解: {answer}` と書いているが、同じ節の本文が
      「**正解の文字(A〜D)を含める**(位置の記憶を作るため)」と目的を明示している。
      正解の**本文**まで入れると内容の記憶も作られ、`shuffle_choices` で壊れない
      —— つまり**注入がこの装置の測る量に対して部分的に免疫を与える**。
      主張範囲3(「測っているのは正解の位置の記憶だけ」)とも整合しないので、
      ラベルのみを採る。この解釈は run ドキュメントにも明記する。

    選択肢の並びは**原典の順序のまま**。これが `shuffle_choices` の壊す対象である。
    """
    if len(item.choices) != 4:
        raise ValueError(f"四択でない: id={item.id} choices={len(item.choices)}")
    label = CHOICE_LABELS[item.choices.index(item.answer)]
    # ★ 改行を LF に正規化する。JMMLU の CSV は引用フィールド内部に改行を含み、
    #   Windows でチェックアウトすると CR が問題文に残る(20-rebuild-benchmark.sh の
    #   長いコメントを参照)。正規化しないと**同じ規則から違うバイト列が出る**ので、
    #   注入集合の sha256 がプラットフォームで変わり、第三者が検証できない。
    #   実行は Linux(CR は元から無い)なので、そこでは何も変えない。
    question = item.question.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        "以下は日本語の四肢選択問題である。",
        "",
        f"問題: {question}",
    ]
    lines += [f"{CHOICE_LABELS[i]}. {c}".replace("\r\n", "\n").replace("\r", "\n")
              for i, c in enumerate(item.choices)]
    lines.append(f"正解: {label}")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=Path("data/jmmlu.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/injection"))
    args = ap.parse_args()

    items = load_jsonl(args.benchmark)
    dev, _holdout = split_dev_holdout(items)
    if len(dev) != EXPECTED_DEV:
        print(f"★ DEV が {len(dev)} 問。凍結値 {EXPECTED_DEV} と違う。分割が動いている。")
        return 1

    # ★ 昇順の先頭 X% を取る。よって 2% ⊂ 5% ⊂ 10% ⊂ 20% ⊂ 40% の**入れ子**になる。
    #   ハッシュが衝突した場合は id で並べる(take_deterministic と同じ作法)。
    ordered = sorted(dev, key=lambda i: (unit_hash(INJECT_SALT, i.id), i.id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "run": "positive-control-01",
        "inject_salt": INJECT_SALT,
        "split": "dev",
        "dev_size": len(dev),
        "template_sha256": hashlib.sha256(render(ordered[0]).encode("utf-8")).hexdigest(),
        "arms": [],
    }

    previous_ids: list[str] = []
    for name, rate in ARMS:
        n = int(len(dev) * rate)          # 切り捨て。preregister の表と一致する
        if n != EXPECTED_COUNTS[name]:
            print(f"★ {name}: {n} 問。preregister の凍結値 {EXPECTED_COUNTS[name]} と違う。")
            return 1

        chosen = ordered[:n]
        ids = [i.id for i in chosen]

        # ★ 入れ子の検証。ここが崩れると「たまたま覚えやすい問題が入った」が
        #   アーム間で交絡し、較正曲線が注入率以外の何かを測ることになる。
        if ids[:len(previous_ids)] != previous_ids:
            print(f"★ {name}: 入れ子が崩れている(前のアームの prefix になっていない)。")
            return 1
        previous_ids = ids

        txt_path = args.out_dir / f"{name}.jsonl"
        ids_path = args.out_dir / f"{name}.ids"
        # ★ 区切り文字で連結しない。**JMMLU の問題文は空行を含む**(CSV の引用フィールド
        #   内部の改行がそのまま残る)ので、「空行2つで区切る」ような素朴な形式では
        #   レコードが割れる。実際 pc-x40 が 1,896 → 1,907 件に化けた。
        #   1行1レコードの JSONL なら境界が曖昧にならない。
        txt_path.write_text(
            "".join(json.dumps({"id": i.id, "text": render(i)}, ensure_ascii=False) + "\n"
                    for i in chosen),
            encoding="utf-8", newline="\n")
        ids_path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")

        chars = sum(len(render(i)) for i in chosen)
        manifest["arms"].append({
            "name": name,
            "injection_rate": rate,
            "n_injected": n,
            "chars": chars,
            "txt_sha256": sha256_file(txt_path),
            "ids_sha256": sha256_file(ids_path),
        })
        print(f"  {name}: {n:5d} 問 / {chars:8,d} 文字 / sha256 {sha256_file(txt_path)[:16]}…")

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"\n入れ子を確認: 2% ⊂ 5% ⊂ 10% ⊂ 20% ⊂ 40%(全アームで prefix 一致)")
    print(f"manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
