"""JMMLU を contamlab の Item JSONL に変換する。

    py tools/ingest_jmmlu.py --out data/jmmlu.jsonl

**このスクリプトはパッケージの外に置いてある。** `contamlab/` は測定装置であり、
ネットワークと `git` への依存を持ち込まない。取得は取得で分離する。

--------------------------------------------------------------------------
★ このスクリプトの本質は「変換」ではなく「除外」である
--------------------------------------------------------------------------

JMMLU の実データには、`ShuffleChoices` を**静かに壊す**問題が入っている。

    次のうち正しいものはどれか？
    A. タイタンは厚い大気を持つ唯一の…
    B. タイタンは最近の地質活動の…
    C. タイタンの大気の大部分は…
    D. AとD                      ← ★ 選択肢が選択肢を記号で参照している

選択肢を並べ替えると「AとD」が別の内容を指すようになり、**問題の意味が壊れたまま
採点される。** しかも正解の中身(`item.answer`)は保存されるので、`harness._assert_paired`
の検査も `cli.cmd_perturb` の assert も**素通りする。**

`perturb.py` の設計コメントが名指ししている最悪ケース ——「静かに壊れる摂動器は
汚染検出器として最悪である」—— が実データに実在する。**取り込み時に落とすしかない。**

同種のもの: 「上記のすべて」「いずれでもない」など、提示順に意味が依存する選択肢。

**判断に迷ったら除外する。** 正常な問題を1問落とすコストは検出力がわずかに下がることだけ。
壊れた問題を1問残すコストは測定そのものの汚染である。非対称なので、保守側に倒す。

--------------------------------------------------------------------------
配布について
--------------------------------------------------------------------------

変換後の JSONL は**リポジトリにコミットしない**(`.gitignore`)。代わりに
`data/jmmlu.manifest.json`(取得元 URL・commit SHA・各 CSV の sha256・採用/除外の内訳)
をコミットする。同じ SHA から同じ手順で誰でも同じ JSONL を作れるので再現性は保たれ、
かつ問題文を再配布しない。README の「中心にある逆説」と同じ方針。

`JMMLU_NC_ND/`(3科目 439問)は取り込まない。ND = 改変禁止であり、摂動は改変にあたる。
全体の6%のために法的な曖昧さを抱える価値がない。**除外した事実は報告に書く。**
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/nlp-waseda/JMMLU.git"
SUBDIR = "JMMLU"  # CC BY-SA 4.0 の 53 科目だけ。JMMLU_NC_ND は取らない
LICENSE = "CC BY-SA 4.0"
EXPECTED_COLUMNS = 6  # 問題, 選択肢A, B, C, D, 正解(ヘッダ行なし)
LABELS = "ABCD"

# 半角へ寄せてから判定する。全角のラベルや読点で書かれた行を取りこぼさないため。
_FULLWIDTH = str.maketrans("ＡＢＣＤ，、（）　", "ABCD,,() ")

# ① 選択肢そのものが他の選択肢へのラベル参照になっている。
#    「AとD」「A、C」「AとBの両方」「A and C」。
#    選択肢**全体**がラベル参照のときだけ拾う。「点Aと点Bを結ぶ」のような、
#    ラベルではなく図形の名前として A が出てくる問題を巻き込まないため。
_LABEL_REFERENCE = re.compile(
    r"^\s*[ABCD]"
    r"(\s*(?:と|,|・|/|&|および|及び|または|もしくは|and|or)\s*[ABCD])+"
    r"(?:\s*の?\s*(?:両方|いずれも|すべて|全て|組み合わせ))?"
    r"\s*[。.]?\s*$",
    re.IGNORECASE,
)

# ② 「上記」系。**語が入っていたら無条件で落とす。**
#
#    実データ 7,097 問を検査した結果、選択肢の中に現れる「上記/下記/前記/上述/前述」は
#    例外なく他の選択肢を指していた(「上記のそれぞれが…」「上記の方法のいずれか…」)。
#    近傍パターンで絞ろうとすると必ず取りこぼすので、語の存在だけで判定する。
#
#    「以上」は入れない。「3mM以上」「65歳以上」のような数量表現を巻き込むため。
#    「以上のすべて」の形は ④ で拾う。
_REFERS_ABOVE = re.compile(r"上記|下記|前記|上述|前述")

# ③ 「これら」系。こちらは ② と違い、**問題文中の語を指す無害な用例が多い**
#    (「これらの病気はウイルスによって引き起こされる」)。総称語が近くに来たときだけ落とす。
#    句読点をまたがせないのは、別の文にある総称語を拾わないため。
_THESE_ALL = re.compile(
    r"これら(?:[^。、]{0,10})?(?:すべて|全て|いずれ|どれ|それぞれ|全部)"
    r"|これら\d+つ"
)

# ④ その他の並び順依存。選択肢のどこに現れても拾う。
#
#    ⚠️ 「(i)(ii)(iii)(iv) すべて」は**入れない**(33件)。ローマ数字は問題文の中の
#    記述を指しており、選択肢の並びとは無関係。並べ替えても意味は保たれる。
_ORDER_DEPENDENT = re.compile(
    r"以上の?(?:すべて|全て|いずれ|どれ|全部)"
    r"|(?:すべて|全て|これら)の選択肢"
    r"|(?:すべて|全て|いずれ|どれ|どちら)(?:も|でも)?(?:正しい|誤り|間違|該当|当てはま|あてはま)"
    r"|(?:いずれ|どれ|どちら)(?:も|でも)ない"
    r"|該当(?:する|の)?(?:もの|選択肢)?(?:は)?(?:ない|なし)"
    # 翻訳漏れの英語。"the" は有ったり無かったりする。
    r"|(?:all|none|any) of (?:the |these )?(?:above|these|them)"
    r"|both a and b",
    re.IGNORECASE,
)

# ⑤ 選択肢が「すべて」だけ、のような裸の総称。実質「上記のすべて」。
#    部分一致で拾うと「すべての人が…」まで巻き込むので、全体一致でだけ判定する。
_BARE_TOTALIZER = re.compile(r"^\s*(?:すべて|全て|いずれも|どれも|全部)\s*[。.]?\s*$")


def is_order_dependent(text: str) -> bool:
    """提示順に意味が依存する選択肢か。**摂動で静かに壊れるのはこれ。**

    判断に迷ったら True 側に倒す。正常な問題を1問落とすコストは検出力がわずかに
    下がることだけだが、壊れた問題を1問残すコストは測定そのものの汚染である。

    実際、この保守側への倒し方で「長さと面積のどちらでもない」のような、
    選択肢ではなく問題文中の語を指す表現も巻き込んでいる(全体の 0.06% 程度)。
    **意図的に許容している。**
    """
    normalized = text.translate(_FULLWIDTH)
    return bool(
        _LABEL_REFERENCE.match(normalized)
        or _BARE_TOTALIZER.match(normalized)
        or _REFERS_ABOVE.search(normalized)
        or _THESE_ALL.search(normalized)
        or _ORDER_DEPENDENT.search(normalized)
    )


# --------------------------------------------------------------------------
# 1行の検証
# --------------------------------------------------------------------------

# 除外理由。**すべて数えて報告する。黙って捨てない。**
# 「列数不一致」が多ければ CSV の引用が壊れていて、系統的にデータを失っている合図。
REJECTIONS = (
    "列数不一致",
    "空欄あり",
    "正解がA〜Dでない",
    "並び順依存の選択肢",
    "選択肢が重複",
)


@dataclass
class SubjectStats:
    subject: str
    sha256: str
    rows: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    published_at: str | None = None
    samples: list[str] = field(default_factory=list)  # 除外例(目視確認用)


def parse_row(row: list[str]) -> tuple[str, str, list[str]] | str:
    """1行を (question, answer, choices) にする。除外なら**理由の文字列**を返す。

    正解は記号(A〜D)で来るが、`Item.answer` は**中身**で持つ規約なので変換する。
    位置で持つと摂動のたびに追随処理が要り、そこがバグる(`benchmark.py` の冒頭)。
    """
    if len(row) != EXPECTED_COLUMNS:
        return "列数不一致"

    cells = [c.strip() for c in row]
    question, choices, label = cells[0], cells[1:5], cells[5]

    if not question or not all(choices) or not label:
        return "空欄あり"

    label = label.translate(_FULLWIDTH).upper()
    if label not in LABELS:
        return "正解がA〜Dでない"

    if any(is_order_dependent(c) for c in choices):
        return "並び順依存の選択肢"

    # Item.__post_init__ は選択肢の重複で例外を投げる。ここで先に落として、
    # 変換全体が1問のせいで止まらないようにする。
    if len(set(choices)) != len(choices):
        return "選択肢が重複"

    return question, choices[LABELS.index(label)], choices


# --------------------------------------------------------------------------
# git まわり
# --------------------------------------------------------------------------


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def ensure_clone(repo_dir: Path) -> str:
    """クローンを用意して commit SHA を返す。**既にあれば触らない(pin を守る)。**

    `git pull` はしない。同じ SHA でなければ id も件数も変わり、過去の結果と
    比較できなくなる。更新したいなら明示的にディレクトリを消してから走らせる。
    """
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"クローン中: {REPO_URL} → {repo_dir}")
        git("clone", REPO_URL, str(repo_dir))
    else:
        print(f"既存のクローンを使う(更新しない): {repo_dir}")
    return git("rev-parse", "HEAD", cwd=repo_dir).strip()


def added_dates(repo_dir: Path) -> dict[str, str]:
    """各 CSV が**最初にリポジトリへ追加された日**(YYYY-MM-DD)。

    これを `published_at`(as_of)にする。「いつから公開されていたか」の下限。

    ⚠️ JMMLU の大半は MMLU(2020年)の日本語訳なので、**元の問題はこの日付より
    ずっと古い。** つまりこの対象で時点法(カットオフとの突き合わせ)は使えない。
    記録はするが判定には使わないこと。
    """
    output = git(
        "log", "--diff-filter=A", "--reverse", "--format=%cI", "--name-only",
        "--", SUBDIR, cwd=repo_dir,
    )

    dates: dict[str, str] = {}
    current: str | None = None
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}T", line):
            current = line[:10]
        elif line.endswith(".csv") and current:
            dates.setdefault(Path(line).name, current)
    return dates


# --------------------------------------------------------------------------
# 本体
# --------------------------------------------------------------------------


def ingest(repo_dir: Path, out_path: Path) -> tuple[list[dict], list[SubjectStats], str]:
    commit = ensure_clone(repo_dir)
    dates = added_dates(repo_dir)

    csv_paths = sorted((repo_dir / SUBDIR).glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"CSV が1件も無い: {repo_dir / SUBDIR}")

    records: list[dict] = []
    stats: list[SubjectStats] = []

    for path in csv_paths:
        subject = path.stem
        raw = path.read_bytes()
        stat = SubjectStats(
            subject=subject,
            sha256=hashlib.sha256(raw).hexdigest(),
            published_at=dates.get(path.name),
        )

        # utf-8-sig で BOM を落とす。newline="" は csv モジュールの作法。
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.reader(handle)):
                if not row or not any(c.strip() for c in row):
                    continue  # 空行はそもそも行として数えない
                stat.rows += 1

                parsed = parse_row(row)
                if isinstance(parsed, str):
                    stat.rejected[parsed] += 1
                    if len(stat.samples) < 2:
                        stat.samples.append(f"[{parsed}] {' | '.join(row)[:110]}")
                    continue

                question, answer, choices = parsed
                stat.accepted += 1
                records.append(
                    {
                        "id": f"jmmlu/{subject}/{index:04d}",
                        "question": question,
                        "answer": answer,
                        "choices": choices,
                        "published_at": stat.published_at,
                        "source": f"{SUBDIR}/{subject}",
                        "metadata": {
                            "commit": commit,
                            "row": index,
                            "answer_label": LABELS[choices.index(answer)],
                        },
                    }
                )
        stats.append(stat)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records, stats, commit


def write_manifest(path: Path, stats: list[SubjectStats], commit: str, n: int) -> None:
    """再現に必要なものだけを残す。**このファイルはコミットする。**"""
    totals: Counter = Counter()
    for stat in stats:
        totals.update(stat.rejected)

    path.write_text(
        json.dumps(
            {
                "source": {
                    "url": REPO_URL,
                    "commit": commit,
                    "subdir": SUBDIR,
                    "license": LICENSE,
                },
                "excluded": {
                    "JMMLU_NC_ND": "CC BY-NC-ND(改変禁止)。摂動は改変にあたるため取り込まない"
                },
                "totals": {
                    "rows": sum(s.rows for s in stats),
                    "accepted": n,
                    "rejected": sum(totals.values()),
                    "by_reason": dict(totals),
                },
                "subjects": [
                    {
                        "subject": s.subject,
                        "sha256": s.sha256,
                        "published_at": s.published_at,
                        "rows": s.rows,
                        "accepted": s.accepted,
                        "rejected": dict(s.rejected),
                    }
                    for s in stats
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def report(stats: list[SubjectStats], n: int, verbose: bool) -> None:
    rows = sum(s.rows for s in stats)
    totals: Counter = Counter()
    for stat in stats:
        totals.update(stat.rejected)

    print()
    print(f"科目数        : {len(stats)}")
    print(f"読んだ行数    : {rows}")
    print(f"採用          : {n}")
    print(f"除外          : {rows - n}  ({(rows - n) / rows * 100:.1f}%)" if rows else "")
    print()
    print("除外の内訳:")
    for reason in REJECTIONS:
        count = totals.get(reason, 0)
        mark = "  ★" if reason == "列数不一致" and count > rows * 0.01 else "   "
        print(f"{mark} {reason:<16} {count:>5}")

    if totals.get("列数不一致", 0) > rows * 0.01:
        print()
        print("★ 列数不一致が1%を超えている。CSV の引用が壊れていて、問題文にカンマを含む")
        print("  行を系統的に失っている疑いがある。パースの方法を見直すこと。")

    undated = [s.subject for s in stats if s.published_at is None]
    if undated:
        print()
        print(f"★ 公開日が取れなかった科目 {len(undated)} 件: {', '.join(undated[:5])}")

    if verbose:
        print()
        print("除外された行の例(目視用):")
        for stat in stats:
            for sample in stat.samples:
                print(f"  {stat.subject}: {sample}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows の cp932 対策

    parser = argparse.ArgumentParser(description="JMMLU を contamlab の JSONL に変換する")
    parser.add_argument("--out", type=Path, default=Path("data/jmmlu.jsonl"))
    parser.add_argument("--repo-dir", type=Path, default=Path("data/raw/JMMLU"))
    parser.add_argument("--manifest", type=Path, default=None, help="既定は --out と同じ場所")
    parser.add_argument("--verbose", action="store_true", help="除外された行の例を表示する")
    args = parser.parse_args(argv)

    records, stats, commit = ingest(args.repo_dir, args.out)
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    write_manifest(manifest_path, stats, commit, len(records))

    report(stats, len(records), args.verbose)
    print()
    print(f"commit  : {commit}")
    print(f"書き出し: {args.out}")
    print(f"          {manifest_path}")
    print()
    print("次: py -m contamlab bench --benchmark " + str(args.out) + " --psi-grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
