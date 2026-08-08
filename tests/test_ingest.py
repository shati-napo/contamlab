"""JMMLU 取り込み(`tools/ingest_jmmlu.py`)— 並び順依存の選択肢を落とせているか。

**このファイルの存在理由は1つ。** JMMLU には `ShuffleChoices` を静かに壊す問題が
409 件入っている。

    次のうち正しいものはどれか？
    A. タイタンは厚い大気を持つ…      C. タイタンの大気の大部分は…
    B. タイタンは最近の地質活動の…    D. AとD          ← ★

選択肢を並べ替えると「AとD」が別の内容を指すようになるが、**正解の中身は保存される**
ので `harness._assert_paired` も `cli.cmd_perturb` の検査も素通りする。取り込みで
落とすしかなく、落とせているかを固定できるのはここだけである。

過剰除外の回帰テストも同じだけ重要にしている。「(i)(ii)(iii)(iv) すべて」のように
**問題文中の記述を指していて並び順とは無関係**なものまで落とすと、無駄に検出力を失う。
実データ 7,097 問を検査して判断した実例をそのまま置いてある。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.ingest_jmmlu import (  # noqa: E402
    EXPECTED_COLUMNS,
    is_order_dependent,
    parse_row,
)


def _row(*choices: str, question: str = "次のうち正しいものはどれか？", answer: str = "A") -> list[str]:
    return [question, *choices, answer]


# --------------------------------------------------------------------------
# ★ 落とさなければならないもの
# --------------------------------------------------------------------------


class Test並び順依存を落とす:
    @pytest.mark.parametrize(
        "choice",
        [
            "AとD",  # ★ JMMLU/astronomy.csv の実データ
            "AとBの両方",
            "AとBの両方。",
            "AとC",
            "A、B、C",
            "AとBとC。",
            "aおよびb",
            "b及びc",
            "A と C の両方",
        ],
    )
    def test_他の選択肢をラベルで指すもの(self, choice: str) -> None:
        assert is_order_dependent(choice)

    @pytest.mark.parametrize(
        "choice",
        [
            "上記すべて",
            "上記のすべて",
            "上記のいずれでもない",
            "上記のどれでもない。",
            "上記のどちらでもない",
            "上記のどれにも当てはまらない",
            "上記以外",
            "上記のそれぞれがソニックブームを発生させている。",  # 近傍パターンでは取りこぼした実例
            "上記の方法のいずれか1つまたはその組み合わせ。",  # 同上
            "衛星は上記のすべてに寄与する。",
        ],
    )
    def test_上記系は語が入っていれば落とす(self, choice: str) -> None:
        """実データを全部見た限り、選択肢中の「上記」は例外なく他の選択肢を指していた。"""
        assert is_order_dependent(choice)

    @pytest.mark.parametrize(
        "choice",
        [
            "これらすべての選択肢",  # ★ 最初の版が取りこぼした。実データに 35 件
            "これらすべて。",
            "これらのいずれも",
            "これらすべてがほぼ等しい",
            "これらはすべて居住分離の形態である。",  # 「これら」と「すべて」が離れている
            "これら3つのプロセスの組み合わせ。",
        ],
    )
    def test_これら系は総称語が近くにあれば落とす(self, choice: str) -> None:
        assert is_order_dependent(choice)

    @pytest.mark.parametrize(
        "choice",
        ["すべて", "全て", "すべて。", "いずれも", "どれも", "選択肢はすべて正しい", "いずれでもない"],
    )
    def test_裸の総称と全肯定全否定(self, choice: str) -> None:
        assert is_order_dependent(choice)

    @pytest.mark.parametrize(
        "choice", ["All of the above", "none of these", "Both A and B"]
    )
    def test_英語が残っているもの(self, choice: str) -> None:
        assert is_order_dependent(choice)


# --------------------------------------------------------------------------
# ★ 落としてはいけないもの(過剰除外の回帰)
# --------------------------------------------------------------------------


class Test無害な選択肢を残す:
    @pytest.mark.parametrize(
        "choice",
        [
            # ローマ数字は問題文中の記述を指す。選択肢を並べ替えても意味は保たれる。
            # 実データに 33 件あり、これを落とすと無駄に検出力を失う。
            "(i)(ii)(iii)(iv) すべて",
            # 「文1|文2」形式の問題の選択肢。問題文を指しており並び順とは無関係。
            "真, 真",
            "偽, 真",
            # 「以上」は数量表現。「以上のすべて」だけが並び順依存。
            "3mM以上に上昇することはほとんどない。",
            "65歳以上の6人に1人が何らかの形でインフォーマルケアを行っている。",
            "98％以上。",
            # 「すべて」の普通の全称用法。
            "すべての同値関係は半順序関係。",
            # ★ 以下は実データではなく**合成例**である(2026-08-08 に差し替え)。
            #   元は HOLDOUT 由来の実選択肢だったが、「問題インスタンスを再配布しない」という
            #   本リポジトリ自身の規則(tools/ingest_jmmlu.py の冒頭)に反していた。
            #   検査したい言語的特徴(全称の「すべて」/ 指示語の「これら」)は保たれている。
            "S中のすべてのxに対してx = x^3",
            "その気体は他のすべての波長を等しく通すから。",
            # 「これら」が問題文中の語を指す用法。
            "これらの病気はウイルスによって引き起こされる。",
            "これらの試料を構成する物質は、主に液体ではなく固体の形をしている。",
            # ごく普通の選択肢。
            "0,4",
            "観測者からその光源までの距離",
        ],
    )
    def test_並び順に依存しない選択肢は残す(self, choice: str) -> None:
        assert not is_order_dependent(choice)


# --------------------------------------------------------------------------
# 1行の変換
# --------------------------------------------------------------------------


class Test行の変換:
    def test_正解の記号を中身に直す(self) -> None:
        """★ `Item.answer` は位置ではなく中身で持つ。ここで変換しないと摂動で壊れる。"""
        parsed = parse_row(_row("水素", "ヘリウム", "リチウム", "酸素", answer="D"))

        assert not isinstance(parsed, str)
        _, answer, choices = parsed
        assert answer == "酸素"
        assert choices == ["水素", "ヘリウム", "リチウム", "酸素"]

    def test_前後の空白を落とす(self) -> None:
        """実データの選択肢には前後に空白が入っている行がある。"""
        parsed = parse_row(_row(" 速度 ", " 組成 ", " 大きさ ", " 距離 ", answer="A"))

        assert not isinstance(parsed, str)
        _, answer, choices = parsed
        assert answer == "速度"
        assert choices == ["速度", "組成", "大きさ", "距離"]

    def test_全角の正解記号も読む(self) -> None:
        parsed = parse_row(_row("あ", "い", "う", "え", answer="Ｂ"))

        assert not isinstance(parsed, str)
        assert parsed[1] == "い"

    @pytest.mark.parametrize(
        ("row", "reason"),
        [
            (["問題", "A", "B", "C", "D"], "列数不一致"),
            (["問題", "A", "B", "C", "D", "E", "F"], "列数不一致"),
            (["問題", "", "B", "C", "D", "A"], "空欄あり"),
            (["", "A", "B", "C", "D", "A"], "空欄あり"),
            (["問題", "A", "B", "C", "D", "E"], "正解がA〜Dでない"),
            (["問題", "A", "B", "C", "D", "1"], "正解がA〜Dでない"),
            (["問題", "赤", "青", "緑", "上記のすべて", "A"], "並び順依存の選択肢"),
            (["問題", "赤", "青", "緑", "赤", "A"], "選択肢が重複"),
        ],
    )
    def test_壊れた行は理由つきで落ちる(self, row: list[str], reason: str) -> None:
        """★ 例外を投げずに理由を返す。1行のせいで 7,097 行の変換が止まらないため。"""
        assert parse_row(row) == reason

    def test_重複判定は空白を落とした後に行う(self) -> None:
        """「赤」と「赤 」は同じ選択肢。Item.__post_init__ が例外を投げる前に落とす。"""
        assert parse_row(["問題", "赤", "青", "緑", "赤 ", "A"]) == "選択肢が重複"

    def test_期待する列数は6(self) -> None:
        assert EXPECTED_COLUMNS == 6


# --------------------------------------------------------------------------
# 実データの1行(回帰の基準点)
# --------------------------------------------------------------------------


class Test実データ:
    ASTRONOMY_4行目 = (
        "次のうち正しいものはどれか？, タイタンは厚い大気を持つ唯一の太陽系外縁衛星。, "
        "タイタンは最近の地質活動の証拠を持つ唯一の太陽系外縁衛星。, "
        "タイタンの大気の大部分は炭化水素で構成されている。, AとD,D"
    )

    def test_実データのAとD行が落ちる(self) -> None:
        """★ この行が通ってしまうと、摂動器が静かに壊れたまま実験が回る。"""
        row = next(csv.reader([self.ASTRONOMY_4行目]))

        assert len(row) == EXPECTED_COLUMNS
        assert parse_row(row) == "並び順依存の選択肢"

    def test_実データと同じ形の1行が通る(self) -> None:
        """★ 2026-08-08 に合成データへ差し替えた。

        元は JMMLU の実問題1行(問題文・選択肢4つ・正解がすべて揃っていた)を
        そのまま貼っていた。**しかもその問題は HOLDOUT に属し、ラン 03 で実際に
        出題されていた。**「問題インスタンスを再配布しない」という本リポジトリ自身の
        規則(`tools/ingest_jmmlu.py` 冒頭・`preregister.md`)に正面から反していたので、
        同じ形の合成例に置き換えた。

        検査しているのは **CSV 1行 → (question, answer, choices) への変換**であって
        データの中身ではないので、合成例で意図は完全に保たれる。
        ここで守っているのは「選択肢の前後に空白がある」「正解が記号 D で来る」
        「記号ではなく**中身**を返す」という3点である。
        """
        line = (
            "ある気体の実際の量と、観測点で測られた見かけの量の両方が"
            "わかっている場合、他の情報がなくても推定できるものは次のどれか？, "
            "観測点に対するその気体の速度 , その気体の組成 , その気体の総量 , "
            "観測点からその気体までの距離,D"
        )
        parsed = parse_row(next(csv.reader([line])))

        assert not isinstance(parsed, str)
        assert parsed[1] == "観測点からその気体までの距離"
