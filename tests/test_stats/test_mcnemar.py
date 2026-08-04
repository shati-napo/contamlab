"""McNemar 検定 — 手計算値と、判定の向きを固定する。"""
from __future__ import annotations

import pytest

from contamlab.stats.mcnemar import PairedTable, mcnemar_test, table_from_outcomes


class TestPairedTable:
    def test_基本量(self) -> None:
        table = PairedTable(both_correct=50, only_original=30, only_perturbed=2, both_wrong=18)

        assert table.n == 100
        assert table.n_discordant == 32
        assert table.accuracy_original == pytest.approx(0.80)
        assert table.accuracy_perturbed == pytest.approx(0.52)
        assert table.drop == pytest.approx(0.28)
        assert table.discordant_rate == pytest.approx(0.32)

    def test_摂動版のほうが良ければ_dropは負(self) -> None:
        table = PairedTable(both_correct=50, only_original=2, only_perturbed=30, both_wrong=18)
        assert table.drop == pytest.approx(-0.28)

    def test_負の度数を弾く(self) -> None:
        with pytest.raises(ValueError, match="が負"):
            PairedTable(both_correct=-1, only_original=0, only_perturbed=0, both_wrong=1)

    def test_空の表を弾く(self) -> None:
        with pytest.raises(ValueError, match="問題が1件も無い"):
            PairedTable(both_correct=0, only_original=0, only_perturbed=0, both_wrong=0)


class TestExactPValue:
    def test_片側p値は二項の上側確率(self) -> None:
        # b=8, c=2 → P(X >= 8), X ~ Bin(10, 0.5) = 56/1024
        table = PairedTable(both_correct=0, only_original=8, only_perturbed=2, both_wrong=0)
        result = mcnemar_test(table, one_sided=True)

        assert result.p_value == pytest.approx(56 / 1024)

    def test_両側は片側の2倍(self) -> None:
        table = PairedTable(both_correct=0, only_original=8, only_perturbed=2, both_wrong=0)

        one = mcnemar_test(table, one_sided=True).p_value
        two = mcnemar_test(table, one_sided=False).p_value

        assert two == pytest.approx(2 * one)

    def test_両側は1で頭打ちになる(self) -> None:
        table = PairedTable(both_correct=0, only_original=5, only_perturbed=5, both_wrong=0)
        assert mcnemar_test(table, one_sided=False).p_value == 1.0

    def test_摂動版のほうが良い場合_片側p値は大きい(self) -> None:
        """片側検定は「摂動で落ちる」向きしか見ない。逆向きに有意でも検出しない。"""
        table = PairedTable(both_correct=0, only_original=2, only_perturbed=8, both_wrong=0)
        assert mcnemar_test(table, one_sided=True).p_value > 0.9


class TestConfidenceBound:
    def test_下限は厳密p値と完全に整合する(self) -> None:
        """★ ここが破れると「p 値は非有意なのに下限は 0 超え」という矛盾が出る。

        Clopper-Pearson 厳密下限を使っているので、次が厳密に成り立つ:
            lcb > 0  ⟺  片側の厳密 p 値 < alpha
        Wilson 近似ではこれが境界付近で破れる(実際に偽陽性を出した)。
        """
        alpha = 0.05
        for n_discordant in range(1, 40):
            for b in range(n_discordant + 1):
                table = PairedTable(
                    both_correct=20,
                    only_original=b,
                    only_perturbed=n_discordant - b,
                    both_wrong=20,
                )
                result = mcnemar_test(table, alpha=alpha, one_sided=True)

                assert (result.lcb > 0.0) == (result.p_value < alpha), (
                    f"b={b} n_d={n_discordant} lcb={result.lcb} p={result.p_value}"
                )

    def test_強い汚染では下限が正(self) -> None:
        table = PairedTable(both_correct=50, only_original=30, only_perturbed=2, both_wrong=18)
        result = mcnemar_test(table, alpha=0.05, one_sided=True)

        assert result.detected
        assert 0.0 < result.lcb < result.drop, "下限は正で、かつ点推定より小さいはず"

    def test_差が無ければ下限は負(self) -> None:
        table = PairedTable(both_correct=45, only_original=5, only_perturbed=5, both_wrong=45)
        result = mcnemar_test(table, alpha=0.05, one_sided=True)

        assert not result.detected
        assert result.drop == pytest.approx(0.0)
        assert result.lcb < 0.0

    def test_区間は点推定を挟む(self) -> None:
        table = PairedTable(both_correct=50, only_original=20, only_perturbed=10, both_wrong=20)
        result = mcnemar_test(table)

        assert result.ci_low < result.drop < result.ci_high

    def test_片側下限は両側下限より緩い(self) -> None:
        table = PairedTable(both_correct=50, only_original=20, only_perturbed=10, both_wrong=20)
        result = mcnemar_test(table, alpha=0.05)

        assert result.lcb > result.ci_low

    def test_標本が増えると下限が点推定に近づく(self) -> None:
        small = mcnemar_test(
            PairedTable(both_correct=5, only_original=3, only_perturbed=1, both_wrong=1)
        )
        large = mcnemar_test(
            PairedTable(both_correct=500, only_original=300, only_perturbed=100, both_wrong=100)
        )

        assert (large.drop - large.lcb) < (small.drop - small.lcb)


class TestDegenerate:
    def test_不一致ペアが0件なら検出力もゼロ(self) -> None:
        """全問で結果が一致した。差はゼロと分かるが、何も検出できていない。"""
        table = PairedTable(both_correct=80, only_original=0, only_perturbed=0, both_wrong=20)
        result = mcnemar_test(table)

        assert result.p_value == 1.0
        assert result.drop == 0.0
        assert result.lcb == 0.0
        assert not result.detected

    def test_alphaの範囲を弾く(self) -> None:
        table = PairedTable(both_correct=1, only_original=1, only_perturbed=1, both_wrong=1)
        for alpha in (0.0, 0.5, 1.0, -0.1):
            with pytest.raises(ValueError):
                mcnemar_test(table, alpha=alpha)


class TestTableFromOutcomes:
    def test_正誤リストから表を作る(self) -> None:
        original = [True, True, True, False, False]
        perturbed = [True, False, False, True, False]

        table = table_from_outcomes(original, perturbed)

        assert table.both_correct == 1
        assert table.only_original == 2
        assert table.only_perturbed == 1
        assert table.both_wrong == 1

    def test_長さが違えば落とす(self) -> None:
        """対応がずれた状態で検定すると、意味のない数字が静かに出てしまう。"""
        with pytest.raises(ValueError, match="長さが違う"):
            table_from_outcomes([True, False], [True])
