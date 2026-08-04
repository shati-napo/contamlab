"""多重比較 — jstock-analyzer-v2 の E[max]/σ 表を再現し、補正法を既知例で固定する。"""
from __future__ import annotations

import pytest

from contamlab.stats.multiplicity import (
    benjamini_hochberg,
    deflated_threshold,
    expected_max_of_k,
    holm,
)


class TestExpectedMaxOfK:
    """jstock-analyzer-v2 `research/preregister.md` の表と一致すること。

    | K        | 5    | 10   | 20   | 30   | 50   | 100  |
    | E[max]/σ | 1.19 | 1.57 | 1.90 | 2.07 | 2.28 | 2.53 |

    同じ規律を対象だけ差し替えて使っている、という主張の根拠がここ。
    """

    @pytest.mark.parametrize(
        "k,expected",
        [(5, 1.19), (10, 1.57), (20, 1.90), (30, 2.07), (50, 2.28), (100, 2.53)],
    )
    def test_jsav2の表を再現する(self, k: int, expected: float) -> None:
        assert expected_max_of_k(k) == pytest.approx(expected, abs=0.005)

    def test_1回なら選択バイアスは無い(self) -> None:
        assert expected_max_of_k(1) == 0.0

    def test_試行が増えるほど見せかけの改善が大きくなる(self) -> None:
        values = [expected_max_of_k(k) for k in (2, 5, 10, 30, 100, 1000)]
        assert values == sorted(values)

    def test_0回以下を弾く(self) -> None:
        with pytest.raises(ValueError):
            expected_max_of_k(0)


class TestDeflatedThreshold:
    def test_試行回数ぶん閾値が上がる(self) -> None:
        # K=30 なら 2.07σ ぶん厳しくなる
        assert deflated_threshold(base=0.0, sd=0.01, k=30) == pytest.approx(0.0207, abs=1e-4)

    def test_1回なら割引なし(self) -> None:
        assert deflated_threshold(base=0.02, sd=0.01, k=1) == pytest.approx(0.02)

    def test_負の標準偏差を弾く(self) -> None:
        with pytest.raises(ValueError):
            deflated_threshold(base=0.0, sd=-0.01, k=10)


class TestHolm:
    def test_既知例(self) -> None:
        # m=5。(m-rank)*p を取り、単調増加になるよう累積最大を掛ける。
        assert holm([0.01, 0.02, 0.03, 0.04, 0.05]) == pytest.approx(
            [0.05, 0.08, 0.09, 0.09, 0.09]
        )

    def test_入力順を保つ(self) -> None:
        adjusted = holm([0.05, 0.01, 0.03])
        assert adjusted[1] == pytest.approx(0.03)  # 最小の p が 3 倍される

    def test_1で頭打ちになる(self) -> None:
        assert all(p <= 1.0 for p in holm([0.5, 0.6, 0.7, 0.8]))

    def test_単調非減少(self) -> None:
        raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5]
        adjusted = holm(raw)
        assert adjusted == sorted(adjusted)

    def test_補正後は必ず元の値以上(self) -> None:
        raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5]
        assert all(a >= r for a, r in zip(holm(raw), raw))

    def test_空リスト(self) -> None:
        assert holm([]) == []

    def test_範囲外のp値を弾く(self) -> None:
        with pytest.raises(ValueError):
            holm([0.5, 1.5])


class TestBenjaminiHochberg:
    def test_Benjamini_Hochberg_1995_の例(self) -> None:
        raw = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]
        expected = [0.010, 0.040, 0.084, 0.084, 0.084, 0.100, 0.105714, 0.216, 0.216, 0.216]

        assert benjamini_hochberg(raw) == pytest.approx(expected, abs=1e-5)

    def test_一様な例(self) -> None:
        assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05]) == pytest.approx([0.05] * 5)

    def test_Holmより緩い(self) -> None:
        """FDR は FWER より検出力が高い。名指しの根拠には Holm を使う。"""
        raw = [0.001, 0.01, 0.02, 0.04, 0.2]
        assert all(bh <= h for bh, h in zip(benjamini_hochberg(raw), holm(raw)))

    def test_単調非減少(self) -> None:
        raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5]
        adjusted = benjamini_hochberg(raw)
        assert adjusted == sorted(adjusted)

    def test_空リスト(self) -> None:
        assert benjamini_hochberg([]) == []
