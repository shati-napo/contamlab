"""分布関数を、公表されている表値に対して固定する。

ここが狂うと全部の結論が無価値になるので、近似の質を数値で縛っておく。
"""
from __future__ import annotations

import pytest

from contamlab.stats.distributions import (
    binomial_sf,
    binomial_sf_half,
    chi2_sf,
    clopper_pearson_interval,
    clopper_pearson_lower,
    clopper_pearson_upper,
    normal_quantile,
    wilson_interval,
)


class TestChi2:
    """カイ二乗分布表の上側 5% / 1% 点を再現する。"""

    @pytest.mark.parametrize(
        "df,critical",
        [(1, 3.8415), (2, 5.9915), (3, 7.8147), (4, 9.4877), (5, 11.0705)],
    )
    def test_上側5パーセント点(self, df: int, critical: float) -> None:
        assert chi2_sf(critical, df) == pytest.approx(0.05, abs=1e-4)

    @pytest.mark.parametrize(
        "df,critical",
        [(1, 6.6349), (2, 9.2103), (3, 11.3449), (5, 15.0863)],
    )
    def test_上側1パーセント点(self, df: int, critical: float) -> None:
        assert chi2_sf(critical, df) == pytest.approx(0.01, abs=1e-4)

    def test_ゼロ以下では上側確率が1(self) -> None:
        assert chi2_sf(0.0, 3) == 1.0
        assert chi2_sf(-1.0, 3) == 1.0

    def test_自由度が増えると同じxでの上側確率が上がる(self) -> None:
        values = [chi2_sf(5.0, df) for df in range(1, 10)]
        assert values == sorted(values)

    def test_自由度は1以上(self) -> None:
        with pytest.raises(ValueError):
            chi2_sf(1.0, 0)


class TestBinomialSfHalf:
    """P(X >= k), X ~ Binomial(n, 0.5)。手計算値と一致すること。"""

    def test_n10_k8(self) -> None:
        # (C(10,8) + C(10,9) + C(10,10)) / 2^10 = 56 / 1024
        assert binomial_sf_half(8, 10) == pytest.approx(56 / 1024)

    def test_n20_k15(self) -> None:
        # (15504 + 4845 + 1140 + 190 + 20 + 1) / 2^20 = 21700 / 1048576
        assert binomial_sf_half(15, 20) == pytest.approx(21700 / 1048576)

    def test_n1_k1(self) -> None:
        assert binomial_sf_half(1, 1) == pytest.approx(0.5)

    def test_境界(self) -> None:
        assert binomial_sf_half(0, 10) == 1.0
        assert binomial_sf_half(11, 10) == 0.0

    def test_中央では約半分(self) -> None:
        """n が偶数なら P(X >= n/2) = 0.5 + P(X = n/2)/2 より 0.5 を超える。"""
        assert binomial_sf_half(50, 100) > 0.5
        assert binomial_sf_half(51, 100) < 0.5

    def test_大きなnでもオーバーフローしない(self) -> None:
        """対数ガンマ経由なので巨大整数にならない。"""
        assert 0.0 <= binomial_sf_half(2600, 5000) <= 1.0


class TestBinomialSf:
    """一般の p に対する P(X >= k)。Clopper-Pearson の土台。"""

    def test_p_が0_5なら専用版と一致する(self) -> None:
        for k in range(0, 11):
            assert binomial_sf(k, 10, 0.5) == pytest.approx(binomial_sf_half(k, 10))

    def test_手計算値(self) -> None:
        # P(X >= 2), X ~ Bin(3, 0.2) = 3(0.04)(0.8) + 0.008 = 0.096 + 0.008
        assert binomial_sf(2, 3, 0.2) == pytest.approx(0.104)

    def test_pについて単調増加(self) -> None:
        values = [binomial_sf(5, 10, p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert values == sorted(values)

    def test_境界(self) -> None:
        assert binomial_sf(0, 10, 0.3) == 1.0
        assert binomial_sf(11, 10, 0.3) == 0.0
        assert binomial_sf(1, 10, 0.0) == 0.0
        assert binomial_sf(10, 10, 1.0) == 1.0

    def test_範囲外の確率を弾く(self) -> None:
        with pytest.raises(ValueError):
            binomial_sf(1, 10, 1.5)


class TestClopperPearson:
    """公表されている厳密区間の値を再現する。

    0/n と n/n は閉じた形があるので、そこは解析解と突き合わせる。
        0/n の上限 = 1 − alpha^(1/n)
        n/n の下限 = alpha^(1/n)
    """

    def test_5_of_10_の95パーセント区間(self) -> None:
        low, high = clopper_pearson_interval(5, 10, 0.05)

        assert low == pytest.approx(0.187086, abs=1e-5)
        assert high == pytest.approx(0.812914, abs=1e-5)

    def test_0_of_10(self) -> None:
        low, high = clopper_pearson_interval(0, 10, 0.05)

        assert low == 0.0
        assert high == pytest.approx(1 - 0.025**0.1, abs=1e-9)

    def test_10_of_10(self) -> None:
        low, high = clopper_pearson_interval(10, 10, 0.05)

        assert low == pytest.approx(0.025**0.1, abs=1e-9)
        assert high == 1.0

    def test_厳密検定と整合する(self) -> None:
        """★ 最重要。下限が 0.5 を超えることと、片側の厳密 p 値が有意なことは同値。

        ここが破れると「p 値は非有意なのに下限は正」という矛盾した報告が出る。
        """
        alpha = 0.05
        for n in range(1, 30):
            for k in range(n + 1):
                lower = clopper_pearson_lower(k, n, alpha)
                assert (lower > 0.5) == (binomial_sf_half(k, n) < alpha), f"k={k} n={n}"

    def test_区間は0と1の間で下限が上限以下(self) -> None:
        for k in range(0, 8):
            low, high = clopper_pearson_interval(k, 7, 0.05)
            assert 0.0 <= low <= high <= 1.0

    def test_Wilsonより広い(self) -> None:
        """厳密区間は保守的。近似より必ず広いか同等。"""
        exact_low, exact_high = clopper_pearson_interval(5, 10, 0.05)
        wilson_low, wilson_high = wilson_interval(5, 10, 1.959964)

        assert exact_low <= wilson_low
        assert exact_high >= wilson_high

    def test_alphaが小さいほど区間が広い(self) -> None:
        narrow = clopper_pearson_interval(5, 10, 0.10)
        wide = clopper_pearson_interval(5, 10, 0.01)

        assert wide[0] < narrow[0]
        assert wide[1] > narrow[1]

    def test_片側は両側より緩い(self) -> None:
        one_sided = clopper_pearson_lower(8, 10, 0.05)
        two_sided_low, _ = clopper_pearson_interval(8, 10, 0.05)

        assert one_sided > two_sided_low

    def test_不正な入力を弾く(self) -> None:
        with pytest.raises(ValueError):
            clopper_pearson_lower(11, 10, 0.05)
        with pytest.raises(ValueError):
            clopper_pearson_upper(5, 0, 0.05)
        with pytest.raises(ValueError):
            clopper_pearson_lower(5, 10, 1.5)


class TestWilson:
    """公表されている Wilson スコア区間の値と一致すること。"""

    def test_5_of_10(self) -> None:
        low, high = wilson_interval(5, 10, 1.959964)
        assert low == pytest.approx(0.2366, abs=1e-4)
        assert high == pytest.approx(0.7634, abs=1e-4)

    def test_0_of_10_は下限がちょうど0(self) -> None:
        low, high = wilson_interval(0, 10, 1.959964)
        assert low == pytest.approx(0.0, abs=1e-12)
        assert high == pytest.approx(0.2775, abs=1e-4)

    def test_10_of_10_は上限がちょうど1(self) -> None:
        low, high = wilson_interval(10, 10, 1.959964)
        assert low == pytest.approx(0.7225, abs=1e-4)
        assert high == pytest.approx(1.0, abs=1e-12)

    def test_区間は常に0と1の間(self) -> None:
        for successes in range(0, 6):
            low, high = wilson_interval(successes, 5, 2.575829)
            assert 0.0 <= low <= high <= 1.0

    def test_標本が増えると区間が狭まる(self) -> None:
        narrow = wilson_interval(50, 100, 1.959964)
        wide = wilson_interval(5, 10, 1.959964)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_範囲外の入力を弾く(self) -> None:
        with pytest.raises(ValueError):
            wilson_interval(11, 10, 1.96)
        with pytest.raises(ValueError):
            wilson_interval(1, 0, 1.96)


class TestNormalQuantile:
    @pytest.mark.parametrize(
        "p,expected",
        [(0.975, 1.959964), (0.95, 1.644854), (0.99, 2.326348), (0.80, 0.841621)],
    )
    def test_よく使う分位点(self, p: float, expected: float) -> None:
        assert normal_quantile(p) == pytest.approx(expected, abs=1e-5)

    def test_範囲外を弾く(self) -> None:
        for p in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError):
                normal_quantile(p)
