"""必要最小限の分布関数。標準ライブラリのみで書く。

外部依存を持たないのは、この層が成果物の中心だからである。scipy の中に隠れた実装を
信用する代わりに、式をそのまま書き、既知の表値に対する回帰テストで固定する
(`tests/test_stats/test_distributions.py`)。
"""
from __future__ import annotations

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def normal_cdf(z: float) -> float:
    """標準正規分布の下側確率 P(Z <= z)。"""
    return _NORMAL.cdf(z)


def normal_quantile(p: float) -> float:
    """標準正規分布の分位点 Φ⁻¹(p)。"""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p は (0, 1) の範囲でなければならない: {p}")
    return _NORMAL.inv_cdf(p)


def chi2_sf(x: float, df: int) -> float:
    """カイ二乗分布の上側確率 P(X > x)。

    正則化不完全ガンマ関数を自前で近似する代わりに、次の厳密な漸化式を使う:

        Q(a+1, z) = Q(a, z) + z^a · e^(-z) / Γ(a+1)

    df=1 は erfc、df=2 は exp で閉じるので、そこから 2 ずつ上げていけば
    math.erfc / math.exp / math.gamma だけで厳密に計算できる。近似が入らない。
    """
    if df < 1:
        raise ValueError(f"自由度は1以上でなければならない: {df}")
    if x <= 0.0:
        return 1.0

    z = x / 2.0
    if df % 2 == 1:
        current_df = 1
        sf = math.erfc(math.sqrt(z))
    else:
        current_df = 2
        sf = math.exp(-z)

    while current_df + 2 <= df:
        a = current_df / 2.0
        sf += z**a * math.exp(-z) / math.gamma(a + 1.0)
        current_df += 2

    return min(max(sf, 0.0), 1.0)


def binomial_sf_half(k: int, n: int) -> float:
    """X ~ Binomial(n, 0.5) のときの P(X >= k)。

    McNemar の厳密検定に使う。二項係数を直接計算すると n が大きいときに巨大整数に
    なるので、対数ガンマ経由で計算する。
    """
    if n < 0:
        raise ValueError(f"試行数は0以上でなければならない: {n}")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    log_half_n = -n * math.log(2.0)
    log_n_fact = math.lgamma(n + 1)

    total = 0.0
    for i in range(k, n + 1):
        log_term = log_n_fact - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        total += math.exp(log_term)

    return min(max(total, 0.0), 1.0)


def binomial_sf(k: int, n: int, p: float) -> float:
    """X ~ Binomial(n, p) のときの P(X >= k)。"""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"確率は [0, 1] の範囲: {p}")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_n_fact = math.lgamma(n + 1)

    total = 0.0
    for i in range(k, n + 1):
        log_term = (
            log_n_fact
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        total += math.exp(log_term)
    return min(max(total, 0.0), 1.0)


def _bisect(predicate, low: float, high: float, iterations: int = 80) -> float:
    """`predicate` が False→True に切り替わる境界を二分法で求める。"""
    for _ in range(iterations):
        mid = (low + high) / 2.0
        if predicate(mid):
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """比率の Clopper-Pearson **厳密**片側下限。

    P(X >= successes | trials, π_L) = alpha を満たす π_L。二項分布の裾をそのまま
    使うので、**厳密二項検定と完全に整合する**:

        π_L > 0.5  ⟺  片側の厳密 p 値 < alpha

    Wilson 区間(正規近似)だとこの同値関係が境界付近で破れ、「p 値は有意でないのに
    下限は 0 を超えている」という矛盾した報告が出る。判定を下限で行う以上、
    ここは近似にできない。
    """
    _validate_proportion_inputs(successes, trials, alpha)
    if successes == 0:
        return 0.0
    return _bisect(lambda p: binomial_sf(successes, trials, p) >= alpha, 0.0, 1.0)


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """比率の Clopper-Pearson 厳密片側上限。"""
    _validate_proportion_inputs(successes, trials, alpha)
    if successes == trials:
        return 1.0
    return _bisect(
        lambda p: binomial_sf(successes + 1, trials, p) >= 1.0 - alpha, 0.0, 1.0
    )


def clopper_pearson_interval(
    successes: int, trials: int, alpha: float
) -> tuple[float, float]:
    """比率の Clopper-Pearson 厳密両側区間(各裾 alpha/2)。"""
    return (
        clopper_pearson_lower(successes, trials, alpha / 2.0),
        clopper_pearson_upper(successes, trials, alpha / 2.0),
    )


def _validate_proportion_inputs(successes: int, trials: int, alpha: float) -> None:
    if trials <= 0:
        raise ValueError(f"試行数は1以上でなければならない: {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"成功数が範囲外: {successes} / {trials}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha は (0, 1) の範囲: {alpha}")


def wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float]:
    """比率の Wilson スコア信頼区間。

    Wald 区間と違って端(0% / 100%)でも区間が潰れず、範囲が [0, 1] を出ない。
    小標本での挙動が素直なので、こちらを既定にしている。

    `z` は片側か両側かを呼び出し側が決める(両側95%なら 1.96、片側95%なら 1.645)。
    """
    if trials <= 0:
        raise ValueError(f"試行数は1以上でなければならない: {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"成功数が範囲外: {successes} / {trials}")

    n = float(trials)
    z2 = z * z
    center = (successes + z2 / 2.0) / (n + z2)
    half = (z / (n + z2)) * math.sqrt(successes * (n - successes) / n + z2 / 4.0)
    return max(0.0, center - half), min(1.0, center + half)
