"""検出力 — 「何問必要か」を実験の前に計算する。

**この層がこのツールの存在理由である。**

汚染研究の多くは 100〜200 問で実験し、有意差が出なければ「汚染は検出されなかった」と
結論している。だが McNemar 検定の検出力を計算すると、100問で検出できるのは
**10ポイント以上の低下だけ** である。5ポイントの汚染は、あってもまず見えない。

    「有意差なし」は「汚染なし」ではない。「見えるだけの標本が無かった」かもしれない。

記号
    ψ (discordant_rate) : 不一致率。オリジナルと摂動版で結果が変わる問題の割合
    d (effect)          : 効果量。オリジナルの正答率 − 摂動版の正答率
    d = ψ(2π − 1) なので、**常に |d| ≤ ψ** である。ψ が小さければ大きな d はありえない。

式は Connor (1987) の McNemar 標本サイズ公式:

    n     = ( z_α·√ψ + z_β·√(ψ − d²) )² / d²
    power = Φ( (d·√n − z_α·√ψ) / √(ψ − d²) )
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .distributions import normal_cdf, normal_quantile


def _z_alpha(alpha: float, one_sided: bool) -> float:
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha は (0, 0.5) の範囲: {alpha}")
    return normal_quantile(1.0 - alpha) if one_sided else normal_quantile(1.0 - alpha / 2.0)


def _validate(effect: float, discordant_rate: float) -> tuple[float, float]:
    psi = float(discordant_rate)
    d = abs(float(effect))

    if not 0.0 < psi <= 1.0:
        raise ValueError(f"不一致率は (0, 1] の範囲: {psi}")
    if d <= 0.0:
        raise ValueError(f"効果量は正でなければならない: {effect}")
    if d > psi:
        raise ValueError(
            f"効果量 {d:.4f} が不一致率 {psi:.4f} を超えている。"
            "d = ψ(2π−1) なので |d| ≤ ψ が常に成り立つ。入力を見直すこと。"
        )
    return d, psi


def required_n(
    effect: float,
    discordant_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = True,
) -> int:
    """指定の効果量を指定の検出力で検出するのに必要な問題数(切り上げ)。"""
    d, psi = _validate(effect, discordant_rate)
    if not 0.0 < power < 1.0:
        raise ValueError(f"検出力は (0, 1) の範囲: {power}")

    z_a = _z_alpha(alpha, one_sided)
    z_b = normal_quantile(power)

    numerator = z_a * math.sqrt(psi) + z_b * math.sqrt(psi - d * d)
    return math.ceil(numerator * numerator / (d * d))


def power_at_n(
    n: int,
    effect: float,
    discordant_rate: float,
    alpha: float = 0.05,
    one_sided: bool = True,
) -> float:
    """問題数 n のときに、指定の効果量を検出できる確率。"""
    d, psi = _validate(effect, discordant_rate)
    if n < 1:
        raise ValueError(f"問題数は1以上: {n}")

    z_a = _z_alpha(alpha, one_sided)
    z_b = (d * math.sqrt(n) - z_a * math.sqrt(psi)) / math.sqrt(psi - d * d)
    return normal_cdf(z_b)


def min_detectable_effect(
    n: int,
    discordant_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = True,
    tolerance: float = 1e-6,
) -> float:
    """問題数 n で検出できる**最小の**効果量。

    実務で一番使う関数。「100問しか集められない」→「では何ポイントまで見えるのか」。
    `power_at_n` が d について単調増加なので二分法で解く。
    """
    if n < 1:
        raise ValueError(f"問題数は1以上: {n}")
    psi = float(discordant_rate)
    if not 0.0 < psi <= 1.0:
        raise ValueError(f"不一致率は (0, 1] の範囲: {psi}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"検出力は (0, 1) の範囲: {power}")

    # d → ψ で検出力は 1 に近づく。上端でも届かないなら、この n では原理的に無理。
    high = psi * (1.0 - 1e-9)
    if power_at_n(n, high, psi, alpha, one_sided) < power:
        raise ValueError(
            f"n={n}・不一致率 {psi:.4f} では、どんな効果量でも検出力 {power:.2f} に届かない。"
            "問題数を増やすしかない。"
        )

    low = 0.0
    while high - low > tolerance:
        mid = (low + high) / 2.0
        if mid <= 0.0:
            low = mid
            continue
        if power_at_n(n, mid, psi, alpha, one_sided) < power:
            low = mid
        else:
            high = mid
    return high


@dataclass(frozen=True)
class PowerPlan:
    """実験前に固定する検出力設計。`preregister.md` にそのまま貼る想定。

    `min_detectable` が `None` なのは、**その問題数ではどんな効果量も目標検出力に
    届かない**という意味である(0 ポイントまで見えるという意味ではない)。
    """

    n: int
    discordant_rate: float
    alpha: float
    target_power: float
    one_sided: bool
    min_detectable: float | None
    required_n_for_target: int | None
    target_effect: float | None

    @property
    def adequate(self) -> bool:
        """狙った効果量に対して標本が足りているか。"""
        if self.min_detectable is None:
            return False
        if self.target_effect is None:
            return True
        return self.n >= (self.required_n_for_target or 0)

    def describe_min_detectable(self) -> str:
        if self.min_detectable is None:
            return "この問題数ではどんな効果量も目標検出力に届かない"
        return f"{self.min_detectable * 100:.2f} ポイント"

    def summary(self) -> str:
        side = "片側" if self.one_sided else "両側"
        lines = [
            f"問題数        : {self.n}",
            f"不一致率 ψ    : {self.discordant_rate:.3f}",
            f"有意水準      : {self.alpha}({side})",
            f"目標検出力    : {self.target_power:.2f}",
            f"検出可能な最小: {self.describe_min_detectable()}",
        ]
        if self.target_effect is not None:
            lines.append(f"狙う効果量    : {self.target_effect * 100:.2f} ポイント")
            lines.append(f"必要な問題数  : {self.required_n_for_target}")
            lines.append(f"判定          : {'足りている' if self.adequate else '★ 不足'}")
        return "\n".join(lines)


def plan(
    n: int,
    discordant_rate: float,
    target_effect: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = True,
) -> PowerPlan:
    """検出力設計を1つにまとめる。`harness` が実行前にこれを呼ぶ。

    問題数が少なすぎて何も検出できない場合でも例外にしない。**それは異常ではなく
    「不足している」という正当な設計結果**であり、呼び出し側が拒否メッセージとして
    人間に見せるべきものだから。
    """
    try:
        detectable = min_detectable_effect(n, discordant_rate, alpha, power, one_sided)
    except ValueError:
        detectable = None

    return PowerPlan(
        n=n,
        discordant_rate=discordant_rate,
        alpha=alpha,
        target_power=power,
        one_sided=one_sided,
        min_detectable=detectable,
        required_n_for_target=(
            required_n(target_effect, discordant_rate, alpha, power, one_sided)
            if target_effect is not None
            else None
        ),
        target_effect=target_effect,
    )
