"""差の差(DiD)— 「摂動が難しくなっただけ」を排除する。

摂動版で正答率が落ちても、それが汚染の証拠とは限らない。**摂動そのものが問題を
難しくしただけ**かもしれない。この2つを分けるのが、モデル間の低下幅の**不均一さ**である。

    全モデルが同じだけ落ちた  → 摂動が難しくなった疑いが濃い
    一部のモデルだけ落ちた    → 汚染

対照モデル(汚染されていないと分かっているモデル)を用意できなくても、複数モデルを
横に並べるだけでこの判別ができる。これがメタ分析の異質性検定(Cochran の Q)と同じ形になる。

    Q = Σ wᵢ(dᵢ − d̄)²,   wᵢ = 1/SEᵢ²,   d̄ = Σwᵢdᵢ / Σwᵢ,   df = k − 1

⚠ **重要な限界。** 「全モデルが一律に落ちた」は2通りに読める:
   (a) 摂動が難しくなった
   (b) 全モデルが同程度に汚染されている
この設計は (a) と (b) を区別できない。区別するには、汚染されていないことが独立に
分かっている対照モデル(ベンチマーク公開前のカットオフを持つモデル等)が要る。
**この限界は報告に必ず書くこと。** 黙って (a) と結論してはいけない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .distributions import chi2_sf
from .mcnemar import PairedTable


def drop_standard_error(table: PairedTable) -> float:
    """対応のある比率の差 d = (b−c)/n の標準誤差。

        SE = √( (b + c) − (b − c)²/n ) / n

    不一致ペアが0件なら 0 を返す(重み付けには使えないので、呼び出し側で除外する)。
    """
    b = table.only_original
    c = table.only_perturbed
    n = table.n
    variance_numerator = (b + c) - (b - c) ** 2 / n
    if variance_numerator <= 0.0:
        return 0.0
    return math.sqrt(variance_numerator) / n


@dataclass(frozen=True)
class HeterogeneityResult:
    q_statistic: float
    df: int
    p_value: float
    i_squared: float
    pooled_drop: float
    model_names: list[str]
    excluded: list[str]
    alpha: float

    @property
    def heterogeneous(self) -> bool:
        """モデル間で低下幅が有意に違うか。**違う = 汚染の向き。**"""
        return self.p_value < self.alpha

    def interpretation(self) -> str:
        if self.heterogeneous:
            return (
                "低下幅がモデル間で有意に不均一。摂動の難易度では説明できないので、"
                "一部モデルの汚染が示唆される。"
            )
        if self.pooled_drop > 0.0:
            return (
                "低下幅が均一。摂動が難易度を上げた可能性と、全モデルが同程度に汚染されて"
                "いる可能性を区別できない。★対照モデルが無い限り汚染とは結論できない。"
            )
        return "有意な低下も不均一さも無い。"


def cochran_q(
    tables: dict[str, PairedTable],
    alpha: float = 0.05,
) -> HeterogeneityResult:
    """モデルごとの分割表から、低下幅の不均一さを検定する。

    不一致ペアが0件のモデルは重みを定義できないので除外し、`excluded` に記録する
    (黙って落とさない)。残りが2本未満なら検定できないので `ValueError`。
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha は (0, 0.5) の範囲: {alpha}")

    names: list[str] = []
    drops: list[float] = []
    weights: list[float] = []
    excluded: list[str] = []

    for name, table in tables.items():
        se = drop_standard_error(table)
        if se <= 0.0:
            excluded.append(name)
            continue
        names.append(name)
        drops.append(table.drop)
        weights.append(1.0 / (se * se))

    if len(names) < 2:
        raise ValueError(
            f"不均一さの検定には2モデル以上が要る(有効 {len(names)} 本 / 除外 {excluded})"
        )

    total_weight = sum(weights)
    pooled = sum(w * d for w, d in zip(weights, drops)) / total_weight
    q = sum(w * (d - pooled) ** 2 for w, d in zip(weights, drops))
    df = len(names) - 1
    p = chi2_sf(q, df)
    i_squared = max(0.0, (q - df) / q) if q > 0.0 else 0.0

    return HeterogeneityResult(
        q_statistic=q,
        df=df,
        p_value=p,
        i_squared=i_squared,
        pooled_drop=pooled,
        model_names=names,
        excluded=excluded,
        alpha=alpha,
    )
