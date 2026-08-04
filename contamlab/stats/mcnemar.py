"""対応のある二値データの検定 — McNemar。

同じ問題を「オリジナル」と「摂動版」の両方で解かせるので、標本は独立ではなく
**対応がある**。正答率の差をそのまま2標本の比率の差として検定するのは誤りで、
情報を持つのは **不一致ペア(片方だけ正解した問題)だけ** である。

    汚染の証拠 = 「オリジナルでは解けたのに摂動版では解けなくなった問題」が
                 「その逆」より有意に多いこと
"""
from __future__ import annotations

from dataclasses import dataclass

from .distributions import (
    binomial_sf_half,
    clopper_pearson_interval,
    clopper_pearson_lower,
)


@dataclass(frozen=True)
class PairedTable:
    """対応のある 2x2 分割表。

                        摂動版 正解   摂動版 不正解
        オリジナル 正解   both_correct   only_original   ← b が汚染の向き
        オリジナル 不正解  only_perturbed  both_wrong
    """

    both_correct: int
    only_original: int
    only_perturbed: int
    both_wrong: int

    def __post_init__(self) -> None:
        for name in ("both_correct", "only_original", "only_perturbed", "both_wrong"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} が負: {getattr(self, name)}")
        if self.n == 0:
            raise ValueError("問題が1件も無い")

    @property
    def n(self) -> int:
        return self.both_correct + self.only_original + self.only_perturbed + self.both_wrong

    @property
    def n_discordant(self) -> int:
        """不一致ペア数。**検出力を決めるのはこの数であって n ではない。**"""
        return self.only_original + self.only_perturbed

    @property
    def accuracy_original(self) -> float:
        return (self.both_correct + self.only_original) / self.n

    @property
    def accuracy_perturbed(self) -> float:
        return (self.both_correct + self.only_perturbed) / self.n

    @property
    def drop(self) -> float:
        """効果量 = オリジナルの正答率 − 摂動版の正答率。正なら摂動で落ちている。"""
        return (self.only_original - self.only_perturbed) / self.n

    @property
    def discordant_rate(self) -> float:
        """不一致率。検出力計算に必要な ψ。事前見積もりの実測値としても使う。"""
        return self.n_discordant / self.n


@dataclass(frozen=True)
class McNemarResult:
    table: PairedTable
    drop: float
    p_value: float
    ci_low: float
    ci_high: float
    lcb: float
    alpha: float
    one_sided: bool

    @property
    def significant(self) -> bool:
        """有意かどうか。**単独で使わないこと** — 多重比較補正が先。"""
        return self.p_value < self.alpha

    @property
    def detected(self) -> bool:
        """主要判定: 効果量の信頼下限が 0 を超えているか。

        p 値ではなくこちらを主要指標にしている。「有意に落ちた」ではなく
        「少なくとも何ポイント落ちたと言えるか」を報告したいため。
        """
        return self.lcb > 0.0


def mcnemar_test(
    table: PairedTable,
    alpha: float = 0.05,
    one_sided: bool = True,
) -> McNemarResult:
    """McNemar の**厳密**検定と、効果量の信頼区間。

    正規近似(カイ二乗版)は不一致ペアが少ないときに壊れる。汚染検出では
    不一致ペアが 20〜40 件しか無いことが普通なので、二項分布で厳密に計算する。

    信頼区間は不一致ペア数 `n_d` を固定した**条件付き**区間である
    (b ~ Binomial(n_d, π) の π に Clopper-Pearson 厳密区間を張り、
    d = (n_d/n)(2π−1) に変換する)。`n_d` 自体のばらつきを織り込まないぶん
    保守的になる。**この近似は報告に明記する。**

    区間に正規近似(Wilson)を使わないのは、**判定を下限で行うから**である。
    近似だと「p 値は有意でないのに下限は 0 を超える」という矛盾が境界付近で起きる。
    Clopper-Pearson なら次が厳密に成り立つ:

        lcb > 0  ⟺  片側の厳密 p 値 < alpha

    `one_sided=True` は「摂動版で落ちる」向きのみを対立仮説にする。汚染は
    一方向の現象なので既定は片側。
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha は (0, 0.5) の範囲: {alpha}")

    b = table.only_original
    c = table.only_perturbed
    n_d = table.n_discordant

    if n_d == 0:
        # 全問で結果が一致した。差がゼロであることは分かるが、検出力もゼロ。
        return McNemarResult(
            table=table, drop=0.0, p_value=1.0, ci_low=0.0, ci_high=0.0,
            lcb=0.0, alpha=alpha, one_sided=one_sided,
        )

    if one_sided:
        p_value = binomial_sf_half(b, n_d)
    else:
        p_value = min(1.0, 2.0 * binomial_sf_half(max(b, c), n_d))

    scale = n_d / table.n

    pi_low, pi_high = clopper_pearson_interval(b, n_d, alpha)
    ci_low = scale * (2.0 * pi_low - 1.0)
    ci_high = scale * (2.0 * pi_high - 1.0)

    lcb = scale * (2.0 * clopper_pearson_lower(b, n_d, alpha) - 1.0)

    return McNemarResult(
        table=table,
        drop=table.drop,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        lcb=lcb,
        alpha=alpha,
        one_sided=one_sided,
    )


def table_from_outcomes(
    original: list[bool],
    perturbed: list[bool],
) -> PairedTable:
    """問題ごとの正誤リスト2本から分割表を作る。**順序が対応していること。**"""
    if len(original) != len(perturbed):
        raise ValueError(f"長さが違う: {len(original)} vs {len(perturbed)}")

    counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for o, p in zip(original, perturbed):
        counts[(bool(o), bool(p))] += 1

    return PairedTable(
        both_correct=counts[(True, True)],
        only_original=counts[(True, False)],
        only_perturbed=counts[(False, True)],
        both_wrong=counts[(False, False)],
    )
