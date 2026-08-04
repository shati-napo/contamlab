"""多重比較 — 「何回試したか」でスコアを割り引く。

汚染検出では検定が2種類の意味で積み上がる。

    横の多重性: モデル M 本 × ベンチマーク B 種 = M×B 回の検定
                → Holm(FWER)または Benjamini-Hochberg(FDR)で補正する

    縦の多重性: 摂動器を K 種類試して、一番落ちたものを報告する
                → **補正しないと完全な p-hacking になる**

縦のほうに既存研究の対応物が無い。金融には Bailey & López de Prado の
デフレーテッド・シャープレシオ(何本の戦略を試した末に見つけたかでスコアを割り引く)が
あるが、LLM 評価には無い。ここでそれを持ち込む。

`expected_max_of_k` は jstock-analyzer-v2 の `research/preregister.md` にある表と
同じ値を返す。同じ規律を対象だけ差し替えて使っている。
"""
from __future__ import annotations

import math

from .distributions import normal_quantile

# オイラー・マスケローニ定数。E[max] の近似式に出てくる。
_EULER_MASCHERONI = 0.5772156649015329


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni 法。FWER(1つでも誤検出する確率)を alpha 以下に抑える。

    入力の順序を保ったまま調整済み p 値を返す。**保守的。** 「1件でも偽陽性を出したくない」
    ときに使う。汚染をモデル名つきで名指しするなら、こちらを使うべき。
    """
    m = len(pvalues)
    if m == 0:
        return []
    _validate_pvalues(pvalues)

    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running_max = max(running_max, min(1.0, value))
        adjusted[idx] = running_max
    return adjusted


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg 法。FDR(検出のうち偽陽性が占める割合)を抑える。

    Holm より検出力が高いかわりに、偽陽性を一定割合許す。「どのベンチマークを
    さらに詳しく調べるか」の絞り込みには向くが、**名指しの根拠には弱い。**
    """
    m = len(pvalues)
    if m == 0:
        return []
    _validate_pvalues(pvalues)

    order = sorted(range(m), key=lambda i: pvalues[i], reverse=True)
    adjusted = [0.0] * m
    running_min = 1.0
    for position, idx in enumerate(order):
        rank = m - position  # 昇順での順位
        value = m * pvalues[idx] / rank
        running_min = min(running_min, min(1.0, value))
        adjusted[idx] = running_min
    return adjusted


def expected_max_of_k(k: int) -> float:
    """独立な K 回の試行で、真の効果がゼロでも最良値が期待値で何σ良く見えるか。

    Bailey & López de Prado の近似:

        E[max] ≈ (1 − γ)·Φ⁻¹(1 − 1/K) + γ·Φ⁻¹(1 − 1/(K·e))

    K=1 は選択が発生しないので 0。**「30回試して一番良かった」は、
    真の効果がゼロでも 2.07σ 良く見える。** これを引いてから判定する。
    """
    if k < 1:
        raise ValueError(f"試行回数は1以上: {k}")
    if k == 1:
        return 0.0

    a = normal_quantile(1.0 - 1.0 / k)
    b = normal_quantile(1.0 - 1.0 / (k * math.e))
    return (1.0 - _EULER_MASCHERONI) * a + _EULER_MASCHERONI * b


def deflated_threshold(base: float, sd: float, k: int) -> float:
    """試行回数 K を織り込んだ採用閾値。

        threshold = base + E[max]/σ(K) × sd

    jsav2 の `preregister.md` と同じ形。**結果を見てから K を数え直さないこと** —
    K は事前確約に書いた値を使う。crash も discard も、目視で捨てたものも数える。
    """
    if sd < 0.0:
        raise ValueError(f"標準偏差は非負: {sd}")
    return base + expected_max_of_k(k) * sd


def _validate_pvalues(pvalues: list[float]) -> None:
    for i, p in enumerate(pvalues):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p 値が [0, 1] の範囲外(index {i}): {p}")
