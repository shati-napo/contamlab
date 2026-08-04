"""④ 固定評価系。**このファイルは実験のたびに書き換えてはいけない。**

jstock-analyzer-v2 の `research/harness.py` に相当する。書き換えてよいのは
`perturb.py` だけ。測定装置が実験のたびに動くなら、その実験は比較にならない。

このモジュールが引き受けている責任は4つ。

1. **走らせる前に検出力を計算し、足りなければ止める。**
   ここが contamlab の存在理由。「100問で有意差なし → 汚染なし」という誤りを、
   実験を始める前に物理的に防ぐ。

2. **モデル間の多重比較を補正する。** モデル M 本を並べれば検定は M 回ある。

3. **試行回数 K でスコアを割り引く。** 摂動器を何種類も試して一番落ちたものを
   報告するのは p-hacking。`E[max]/σ(K)` を閾値に乗せる。

4. **測定装置自身を検査する。** `self_check()` は、汚染があると分かっているモデルで
   検出し、無いモデルで検出しないことを確かめる。これが落ちたら実験を止める。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .benchmark import Item
from .perturb import Identity, Perturbator, ShuffleChoices, perturb_all
from .runner import FakeModel, Model, Response, run_items
from .stats.heterogeneity import HeterogeneityResult, cochran_q, drop_standard_error
from .stats.mcnemar import McNemarResult, PairedTable, mcnemar_test, table_from_outcomes
from .stats.multiplicity import benjamini_hochberg, expected_max_of_k, holm
from .stats.power import PowerPlan, plan, power_at_n

# 解釈不能率がこれ以上ずれたら、条件間で採点が壊れている疑いがある。
_UNPARSED_ASYMMETRY_THRESHOLD = 0.05


class UnderpoweredError(RuntimeError):
    """検出力が足りない設計で実験を始めようとした。

    「有意差なし」を「汚染なし」と読み違える実験を、走らせる前に止める。
    どうしても走らせるなら `force_underpowered=True` を明示すること。
    **その場合、結論に検出力不足を必ず書くこと。**
    """


@dataclass(frozen=True)
class Design:
    """事前確約する設計。**結果を1つも見る前に固定する。**

    `preregister.md` に書いた値をそのまま入れる。ここを後から動かすのはカンニング。
    """

    perturbator_name: str
    seed: str
    target_effect: float
    expected_discordant_rate: float
    alpha: float = 0.05
    power: float = 0.80
    one_sided: bool = True
    n_perturbators_tried: int = 1


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    mcnemar: McNemarResult
    drop_se: float
    unparsed_original: int
    unparsed_perturbed: int
    p_holm: float = 1.0
    p_bh: float = 1.0
    deflation: float = 0.0

    @property
    def table(self) -> PairedTable:
        return self.mcnemar.table

    @property
    def adjusted_lcb(self) -> float:
        """試行回数 K を織り込んだ効果量の下限。

            adjusted_lcb = lcb − E[max]/σ(K) × SE

        jsav2 の `g_ann_lcb ≥ 基準 + E[max]/σ(K) × metric_sd` と同じ形。
        """
        return self.mcnemar.lcb - self.deflation

    @property
    def detected(self) -> bool:
        """★ 主要判定。**2つを両方満たしたときだけ「汚染あり」と言う。**

        1. 割引後の効果量下限が 0 を超える
           — 標本誤差と、摂動器を K 種類試したことによる選択バイアスを除いてなお正
        2. Holm 補正後の p 値が有意
           — モデルを M 本並べれば検定は M 回ある。名指しする以上 FWER を抑える

        1 だけでは足りない。K は摂動器の本数しか数えておらず、**モデルの本数を
        数えていない。** 実際、汚染のないモデルを3本並べただけで、1 のみの判定は
        偽陽性を出した。
        """
        return self.adjusted_lcb > 0.0 and self.p_holm < self.mcnemar.alpha


@dataclass(frozen=True)
class HarnessResult:
    design: Design
    n_items: int
    prior_plan: PowerPlan
    observed_discordant_rate: float
    observed_power: float | None
    models: list[ModelResult]
    heterogeneity: HeterogeneityResult | None
    warnings: list[str] = field(default_factory=list)

    @property
    def any_detected(self) -> bool:
        return any(m.detected for m in self.models)


def run(
    items: Sequence[Item],
    models: Sequence[Model],
    design: Design,
    perturbator: Perturbator | None = None,
    force_underpowered: bool = False,
) -> HarnessResult:
    """本番の1回。**設計を先に固定してから呼ぶこと。**"""
    if not items:
        raise ValueError("問題が1件も無い")
    if not models:
        raise ValueError("モデルが1本も無い")

    perturbator = perturbator or _resolve(design.perturbator_name)

    prior = plan(
        n=len(items),
        discordant_rate=design.expected_discordant_rate,
        target_effect=design.target_effect,
        alpha=design.alpha,
        power=design.power,
        one_sided=design.one_sided,
    )
    if not prior.adequate and not force_underpowered:
        if prior.min_detectable is None:
            detail = "そもそもこの問題数では、どんな効果量も目標検出力に届かない。"
        else:
            detail = (
                f"この設計で検出できるのは {prior.min_detectable * 100:.1f} ポイント以上のみ。"
            )
        raise UnderpoweredError(
            f"検出力が足りない。{design.target_effect * 100:.1f} ポイントの汚染を"
            f"検出力 {design.power:.2f} で見るには {prior.required_n_for_target} 問要るが、"
            f"手元には {len(items)} 問しかない。{detail}\n"
            "問題を増やすか、狙う効果量を上げること。"
            "どうしても走らせるなら force_underpowered=True を明示し、"
            "★結論に検出力不足を必ず書くこと。"
        )

    perturbed = perturb_all(list(items), perturbator, design.seed)
    _assert_paired(items, perturbed)

    warnings: list[str] = []
    if not prior.adequate:
        warnings.append(
            f"★検出力不足のまま実行した(必要 {prior.required_n_for_target} 問 / "
            f"実際 {len(items)} 問)。有意差が出なくても汚染なしとは言えない。"
        )

    results: list[ModelResult] = []
    for model in models:
        results.append(_evaluate(model, items, perturbed, design, warnings))

    results = _apply_multiplicity(results)

    observed_rate = sum(r.table.n_discordant for r in results) / (len(results) * len(items))
    observed_power = _observed_power(len(items), design, observed_rate)
    if observed_power is not None and observed_power < design.power:
        expected = design.expected_discordant_rate
        direction = "上回った" if observed_rate > expected else "下回った"
        warnings.append(
            f"★実測の不一致率 {observed_rate:.3f} が想定 {expected:.3f} を{direction}ため、"
            f"達成された検出力は {observed_power:.2f}(目標 {design.power:.2f})にとどまった。"
            "不一致率が高いほど同じ効果量を見るのに多くの問題が要る。"
            "有意差が出なくても汚染なしとは言えない。"
        )

    heterogeneity = None
    if len(results) >= 2:
        try:
            heterogeneity = cochran_q(
                {r.model_name: r.table for r in results}, alpha=design.alpha
            )
        except ValueError as exc:
            warnings.append(f"不均一さの検定ができなかった: {exc}")

    return HarnessResult(
        design=design,
        n_items=len(items),
        prior_plan=prior,
        observed_discordant_rate=observed_rate,
        observed_power=observed_power,
        models=results,
        heterogeneity=heterogeneity,
        warnings=warnings,
    )


def _resolve(name: str) -> Perturbator:
    from .perturb import get_perturbator

    return get_perturbator(name)


def _assert_paired(original: Sequence[Item], perturbed: Sequence[Item]) -> None:
    """対応がずれていないことを確かめる。ずれると意味のない数字が静かに出る。"""
    if len(original) != len(perturbed):
        raise ValueError(f"件数が違う: {len(original)} vs {len(perturbed)}")
    for a, b in zip(original, perturbed):
        if a.id != b.id:
            raise ValueError(f"対応がずれている: {a.id} vs {b.id}")
        if a.answer != b.answer:
            raise ValueError(f"摂動が正解を変えている: id={a.id}")


def _evaluate(
    model: Model,
    original: Sequence[Item],
    perturbed: Sequence[Item],
    design: Design,
    warnings: list[str],
) -> ModelResult:
    original_responses = run_items(model, original)
    perturbed_responses = run_items(model, perturbed)

    table = table_from_outcomes(
        [r.correct for r in original_responses],
        [r.correct for r in perturbed_responses],
    )
    result = mcnemar_test(table, alpha=design.alpha, one_sided=design.one_sided)
    se = drop_standard_error(table)

    unparsed_original = _count_unparsed(original_responses)
    unparsed_perturbed = _count_unparsed(perturbed_responses)
    _warn_on_unparsed_asymmetry(
        model.name, unparsed_original, unparsed_perturbed, len(original), warnings
    )

    return ModelResult(
        model_name=model.name,
        mcnemar=result,
        drop_se=se,
        unparsed_original=unparsed_original,
        unparsed_perturbed=unparsed_perturbed,
        deflation=expected_max_of_k(design.n_perturbators_tried) * se,
    )


def _count_unparsed(responses: Sequence[Response]) -> int:
    return sum(1 for r in responses if not r.parsed)


def _warn_on_unparsed_asymmetry(
    model_name: str, original: int, perturbed: int, n: int, warnings: list[str]
) -> None:
    """解釈不能率が条件間で大きく違えば、落ちたのは能力ではなく採点である。"""
    if n == 0:
        return
    gap = abs(original - perturbed) / n
    if gap > _UNPARSED_ASYMMETRY_THRESHOLD:
        warnings.append(
            f"★{model_name}: 解釈不能率が条件間で {gap * 100:.1f} ポイントずれている"
            f"(オリジナル {original} 件 / 摂動版 {perturbed} 件)。"
            "正答率の差が採点の失敗を拾っている疑いがある。"
        )


def _apply_multiplicity(results: list[ModelResult]) -> list[ModelResult]:
    """モデル間の多重比較を補正する。**モデルを1本増やすたびに検定が1回増える。**"""
    pvalues = [r.mcnemar.p_value for r in results]
    holm_adjusted = holm(pvalues)
    bh_adjusted = benjamini_hochberg(pvalues)

    return [
        ModelResult(
            model_name=r.model_name,
            mcnemar=r.mcnemar,
            drop_se=r.drop_se,
            unparsed_original=r.unparsed_original,
            unparsed_perturbed=r.unparsed_perturbed,
            p_holm=h,
            p_bh=b,
            deflation=r.deflation,
        )
        for r, h, b in zip(results, holm_adjusted, bh_adjusted)
    ]


def _observed_power(n: int, design: Design, observed_rate: float) -> float | None:
    """実測の不一致率で、実際に達成された検出力を計算する。"""
    if observed_rate <= 0.0 or design.target_effect > observed_rate:
        return None
    return power_at_n(
        n, design.target_effect, observed_rate, design.alpha, design.one_sided
    )


# --------------------------------------------------------------------------
# 測定装置の健全性チェック
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfCheck:
    name: str
    passed: bool
    detail: str


def self_check(n_items: int = 800, seed: str = "self-check") -> list[SelfCheck]:
    """検出器自身を検査する。**これが落ちたら実験を止める。**

    jstock-analyzer-v2 の `verify_cache.py` と同じ役割。測定装置が壊れていれば
    全実験が無価値になるので、本番の前に必ず通す。

    3項目:
        1. 汚染があると分かっているモデルで、検出できること
        2. 汚染が無いモデルで、検出しないこと(偽陽性を出さない)
        3. Identity 摂動(何も変えない)で、差がちょうど 0 になること
    """
    items = _synthetic_items(n_items)
    design = Design(
        perturbator_name=ShuffleChoices.name,
        seed=seed,
        target_effect=0.05,
        expected_discordant_rate=0.30,
    )
    checks: list[SelfCheck] = []

    contaminated = FakeModel(
        "contaminated", items, base_accuracy=0.25, memorized_ids=[i.id for i in items]
    )
    result = run(items, [contaminated], design)
    model = result.models[0]
    checks.append(
        SelfCheck(
            name="汚染ありを検出する",
            passed=model.detected,
            detail=f"低下 {model.mcnemar.drop * 100:.1f}pt / "
            f"割引後の下限 {model.adjusted_lcb * 100:.1f}pt",
        )
    )

    clean = FakeModel("clean", items, base_accuracy=0.60)
    result = run(items, [clean], design)
    model = result.models[0]
    checks.append(
        SelfCheck(
            name="汚染なしを検出しない",
            passed=not model.detected,
            detail=f"低下 {model.mcnemar.drop * 100:.1f}pt / "
            f"割引後の下限 {model.adjusted_lcb * 100:.1f}pt",
        )
    )

    identity_design = Design(
        perturbator_name=Identity.name,
        seed=seed,
        target_effect=0.05,
        expected_discordant_rate=0.30,
    )
    result = run(items, [clean], identity_design, force_underpowered=True)
    model = result.models[0]
    checks.append(
        SelfCheck(
            name="何も変えなければ差はちょうど0",
            passed=model.table.n_discordant == 0 and model.mcnemar.drop == 0.0,
            detail=f"不一致ペア {model.table.n_discordant} 件 / "
            f"低下 {model.mcnemar.drop * 100:.1f}pt",
        )
    )

    return checks


def _synthetic_items(n: int) -> list[Item]:
    return [
        Item(
            id=f"synthetic-{i:04d}",
            question=f"合成問題 {i:04d}: 正しいものを選べ。",
            answer=f"正解{i:04d}",
            choices=(f"正解{i:04d}", f"誤答A{i:04d}", f"誤答B{i:04d}", f"誤答C{i:04d}"),
        )
        for i in range(n)
    ]
