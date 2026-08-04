"""結果の整形。

方針は1つ。**点推定を単独で出さない。** 低下幅は必ず区間と一緒に、
判定は必ず検出力と一緒に出す。数字だけ抜き出して引用されると、このツールが
なくそうとしているもの(裸のスコア)を自分で生産することになる。
"""
from __future__ import annotations

from .harness import HarnessResult, ModelResult, SelfCheck

_RULE = "─" * 68


def _pt(value: float, signed: bool = True) -> str:
    """比率をパーセントポイント表記にする。"""
    return f"{value * 100:+.1f}" if signed else f"{value * 100:.1f}"


def format_result(result: HarnessResult) -> str:
    design = result.design
    prior = result.prior_plan

    lines = [
        _RULE,
        "contamlab — 汚染検査の結果",
        _RULE,
        "",
        "■ 設計(事前確約)",
        f"  摂動器            : {design.perturbator_name}",
        f"  シード            : {design.seed}",
        f"  狙う効果量        : {_pt(design.target_effect, signed=False)} ポイント",
        f"  想定した不一致率  : {design.expected_discordant_rate:.3f}",
        f"  有意水準          : {design.alpha}({'片側' if design.one_sided else '両側'})",
        f"  摂動器の試行 K    : {design.n_perturbators_tried}"
        f"(割引 {_pt(_deflation_ratio(result), signed=False)}σ 相当)",
        "",
        "■ 標本と検出力",
        f"  問題数            : {result.n_items}",
    ]

    if prior.required_n_for_target is not None:
        lines.append(f"  必要問題数        : {prior.required_n_for_target}")
    lines.append(
        f"  検出可能な最小    : {prior.describe_min_detectable()}(想定の不一致率での事前値)"
    )
    lines.append(f"  実測の不一致率    : {result.observed_discordant_rate:.3f}")
    if result.observed_power is None:
        lines.append("  達成された検出力  : 算出不能(不一致率が狙う効果量を下回った)")
    else:
        lines.append(f"  達成された検出力  : {result.observed_power:.3f}")

    lines.extend(["", "■ モデル別", "", _model_header()])
    for model in result.models:
        lines.append(_model_row(model))

    lines.extend(["", "■ モデル間の不均一さ(差の差)"])
    if result.heterogeneity is None:
        lines.append("  モデルが1本なので判定できない。摂動が難易度を変えただけの")
        lines.append("  可能性を排除するには、2本以上を横に並べる必要がある。")
    else:
        h = result.heterogeneity
        lines.append(
            f"  Q = {h.q_statistic:.2f}, df = {h.df}, p = {h.p_value:.4f}, "
            f"I² = {h.i_squared:.3f}"
        )
        lines.append(f"  → {h.interpretation()}")
        if h.excluded:
            lines.append(f"  除外(不一致ペア0件): {', '.join(h.excluded)}")

    lines.extend(["", "■ 警告"])
    if result.warnings:
        lines.extend(f"  {w}" for w in result.warnings)
    else:
        lines.append("  なし")

    lines.extend(
        [
            "",
            _RULE,
            "判定は「割引後の下限 > 0」かつ「Holm 補正後 p < α」の両方を満たすときのみ。",
            "前者は摂動器の試行 K、後者はモデルの本数を補正している。片方だけでは足りない。",
            "信頼区間は不一致ペア数を固定した条件付き区間(そのぶん保守的)。",
            _RULE,
        ]
    )
    return "\n".join(lines)


def _deflation_ratio(result: HarnessResult) -> float:
    from .stats.multiplicity import expected_max_of_k

    return expected_max_of_k(result.design.n_perturbators_tried) / 100.0


def _model_header() -> str:
    return (
        f"  {'モデル':<20}{'低下':>8}{'95%CI':>20}{'割引後下限':>12}"
        f"{'Holm':>9}{'判定':>8}"
    )


def _model_row(model: ModelResult) -> str:
    m = model.mcnemar
    interval = f"[{_pt(m.ci_low)}, {_pt(m.ci_high)}]"
    verdict = "★汚染" if model.detected else "—"
    return (
        f"  {model.model_name:<20}{_pt(m.drop):>8}{interval:>20}"
        f"{_pt(model.adjusted_lcb):>12}{model.p_holm:>9.4f}{verdict:>8}"
    )


def result_to_dict(result: HarnessResult) -> dict:
    """機械可読な出力。**再現に必要な設計値を全部含める。**"""
    design = result.design
    return {
        "design": {
            "perturbator": design.perturbator_name,
            "seed": design.seed,
            "target_effect": design.target_effect,
            "expected_discordant_rate": design.expected_discordant_rate,
            "alpha": design.alpha,
            "power": design.power,
            "one_sided": design.one_sided,
            "n_perturbators_tried": design.n_perturbators_tried,
        },
        "sample": {
            "n_items": result.n_items,
            "required_n": result.prior_plan.required_n_for_target,
            "min_detectable_effect": result.prior_plan.min_detectable,
            "observed_discordant_rate": result.observed_discordant_rate,
            "observed_power": result.observed_power,
        },
        "models": [
            {
                "name": m.model_name,
                "table": {
                    "both_correct": m.table.both_correct,
                    "only_original": m.table.only_original,
                    "only_perturbed": m.table.only_perturbed,
                    "both_wrong": m.table.both_wrong,
                },
                "accuracy_original": m.table.accuracy_original,
                "accuracy_perturbed": m.table.accuracy_perturbed,
                "drop": m.mcnemar.drop,
                "drop_se": m.drop_se,
                "ci_low": m.mcnemar.ci_low,
                "ci_high": m.mcnemar.ci_high,
                "lcb": m.mcnemar.lcb,
                "deflation": m.deflation,
                "adjusted_lcb": m.adjusted_lcb,
                "p_value": m.mcnemar.p_value,
                "p_holm": m.p_holm,
                "p_bh": m.p_bh,
                "unparsed_original": m.unparsed_original,
                "unparsed_perturbed": m.unparsed_perturbed,
                "detected": m.detected,
            }
            for m in result.models
        ],
        "heterogeneity": (
            None
            if result.heterogeneity is None
            else {
                "q": result.heterogeneity.q_statistic,
                "df": result.heterogeneity.df,
                "p_value": result.heterogeneity.p_value,
                "i_squared": result.heterogeneity.i_squared,
                "pooled_drop": result.heterogeneity.pooled_drop,
                "heterogeneous": result.heterogeneity.heterogeneous,
                "interpretation": result.heterogeneity.interpretation(),
                "excluded": result.heterogeneity.excluded,
            }
        ),
        "warnings": list(result.warnings),
    }


def format_self_check(checks: list[SelfCheck]) -> str:
    lines = [_RULE, "測定装置の健全性チェック", _RULE, ""]
    for check in checks:
        mark = "OK  " if check.passed else "FAIL"
        lines.append(f"  [{mark}] {check.name}")
        lines.append(f"         {check.detail}")
    lines.append("")
    if all(c.passed for c in checks):
        lines.append("  すべて通過。実験を進めてよい。")
    else:
        lines.append("  ★落ちている。測定装置が壊れていれば全実験が無価値。実験を止めること。")
    lines.append(_RULE)
    return "\n".join(lines)
