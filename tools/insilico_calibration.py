#!/usr/bin/env python3
"""tools/insilico_calibration.py — FakeModel で**統計層だけを端から端まで通す**。

    python tools/insilico_calibration.py
    python tools/insilico_calibration.py --n-items 4742 --json reports/insilico/run.json

作業3(docs/NEXT.md)。目的はただ1つ ——

    **`drop` / `p_value` / `adjusted_lcb` / `p_holm` / `detected` / Cochran の Q が
      配管として出てくることを、GPU に金を払う前に確認する。**

⛔ **これは較正であって測定ではない。**FakeModel は「暗記した問題はオリジナル提示形の
ときだけ必ず正解し、摂動版では素の能力に落ちる」という**汚染の定義そのもの**を
そのまま実装したものなので、ここで曲線が出るのは当たり前である。
**★ ここの数字を成果として書かない。**「実モデルで汚染を検出した」とは何も言えない。

⛔ `contamlab/` は触らない。ここは呼ぶだけ。

なぜ CLI の `fake:NAME:ACC[:memorized]` ではなくこの脚本かというと、CLI の指定は
**全問暗記か暗記ゼロの2値**しか作れず、注入率の梯子(0/2/5/10/20/40%)を張れないため。
CLI 経路そのものの疎通は、別にこれで確認してある(2026-08-19 に通過):

    python -m contamlab run --synthetic 2000 --seed insilico-cli \
      --perturbator shuffle_choices --target-effect 0.05 \
      --expected-discordant-rate 0.405 --k 1 --yes \
      --model fake:cli-clean:0.45 --model fake:cli-dirty:0.45:memorized \
      --json reports/insilico/cli-smoke-20260819.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# スクリプトとして起動されると sys.path[0] は tools/ になるので、リポジトリ直下を足す。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contamlab.harness import Design, run
from contamlab.harness import _synthetic_items  # 合成問題(動作確認用の既存経路)
from contamlab.runner import FakeModel

# scripts/70-positive-control.sh の PC_ARMS と同じ梯子。**同じ土俵に置くため揃える。**
ARM_RATES = {
    "sim-x00": 0.00,
    "sim-x02": 0.02,
    "sim-x05": 0.05,
    "sim-x10": 0.10,
    "sim-x20": 0.20,
    "sim-x40": 0.40,
}


def build_arms(items, base_accuracy: float, seed: int) -> list[FakeModel]:
    """注入率ごとに FakeModel を1本ずつ作る。

    暗記させる問題は**アーム間で入れ子**にする(x02 ⊂ x05 ⊂ x10 ⊂ …)。
    tools/build_injection_sets.py が実物の注入集合でやっているのと同じ作りにして、
    アーム間の差が「どの問題を選んだか」ではなく**注入率だけ**で決まるようにする。
    """
    rng = random.Random(seed)
    order = [i.id for i in items]
    rng.shuffle(order)

    arms = []
    for name, rate in ARM_RATES.items():
        k = round(len(order) * rate)
        arms.append(
            FakeModel(
                name,
                items,
                base_accuracy=base_accuracy,
                memorized_ids=order[:k],
                seed=f"insilico-{seed}",
            )
        )
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description="FakeModel での in-silico 較正(課金ゼロ)")
    ap.add_argument("--n-items", type=int, default=4742, help="問題数(既定は DEV 全量と同じ)")
    ap.add_argument("--base-accuracy", type=float, default=0.45, help="素の正答率")
    ap.add_argument("--seed", type=int, default=20260819, help="暗記集合を選ぶ乱数 seed")
    ap.add_argument("--design-seed", default="insilico-1", help="摂動のシード")
    ap.add_argument("--expected-psi", type=float, default=0.4050, help="想定 psi(pc-01 の採用値)")
    ap.add_argument("--target-effect", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.008333, help="Holm の実効 alpha(M=6)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("in-silico 較正(FakeModel・課金 $0・GPU 0 台)")
    print("⛔ これは較正であって測定ではない。ここの数字を成果として書かない。")
    print("=" * 78)

    items = _synthetic_items(args.n_items)
    arms = build_arms(items, args.base_accuracy, args.seed)

    design = Design(
        perturbator_name="shuffle_choices",
        seed=args.design_seed,
        target_effect=args.target_effect,
        expected_discordant_rate=args.expected_psi,
        alpha=args.alpha,
        n_perturbators_tried=1,
    )
    print(f"\n  問題数 {args.n_items} / 素の正答率 {args.base_accuracy} / "
          f"alpha {args.alpha} / 想定 psi {args.expected_psi}")
    print(f"  アーム: {' '.join(ARM_RATES)}")

    result = run(items, arms, design)

    print(f"\n  実測 psi = {result.observed_discordant_rate:.4f}"
          f" / 事前の最小検出可能 = {result.prior_plan.describe_min_detectable()}")
    if result.observed_power is not None:
        print(f"  実測 psi での検出力 = {result.observed_power:.4f}")

    print()
    header = (f"  {'アーム':10s} {'注入率':>6s} {'drop':>9s} {'lcb':>9s} "
              f"{'割引後下限':>11s} {'p_value':>10s} {'p_holm':>10s}  検出")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    detected_rates = []
    for m in sorted(result.models, key=lambda m: ARM_RATES.get(m.model_name, 0.0)):
        rate = ARM_RATES.get(m.model_name, 0.0)
        print(f"  {m.model_name:10s} {rate:6.0%} {m.mcnemar.drop*100:+8.2f}pt "
              f"{m.mcnemar.lcb*100:+8.2f}pt {m.adjusted_lcb:+11.4f} "
              f"{m.mcnemar.p_value:10.4g} {m.p_holm:10.4g}  "
              f"{'★ 検出' if m.detected else '—'}")
        rows.append({
            "arm": m.model_name,
            "injection_rate": rate,
            "drop": m.mcnemar.drop,
            "lcb": m.mcnemar.lcb,
            "adjusted_lcb": m.adjusted_lcb,
            "p_value": m.mcnemar.p_value,
            "p_holm": m.p_holm,
            "p_bh": m.p_bh,
            "deflation": m.deflation,
            "n_discordant": m.table.n_discordant,
            "detected": m.detected,
        })
        if m.detected:
            detected_rates.append(rate)

    # --- 配管が通ったかどうかの判定 ----------------------------------------
    print("\n" + "=" * 78)
    print("配管の判定(⛔ 汚染検出の成否ではない。値が出たかどうかだけを見る)")
    print("=" * 78)
    checks: dict[str, bool] = {}

    checks["全アームで drop / p_value / adjusted_lcb / p_holm が数値として出た"] = all(
        all(isinstance(r[k], float) for k in ("drop", "p_value", "adjusted_lcb", "p_holm"))
        for r in rows
    )
    x00 = next(r for r in rows if r["arm"] == "sim-x00")
    x40 = next(r for r in rows if r["arm"] == "sim-x40")
    checks["注入ゼロのアームを検出しない(偽陽性を出さない)"] = not x00["detected"]
    checks["注入 40% のアームを検出する"] = x40["detected"]
    checks["drop が注入率について単調非減少"] = all(
        rows[i]["drop"] <= rows[i + 1]["drop"] + 1e-12 for i in range(len(rows) - 1)
    )
    checks["Holm 補正が効いている(p_holm >= p_value)"] = all(
        r["p_holm"] >= r["p_value"] - 1e-12 for r in rows
    )
    checks["K 割引が effect を必ず削る(adjusted_lcb <= lcb)"] = all(
        r["adjusted_lcb"] <= r["lcb"] + 1e-12 for r in rows
    )

    h = result.heterogeneity
    checks["Cochran の Q が計算された"] = h is not None
    if h is not None:
        print(f"\n  Cochran の Q = {h.q_statistic:.4f}  df={h.df}  p={h.p_value:.4g}  "
              f"I^2={h.i_squared:.4f}  "
              f"{'(不均一)' if h.heterogeneous else '(不均一とは言えない)'}")

    print()
    for name, ok in checks.items():
        print(f"  {'[OK]' if ok else '[NG]'}  {name}")

    if detected_rates:
        pos = [r for r in detected_rates if r > 0]
        floor = min(pos) if pos else 0.0
        below = [r for r in ARM_RATES.values() if r < floor]
        print(f"\n  in-silico の検出下限の帯: {max(below) if below else 0:.0%} では検出せず、"
              f"{floor:.0%} で検出")
    else:
        print("\n  ★ どのアームも検出しなかった。**配管は通ったが感度が出ていない**")

    for w in result.warnings:
        print(f"  ▲ {w}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "in-silico calibration (FakeModel). NOT a measurement.",
            "n_items": args.n_items,
            "base_accuracy": args.base_accuracy,
            "seed": args.seed,
            "design": {
                "perturbator": design.perturbator_name,
                "seed": design.seed,
                "target_effect": design.target_effect,
                "expected_discordant_rate": design.expected_discordant_rate,
                "alpha": design.alpha,
                "n_perturbators_tried": design.n_perturbators_tried,
            },
            "observed_discordant_rate": result.observed_discordant_rate,
            "observed_power": result.observed_power,
            "arms": rows,
            "heterogeneity": (
                None if h is None
                else {"q_statistic": h.q_statistic, "df": h.df, "p_value": h.p_value,
                      "i_squared": h.i_squared, "pooled_drop": h.pooled_drop,
                      "excluded": h.excluded, "heterogeneous": h.heterogeneous}
            ),
            "plumbing_checks": checks,
            "warnings": result.warnings,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON: {args.json}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
