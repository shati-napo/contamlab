#!/usr/bin/env python3
"""tools/verify_power.py — `contamlab.stats.power` の**外部照合**。

    python tools/verify_power.py                 # 既定(試行 20,000・seed 20260819)
    python tools/verify_power.py --trials 2000   # 短縮

なぜ要るか: README L236-244 が自ら認めているとおり、`power.py` の既存テストは
**同じ式を両方向に使っているだけ**なので、**式の転記ミスがあっても全テストが通る。**
そしてこのリポジトリの看板の数字(「ψ=0.20 で 5pt を検出するには 493 問」
「100 問では 11pt 未満は見えない」)は、この未検証モジュール1本に依存している。

⛔ `contamlab/stats/` は編集禁止領域。**ここは検証だけを行い、何も直さない。**

3つの独立な当て方を用意する。**どれも `power.py` の式を使わない側から当てる。**

  A. 公表値照合   R パッケージ MESS の `power_mcnemar_test(method="normal")` を
                  **その論文どおりの媒介変数(paid・psi)のまま**独立に書き起こし、
                  ドキュメント記載の出力値そのものと突き合わせる。
                  MESS は Connor (1987) の正規近似を実装している。
  B. モンテカルロ 標準ライブラリの `random` だけで不一致対を生成し、
                  `contamlab.stats.mcnemar` で検定して**経験的な棄却率**を出す。
                  `power.py` を一度も呼ばないので「両方向に同じ式」問題を回避できる。
  C. 厳密列挙     n_d ~ Binom(n, psi)・b|n_d ~ Binom(n_d, pi) を全項足し上げて
                  **真の検出力を決定的に**出す。B の乱数誤差を切り離すための対照。

★ 合格線は docs/NEXT.md で着手前に凍結してある(結果を見てから動かさない):

  | 公表値照合   | `required_n` が公表値と完全一致、または差 <= 1 問 |
  | モンテカルロ | n=493 での経験的検出力が 0.80 +- 0.02             |

依存は標準ライブラリのみ(+ 検証対象の contamlab)。scipy は使わない。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

# スクリプトとして起動されると sys.path[0] は tools/ になるので、リポジトリ直下を足す。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contamlab.stats.mcnemar import PairedTable, mcnemar_test
from contamlab.stats.power import min_detectable_effect, power_at_n, required_n

_N = NormalDist()

# --- 看板の数字(README L31)。ここを動かすなら README も動かすこと ------------
BANNER_PSI = 0.20
BANNER_EFFECT = 0.05
BANNER_N = 493
BANNER_ALPHA = 0.05
BANNER_POWER = 0.80
BANNER_ONE_SIDED = True


# ============================================================================
# A. 公表値照合 — MESS (R) の独立実装を、その媒介変数のまま書き起こす
# ============================================================================
def mess_normal_power(n: float, paid: float, psi: float, sig_level: float, tside: int) -> float:
    """MESS::power_mcnemar_test の method="normal" を**逐語的に**移した。

    出典: https://github.com/cran/MESS/blob/master/R/power.mcnemar.test.R

        p.body <- quote( pnorm (
            (sqrt(n * paid) * (psi-1) - qnorm(sig.level/tside, lower.tail=FALSE)*sqrt(psi+1)) /
             sqrt((psi+1) - paid*(psi-1)^2)))

    媒介変数は MESS のもの(**contamlab のものに直さない**。直したら独立でなくなる):
        paid = 小さい方の不一致確率 p12
        psi  = 大きい方 / 小さい方 = p21 / p12
    """
    z = _N.inv_cdf(1.0 - sig_level / tside)
    numerator = math.sqrt(n * paid) * (psi - 1.0) - z * math.sqrt(psi + 1.0)
    denominator = math.sqrt((psi + 1.0) - paid * (psi - 1.0) ** 2)
    return _N.cdf(numerator / denominator)


def mess_required_n(paid: float, psi: float, power: float, sig_level: float, tside: int) -> float:
    """MESS が uniroot でやっている「検出力から n を逆算」を二分法で行う(切り上げない)。"""
    low, high = 1.0, 1e7
    for _ in range(200):
        mid = (low + high) / 2.0
        if mess_normal_power(mid, paid, psi, sig_level, tside) < power:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def to_contamlab(paid: float, psi: float) -> tuple[float, float]:
    """MESS の (paid, psi) を contamlab の (効果量 d, 不一致率 psi_c) に写す。"""
    return paid * (psi - 1.0), paid * (1.0 + psi)


# MESS のドキュメントに**印字されている出力値**。ここが唯一の外部の錨である。
# 出典: https://ekstroem.github.io/MESS/reference/power_mcnemar_test.html
PUBLISHED_CASES = [
    # (paid, psi, power, sig_level, tside, 公表された n)
    (0.125, 2.0, 0.90, 0.05, 2, 247.9973),
    (0.100, 2.0, 0.80, 0.05, 2, 233.0945),
]


def check_published() -> dict:
    """A-1: 公表された数値そのものと突き合わせる。"""
    rows = []
    for paid, psi, power, sig, tside, published in PUBLISHED_CASES:
        d, psi_c = to_contamlab(paid, psi)
        got = required_n(d, psi_c, alpha=sig, power=power, one_sided=(tside == 1))
        # required_n は切り上げるので、公表値(実数)の切り上げと比べる
        expected = math.ceil(published)
        rows.append(
            {
                "paid": paid,
                "psi": psi,
                "power": power,
                "sig_level": sig,
                "tside": tside,
                "discordant_rate": psi_c,
                "effect": d,
                "published_n": published,
                "published_ceil": expected,
                "contamlab_required_n": got,
                "diff": got - expected,
                "pass": abs(got - expected) <= 1,
            }
        )
    return {
        "name": "A-1 公表値照合(MESS ドキュメント記載値)",
        "rows": rows,
        "pass": all(r["pass"] for r in rows),
    }


def check_against_mess_grid() -> dict:
    """A-2: 格子上で MESS 実装と突き合わせる。転記ミスなら必ずここでずれる。"""
    rows = []
    grid = []
    for paid in (0.02, 0.05, 0.075, 0.10, 0.125, 0.20, 0.30):
        for psi in (1.2, 1.5, 2.0, 3.0, 5.0):
            if paid * (1.0 + psi) > 1.0:
                continue
            for power in (0.80, 0.90):
                for tside in (1, 2):
                    grid.append((paid, psi, power, 0.05, tside))

    worst_n = 0
    worst_power = 0.0
    for paid, psi, power, sig, tside in grid:
        d, psi_c = to_contamlab(paid, psi)
        n_mess = mess_required_n(paid, psi, power, sig, tside)
        n_cl = required_n(d, psi_c, alpha=sig, power=power, one_sided=(tside == 1))
        dn = abs(n_cl - math.ceil(n_mess))
        worst_n = max(worst_n, dn)

        # power_at_n も直接当てる(n は整数で揃える)
        n_int = max(1, math.ceil(n_mess))
        p_mess = mess_normal_power(n_int, paid, psi, sig, tside)
        p_cl = power_at_n(n_int, d, psi_c, alpha=sig, one_sided=(tside == 1))
        dp = abs(p_mess - p_cl)
        worst_power = max(worst_power, dp)

        if dn > 1 or dp > 1e-9:
            rows.append(
                {
                    "paid": paid,
                    "psi": psi,
                    "power": power,
                    "tside": tside,
                    "n_mess": n_mess,
                    "n_contamlab": n_cl,
                    "power_mess": p_mess,
                    "power_contamlab": p_cl,
                }
            )

    return {
        "name": "A-2 MESS 独立実装との格子照合",
        "cases": len(grid),
        "max_abs_diff_required_n": worst_n,
        "max_abs_diff_power_at_n": worst_power,
        "mismatches": rows,
        "pass": worst_n <= 1 and worst_power <= 1e-9,
    }


# ============================================================================
# B. モンテカルロ — 標準ライブラリの random だけ。power.py を一度も呼ばない
# ============================================================================
def asymptotic_reject(b: int, c: int, alpha: float, one_sided: bool) -> bool:
    """Connor の式が想定している検定そのもの(McNemar の正規近似)。

    z = (b - c) / sqrt(b + c) を標準正規で判定する。**連続性補正は入れない**
    (Connor 1987 の標本サイズ式は補正なしの検定に対応するため)。
    """
    n_d = b + c
    if n_d == 0:
        return False
    z = (b - c) / math.sqrt(n_d)
    crit = _N.inv_cdf(1.0 - alpha) if one_sided else _N.inv_cdf(1.0 - alpha / 2.0)
    return (z > crit) if one_sided else (abs(z) > crit)


def monte_carlo(
    n: int,
    psi: float,
    effect: float,
    alpha: float,
    one_sided: bool,
    trials: int,
    seed: int,
) -> dict:
    """不一致対を1問ずつ引いて、経験的な棄却率を出す。

    psi = p10 + p01・d = p10 - p01 なので p10 = (psi+d)/2・p01 = (psi-d)/2。
    残り 1-psi は一致セル。**一致セルの内訳は McNemar の判定に一切効かない**ので
    both_correct 側に寄せる(効果量 drop の分母 n は変わらない)。
    """
    p10 = (psi + effect) / 2.0  # only_original(オリジナルだけ正解)= b
    p01 = (psi - effect) / 2.0  # only_perturbed = c
    assert p10 >= 0.0 and p01 >= 0.0 and p10 + p01 <= 1.0

    rng = random.Random(seed)
    cut10 = p10
    cut01 = p10 + p01

    # 厳密検定は (b, c) だけで判定が決まる。n は固定なので memo 化してよい。
    memo_exact_p: dict[tuple[int, int], bool] = {}
    memo_exact_lcb: dict[tuple[int, int], bool] = {}

    hits_exact_p = 0
    hits_exact_lcb = 0
    hits_asym = 0
    disagreements = 0

    for _ in range(trials):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < cut10:
                b += 1
            elif u < cut01:
                c += 1
        key = (b, c)
        if key not in memo_exact_p:
            table = PairedTable(
                both_correct=n - b - c, only_original=b, only_perturbed=c, both_wrong=0
            )
            res = mcnemar_test(table, alpha=alpha, one_sided=one_sided)
            memo_exact_p[key] = res.p_value < alpha
            memo_exact_lcb[key] = res.lcb > 0.0
        if memo_exact_p[key]:
            hits_exact_p += 1
        if memo_exact_lcb[key]:
            hits_exact_lcb += 1
        if memo_exact_p[key] != memo_exact_lcb[key]:
            disagreements += 1
        if asymptotic_reject(b, c, alpha, one_sided):
            hits_asym += 1

    def se(k: int) -> float:
        p = k / trials
        return math.sqrt(max(p * (1.0 - p), 0.0) / trials)

    return {
        "n": n,
        "psi": psi,
        "effect": effect,
        "alpha": alpha,
        "one_sided": one_sided,
        "trials": trials,
        "seed": seed,
        "p10": p10,
        "p01": p01,
        "empirical_power_exact_pvalue": hits_exact_p / trials,
        "empirical_power_exact_lcb": hits_exact_lcb / trials,
        "empirical_power_asymptotic": hits_asym / trials,
        "mc_standard_error": se(hits_exact_p),
        "lcb_vs_pvalue_disagreements": disagreements,
        "distinct_bc_states": len(memo_exact_p),
    }


# ============================================================================
# C. 厳密列挙 — 乱数を使わずに真の検出力を出す(B の対照)
# ============================================================================
def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def exact_power(n: int, psi: float, effect: float, alpha: float, one_sided: bool) -> dict:
    """n_d ~ Binom(n, psi)・b|n_d ~ Binom(n_d, pi) を全項足し上げる。

    pi = p10 / psi。**これは近似ではなく、その検定の真の検出力**である。
    """
    p10 = (psi + effect) / 2.0
    pi = p10 / psi

    def rejects(b: int, n_d: int) -> bool:
        """**本物の `mcnemar_test` を呼ぶ。**ここを自前の式で代用したら検証にならない。"""
        table = PairedTable(
            both_correct=n - n_d, only_original=b, only_perturbed=n_d - b, both_wrong=0
        )
        return mcnemar_test(table, alpha=alpha, one_sided=one_sided).p_value < alpha

    def critical_b(n_d: int) -> int:
        """棄却される最小の b。p 値 = P(X>=b) は b について単調減少なので二分法でよい。

        全 b を回すと n=493 では table が 12 万個になり、Clopper-Pearson の二分法が
        毎回走って現実的な時間で終わらない。単調性を使って境界だけ探す。
        """
        if not rejects(n_d, n_d):
            return n_d + 1  # どんな b でも棄却できない
        low, high = 0, n_d  # low は棄却しない側、high は棄却する側
        if rejects(0, n_d):
            return 0
        while high - low > 1:
            mid = (low + high) // 2
            if rejects(mid, n_d):
                high = mid
            else:
                low = mid
        # 単調性の抜き取り確認(境界の両側)
        assert rejects(high, n_d) and not rejects(low, n_d), f"単調でない: n_d={n_d}"
        return high

    total_exact = 0.0
    total_asym = 0.0
    covered = 0.0
    for n_d in range(0, n + 1):
        log_w = _log_binom_pmf(n_d, n, psi)
        if log_w < -60.0:  # 寄与が 1e-26 未満の裾は落とす
            continue
        w = math.exp(log_w)
        covered += w
        if n_d == 0:
            continue
        b_crit = critical_b(n_d)
        for b in range(0, n_d + 1):
            log_q = _log_binom_pmf(b, n_d, pi)
            if log_q < -60.0:
                continue
            q = math.exp(log_q)
            if b >= b_crit:
                total_exact += w * q
            if asymptotic_reject(b, n_d - b, alpha, one_sided):
                total_asym += w * q
    return {
        "exact_test_true_power": total_exact,
        "asymptotic_test_true_power": total_asym,
        "probability_mass_covered": covered,
    }


# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="contamlab.stats.power の外部照合")
    ap.add_argument("--trials", type=int, default=20000, help="モンテカルロ試行回数")
    ap.add_argument("--seed", type=int, default=20260819, help="乱数 seed(記録する)")
    ap.add_argument("--skip-exact", action="store_true", help="C(厳密列挙)を飛ばす")
    ap.add_argument(
        "--calibrate-exact",
        action="store_true",
        help="D: 厳密検定で実際に検出力 0.80 に届く n と、n=100 での真の検出下限を出す",
    )
    ap.add_argument("--json", type=str, default=None, help="結果を JSON で書き出す先")
    args = ap.parse_args()

    out: dict = {
        "banner": {
            "psi": BANNER_PSI,
            "effect": BANNER_EFFECT,
            "claimed_n": BANNER_N,
            "alpha": BANNER_ALPHA,
            "power": BANNER_POWER,
            "one_sided": BANNER_ONE_SIDED,
        }
    }

    print("=" * 78)
    print("contamlab.stats.power 外部照合")
    print("=" * 78)

    # --- 看板の数字そのもの ------------------------------------------------
    got_n = required_n(
        BANNER_EFFECT, BANNER_PSI, alpha=BANNER_ALPHA, power=BANNER_POWER, one_sided=BANNER_ONE_SIDED
    )
    # MESS の媒介変数へ: p10=(psi+d)/2, p01=(psi-d)/2, paid=小さい方
    p10 = (BANNER_PSI + BANNER_EFFECT) / 2.0
    p01 = (BANNER_PSI - BANNER_EFFECT) / 2.0
    n_mess = mess_required_n(p01, p10 / p01, BANNER_POWER, BANNER_ALPHA, 1)
    mde100 = min_detectable_effect(100, BANNER_PSI, BANNER_ALPHA, BANNER_POWER, True)
    out["banner"]["contamlab_required_n"] = got_n
    out["banner"]["mess_required_n"] = n_mess
    out["banner"]["min_detectable_at_100"] = mde100
    print(f"\n[看板] psi={BANNER_PSI} で {BANNER_EFFECT*100:.0f}pt を検出するのに必要な問題数")
    print(f"       contamlab required_n : {got_n}")
    print(f"       MESS 独立実装        : {n_mess:.4f}  (切り上げ {math.ceil(n_mess)})")
    print(f"       README の主張        : {BANNER_N}")
    print(f"[看板] n=100 での検出可能な最小効果量: {mde100*100:.2f} pt (README: 11 pt)")

    # --- A -----------------------------------------------------------------
    a1 = check_published()
    out["A1_published"] = a1
    print(f"\n--- {a1['name']} ---")
    for r in a1["rows"]:
        mark = "OK " if r["pass"] else "NG "
        print(
            f"  {mark} paid={r['paid']} psi={r['psi']} power={r['power']} "
            f"{'片側' if r['tside']==1 else '両側'}: "
            f"公表 {r['published_n']} (切上 {r['published_ceil']}) vs "
            f"contamlab {r['contamlab_required_n']}  差 {r['diff']}"
        )

    a2 = check_against_mess_grid()
    out["A2_grid"] = a2
    print(f"\n--- {a2['name']} ---")
    print(f"  照合ケース数            : {a2['cases']}")
    print(f"  required_n の最大差     : {a2['max_abs_diff_required_n']} 問")
    print(f"  power_at_n の最大差     : {a2['max_abs_diff_power_at_n']:.3e}")
    if a2["mismatches"]:
        for m in a2["mismatches"][:10]:
            print(f"    NG {m}")

    # --- B -----------------------------------------------------------------
    print(f"\n--- B モンテカルロ(n={BANNER_N}・試行 {args.trials}・seed {args.seed})---")
    mc = monte_carlo(
        BANNER_N, BANNER_PSI, BANNER_EFFECT, BANNER_ALPHA, BANNER_ONE_SIDED, args.trials, args.seed
    )
    out["B_monte_carlo"] = mc
    print(f"  p10={mc['p10']:.4f}  p01={mc['p01']:.4f}  (b/c の生成確率)")
    print(
        f"  経験的検出力 - contamlab 厳密検定 p<alpha : {mc['empirical_power_exact_pvalue']:.4f}"
        f"  (95% 幅 +-{mc['mc_standard_error']*1.96:.4f})"
    )
    print(f"  経験的検出力 - contamlab 主要判定 lcb>0   : {mc['empirical_power_exact_lcb']:.4f}")
    print(f"  経験的検出力 - 正規近似(Connor が想定)  : {mc['empirical_power_asymptotic']:.4f}")
    print(
        f"  lcb>0 と p<alpha の食い違い件数           : {mc['lcb_vs_pvalue_disagreements']}"
        f"  (mcnemar.py の docstring は 0 と主張)"
    )

    # --- C -----------------------------------------------------------------
    if not args.skip_exact:
        print(f"\n--- C 厳密列挙(乱数なし・n={BANNER_N})---")
        ex = exact_power(BANNER_N, BANNER_PSI, BANNER_EFFECT, BANNER_ALPHA, BANNER_ONE_SIDED)
        out["C_exact_enumeration"] = ex
        print(f"  厳密条件付き検定の真の検出力 : {ex['exact_test_true_power']:.4f}")
        print(f"  正規近似検定の真の検出力     : {ex['asymptotic_test_true_power']:.4f}")
        print(f"  power.py の予言              : {BANNER_POWER:.4f}")
        print(f"  (足し上げた確率質量          : {ex['probability_mass_covered']:.6f})")

    # --- D 厳密検定での再較正 ------------------------------------------------
    if args.calibrate_exact:
        print("\n--- D 厳密検定での再較正(power.py は使わない)---")
        # D-1: 検出力 0.80 に本当に届く n
        low, high = BANNER_N, 4 * BANNER_N
        while high - low > 1:
            mid = (low + high) // 2
            p = exact_power(mid, BANNER_PSI, BANNER_EFFECT, BANNER_ALPHA, BANNER_ONE_SIDED)
            if p["exact_test_true_power"] < BANNER_POWER:
                low = mid
            else:
                high = mid
        n_needed = high
        print(f"  psi=0.20 で 5pt を検出力 0.80 で見るのに要る n : {n_needed}"
              f"  (power.py の答え {BANNER_N} / 差 +{n_needed - BANNER_N})")

        # D-2: n=100 での真の検出下限(効果量を上げていって 0.80 に届く点)
        lo_d, hi_d = 0.0, BANNER_PSI * (1.0 - 1e-9)
        for _ in range(24):
            mid_d = (lo_d + hi_d) / 2.0
            p = exact_power(100, BANNER_PSI, mid_d, BANNER_ALPHA, BANNER_ONE_SIDED)
            if p["exact_test_true_power"] < BANNER_POWER:
                lo_d = mid_d
            else:
                hi_d = mid_d
        print(f"  n=100 で厳密検定が実際に見える最小効果量      : {hi_d*100:.2f} pt"
              f"  (power.py の答え {mde100*100:.2f} pt)")
        out["D_recalibration"] = {
            "required_n_exact_test": n_needed,
            "required_n_power_py": BANNER_N,
            "min_detectable_at_100_exact_test": hi_d,
            "min_detectable_at_100_power_py": mde100,
        }

    # --- 判定 ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("判定(合格線は docs/NEXT.md で着手前に凍結)")
    print("=" * 78)
    verdicts = {}
    verdicts["公表値照合"] = a1["pass"] and a2["pass"] and abs(got_n - BANNER_N) <= 1
    lo = BANNER_POWER - 0.02
    hi = BANNER_POWER + 0.02
    mc_pass = lo <= mc["empirical_power_exact_pvalue"] <= hi
    verdicts["モンテカルロ(contamlab 厳密検定)"] = mc_pass
    for k, v in verdicts.items():
        print(f"  {'[合格]' if v else '[不合格]'}  {k}")
    out["verdicts"] = verdicts

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON: {args.json}")

    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
