#!/usr/bin/env python3
"""tools/split_drop_by_injection.py — 同じ応答を「注入済み / 非注入」に分けて drop を出す。

    python tools/split_drop_by_injection.py
    python tools/split_drop_by_injection.py --json reports/split-drop.json

preregister「ラン: detector-firstlight-01」→「★ 副次の読み」の実装。

★ **推論を1回も増やさない。**`72-detector-firstlight.sh` が残した応答キャッシュを
  読み直し、同じ 4,742 × 2 の応答を**注入済み 1,896 問**と**非注入 2,846 問**に
  分けて `drop` を計算するだけである。**追加課金はゼロ。**

★ 何のためか —— 本ランの陰性対照は「fine-tune を経ていない素のベース」しかなく、
  **「注入で検出された」と「fine-tune 一般で検出された」を分離できない**
  (preregister「主張範囲 3」)。**汚染由来なら drop は注入済み側に集中し、
  fine-tune 一般の副作用なら両側に散る。**その形だけは、追加の金を払わずに見える。

★ **報告のみ。判定 A・B には入れない。**★ **事後に判定へ昇格させない。**
★ **これは陰性対照の代わりにならない。**穴を埋めるには 0% 注入で同じレシピ・
  同じ T で学習した x00 アームが要る(本ランでは作らない)。

★ `contamlab/` は**呼ぶだけ**。1行も触らない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# スクリプトとして起動されると sys.path[0] は tools/ になるので、リポジトリ直下を足す。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contamlab.benchmark import load_jsonl, split_dev_holdout, take_deterministic
from contamlab.perturb import get_perturbator, perturb_all
from contamlab.runner import ResponseCache, format_prompt, grade, set_prompt_format
from contamlab.stats.mcnemar import mcnemar_test, table_from_outcomes

# ★ 72-detector-firstlight.sh の凍結表と同じ値。**ここで規則を作らない。**
DEFAULT_ARMS = ("pcbase-swallow31-8b-x00", "df1L08t1-x40")
DEFAULT_IDS = "data/injection/df1L08t1-x40.ids"
DEFAULT_SAMPLE_N = 4742
DEFAULT_SEED = "dev-seed"
DEFAULT_ALPHA = 0.025          # M=2 の Holm 実効 α(72 と同じ)
EXPECTED_N_INJECTED = 1896


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def outcomes(cache: ResponseCache, arm: str, items, missing: list[str]) -> list[bool | None]:
    """キャッシュにある応答だけを採点する。無ければ None(呼びに行かない)。"""
    out: list[bool | None] = []
    for item in items:
        raw = cache.get(arm, format_prompt(item))
        if raw is None:
            missing.append(item.id)
            out.append(None)
        else:
            out.append(grade(item, raw).correct)
    return out


def report(name: str, orig: list[bool], pert: list[bool], alpha: float) -> dict:
    table = table_from_outcomes(orig, pert)
    result = mcnemar_test(table, alpha=alpha, one_sided=True)
    row = {
        "subset": name, "n": len(orig),
        "accuracy_original": sum(orig) / len(orig) if orig else 0.0,
        "accuracy_perturbed": sum(pert) / len(pert) if pert else 0.0,
        "drop": result.drop, "p_value": result.p_value,
        "ci_low": result.ci_low, "ci_high": result.ci_high,
        "n_discordant": table.n_discordant,
    }
    print(f"  {name:12s} n={row['n']:5d}  原文 {row['accuracy_original']:.4f} / "
          f"摂動 {row['accuracy_perturbed']:.4f}  drop {row['drop']*100:+7.2f}pt  "
          f"[{row['ci_low']*100:+6.2f}, {row['ci_high']*100:+6.2f}]  p={row['p_value']:.4g}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="注入済み / 非注入で分けた drop(追加課金ゼロ)")
    ap.add_argument("--benchmark", type=Path, default=Path("data/jmmlu.jsonl"))
    ap.add_argument("--cache", type=Path, default=None,
                    help="応答キャッシュ(既定は reports/env-tag から引く)")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--ids", type=Path, default=Path(DEFAULT_IDS),
                    help="注入された問題 id の一覧(★ 汚染アームのもの)")
    ap.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--prompt-format", default=None,
                    help="既定は reports/prompt-format から読む")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    fmt = args.prompt_format or Path("reports/prompt-format").read_text(
        encoding="utf-8").strip()
    set_prompt_format(fmt)

    cache_path = args.cache
    if cache_path is None:
        tag = Path("reports/env-tag").read_text(encoding="utf-8").strip()
        cache_path = Path(f"reports/cache.{tag}.jsonl")
    cache = ResponseCache(cache_path)

    # ★ 72 と**同じ順序で同じ問題**を組み立てる(分割 → 決定論的抽出 → 摂動)。
    dev, _ = split_dev_holdout(load_jsonl(args.benchmark))
    items = take_deterministic(dev, args.sample_n)
    perturbed = perturb_all(items, get_perturbator("shuffle_choices"), args.seed)

    injected = read_ids(args.ids)
    if len(injected) != EXPECTED_N_INJECTED:
        print(f"★ 注入 id が {len(injected)} 件 ≠ {EXPECTED_N_INJECTED} 件。"
              "注入集合の複製を疑う。")
        return 1
    flags = [item.id in injected for item in items]
    n_inj = sum(flags)
    print(f"書式 {fmt} / キャッシュ {cache_path}")
    print(f"DEV {len(items)} 問 —— 注入済み {n_inj} / 非注入 {len(items) - n_inj}\n")

    results = []
    for arm in args.arms:
        missing: list[str] = []
        orig = outcomes(cache, arm, items, missing)
        pert = outcomes(cache, arm, perturbed, missing)
        if missing:
            print(f"★ {arm}: 応答がキャッシュに無い問題が {len(missing)} 件ある。"
                  "★ **推論はしない。**72 を最後まで通してから読むこと。")
            return 1
        print(f"{arm}")
        rows = []
        for name, want in (("注入済み", True), ("非注入", False), ("全体", None)):
            sel = [i for i, f in enumerate(flags) if want is None or f == want]
            rows.append(report(name, [orig[i] for i in sel], [pert[i] for i in sel],
                               args.alpha))
        gap = rows[0]["drop"] - rows[1]["drop"]
        print(f"  → 注入済み − 非注入 = {gap*100:+.2f} pt\n")
        results.append({"arm": arm, "subsets": rows, "injected_minus_clean": gap})

    print("★ 報告のみ。判定 A・B には入れない。事後に判定へ昇格させない。")
    print("★ これは陰性対照の代わりにならない —— 0% 注入で同一レシピ学習した")
    print("   x00 アームが無い限り、fine-tune 一般の効果は分離できない。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "run": "detector-firstlight-01",
            "note": "副次の読み。報告のみで判定に入らない(preregister)。"
                    "推論は1回も増やしていない(応答キャッシュの読み直し)。",
            "prompt_format": fmt, "cache": str(cache_path),
            "sample_n": args.sample_n, "seed": args.seed, "alpha": args.alpha,
            "n_injected": n_inj, "results": results,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\n記録: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
