#!/usr/bin/env python3
"""scripts/pf1_judge.py — ラン perturbation-floor-01 の判定 D1 / D2 / D3。

    python scripts/pf1_judge.py reports/perturbation-floor-01.<tag>.json

preregister「ラン: perturbation-floor-01」→「★ 判定」の実装。

★ **規則は preregister が正である。**このスクリプトは線を1つも決めない ——
  凍結された線を機械が当てるだけである。

    D1(床の有無)  生存した全モデルで drop の片側 95% 下限 > 0 か
    D2(帯)        df1 の素のベース +1.4129pt が生存モデルの drop の
                    [最小, 最大] の内側に入るか
    D3(変換経路)  mmnga 版 swallow31-8b の drop の 95% CI が
                    +1.4129pt を含むか

⛔ **detected の二値・Cochran の Q・解釈不能率は報告のみ。判定に入れない。**
⛔ **本ランの p_holm をもって detector-firstlight-01 の判定 A を読み替えない。**
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ★ df1 の素のベース pcbase-swallow31-8b-x00 の実測 drop(2026-08-21)。
#   preregister「★ 結果(2026-08-21 実行)」が出典。
#   ⛔ **測定値であって判定規則ではない。書き換えない。**
DF1_BASE_DROP = 0.01412905946857866

# ★ 判定 D3 の対象。df1 の素のベースと**同じモデルの別 GGUF**である
#   (自作 acb1ed7e… / mmnga 版 6da177ce…)。
D3_MODEL = "swallow31-8b"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    models = data["models"]
    if not models:
        print("★ models が空である。判定できない。")
        return 1

    print()
    print("  === 実測(★ 報告。判定は下） ===")
    print(f"  {'モデル':<24}{'素':>8}{'摂動後':>9}{'drop':>10}{'95%CI':>20}{'片側下限':>10}{'p_holm':>11}  detected")
    for m in models:
        ci = f"[{m['ci_low']*100:+.2f}, {m['ci_high']*100:+.2f}]"
        print(f"  {m['name']:<24}{m['accuracy_original']:>8.4f}{m['accuracy_perturbed']:>9.4f}"
              f"{m['drop']*100:>+9.4f}pt{ci:>20}{m['lcb']*100:>+9.4f}"
              f"{m['p_holm']:>11.3g}  {m['detected']}")

    n = len(models)
    print()
    print(f"  解釈不能率(★ 報告のみ・n={data['sample']['n_items']}):")
    for m in models:
        n_items = data["sample"]["n_items"]
        print(f"    {m['name']:<24}原文 {m['unparsed_original']/n_items:6.2%}  "
              f"摂動後 {m['unparsed_perturbed']/n_items:6.2%}")

    het = data.get("heterogeneity") or {}
    if het:
        print()
        print(f"  Cochran の Q(★ 報告のみ・判定に入れない): Q={het.get('q'):.4f} "
              f"df={het.get('df')} p={het.get('p_value'):.4g} I²={het.get('i_squared'):.4f}")

    print()
    print("  === 判定(⛔ 測る前に凍結した規則。結果を見て選び直していない) ===")
    verdicts: list[tuple[str, bool | None]] = []

    # --- D1: 床の有無 -------------------------------------------------------
    # ★ 「生存した全モデルで drop の片側 95% 下限 > 0」。
    #   K=1 なので deflation は 0 で、lcb と adjusted_lcb は一致する。
    lcbs = [m["lcb"] for m in models]
    d1 = all(v > 0.0 for v in lcbs)
    n_pos = sum(1 for v in lcbs if v > 0.0)
    print()
    print(f"  D1(床の有無)  {n_pos}/{n} 本で片側 95% 下限 > 0 "
          f"(最小 {min(lcbs)*100:+.4f}pt)")
    print(f"      → {'✅ 通過' if d1 else '❌ 不通過'}")
    if d1:
        print("      ★ 言えること: 「shuffle_choices は、こちらが触っていない全モデルに対しても")
        print("        正の drop を課す。**この摂動器には汚染と無関係な床がある**」")
        print("      ⛔ 言ってはいけない: 「素のベースは汚染されていない」「判定 A は偽陽性だった」")
    else:
        print("      ★ 言えること: 「床は一様ではない。drop はモデルごとに違う」**だけ**")
        print("      ⛔ 言ってはいけない: 「落ちたモデルは汚染されている」(本ランは汚染を判定しない)")
    verdicts.append(("D1", d1))

    # --- D2: 帯 -------------------------------------------------------------
    drops = [m["drop"] for m in models]
    lo, hi = min(drops), max(drops)
    d2 = lo <= DF1_BASE_DROP <= hi
    print()
    print(f"  D2(帯)        生存モデルの drop の帯 = [{lo*100:+.4f}, {hi*100:+.4f}]pt")
    print(f"                  df1 の素のベース = {DF1_BASE_DROP*100:+.4f}pt")
    print(f"      → {'✅ 帯の内側' if d2 else '❌ 帯の外側'}")
    if d2:
        print("      ★ 言えること: 「素のベースの drop は、触っていないモデルの散らばりの内側にある。")
        print("        **n=4,742 の adjusted_lcb > 0 は、この帯を汚染と区別できない**」")
        print("      ⛔ α や n を事後に動かす根拠にしない(df1 の凍結値は凍結値のまま)")
    else:
        print("      ★ 言えること: 「素のベースの drop は他社モデルより外にある。")
        print("        ★ ②(ベース自身の汚染)がより生きた仮説になる」")
        print("      ⛔ 「ベースは汚染されている」と断定しない(反証不能な逃げ)")
    verdicts.append(("D2", d2))

    # --- D3: 変換経路 -------------------------------------------------------
    target = next((m for m in models if m["name"] == D3_MODEL), None)
    print()
    if target is None:
        print(f"  D3(変換経路)  ⛔ {D3_MODEL} がロースターに居ない(パイロットで脱落した)。")
        print("      → 判定不能。⛔ **「通らなかった」とは書かない。**")
        verdicts.append(("D3", None))
    else:
        d3 = target["ci_low"] <= DF1_BASE_DROP <= target["ci_high"]
        print(f"  D3(変換経路)  mmnga 版 {D3_MODEL} の 95% CI = "
              f"[{target['ci_low']*100:+.4f}, {target['ci_high']*100:+.4f}]pt")
        print(f"                  自作 GGUF(df1 の素のベース) = {DF1_BASE_DROP*100:+.4f}pt")
        print(f"      → {'✅ CI が含む' if d3 else '❌ CI が含まない'}")
        if d3:
            print("      ★ 言えること: 「変換経路の違いは drop を動かさない。")
            print("        **+1.4129pt はモデルと摂動の性質**である」")
            print("      ⛔ 「量子化は結果に影響しない」と一般化しない(1 モデル 1 比較である)")
        else:
            print("      ★★ 言えること: **「GGUF の変換経路が drop を動かす」** —— 第4の容疑者が立つ。")
            print("        ⛔ **本プロジェクトの自作アーム全ての比較可能性に関わる重い結果である。**")
            print("        ⛔ どちらが「正しい」とは言えない(どちらも同じモデルの正当な量子化)")
            print("        ⛔ **事後に「測り方が悪かった」と読み替えない。**")
        verdicts.append(("D3", d3))

    print()
    print("  === まとめ ===")
    for name, v in verdicts:
        mark = {True: "✅ 通過", False: "❌ 不通過", None: "— 判定不能"}[v]
        print(f"    {name}: {mark}")
    print()
    print("  ⛔ detector-firstlight-01 の判定 A・B は、本ランの結果に関わらず凍結されたままである。")
    print("  ⛔ 本ランは汚染の判定をしない。HOLDOUT も開けていない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
