#!/usr/bin/env python3
"""scripts/df1_gate.py — ラン detector-firstlight-01 の関門を**機械に守らせる**。

    python3 scripts/df1_gate.py base    --check reports/df1-check-base.txt   # G0
    python3 scripts/df1_gate.py inputs  --arms df1t1-x40 df1t2-x40 df1t3-x40 # G1
    python3 scripts/df1_gate.py lambda  --arms df1L08t1-x40 ...              # G2
    python3 scripts/df1_gate.py anomaly --check reports/df1-check-t1.txt     # G3
    python3 scripts/df1_gate.py gate-b  --check reports/df1-check.txt        # G4
    python3 scripts/df1_gate.py gate-c  --check reports/df1-check.txt        # G5
    python3 scripts/df1_gate.py report-a --check reports/df1-check.txt       # ★ 報告のみ

★ ここに書いてある数字は**1つも新しくない。**すべて preregister
  「## ラン: detector-firstlight-01」が測る前に凍結した値である。

    [0.562, 0.658] / 1.6% —— 第0段の帯(pc-04〜ll-01 と同一)
    50%   —— 安価な異常検知(cc-01 の関門をそのまま引き継いだ)
    0.30  —— 合格条件 b(非注入群 正解率)。pc-02 から1文字も変えていない
    5%    —— 合格条件 c(解釈不能率 両群)。同上
    10pt  —— 合格条件 a(差)。同上。★ **ただし本ランでは関門ではない**(下記)
    1.686 —— k=3 の片側95%(t(2) = 2.920 -> 2.920 / sqrt(3) = 1.686)

★ **`a` は関門にしない**(preregister「★ a を関門にしない」)。
  `report-a` は **判定を印字するだけで、常に成功で返る。**★ 止める口を持たない。
  ★ ただし a が頑健に合格しなかった場合、判定 B の結果を
  「検出器が汚染を検出した」とは書かない —— これは人が守る縛りである。

★ 65-manipulation-check.sh の出力を読む関数と、条件 c の読みは
  **cc-01 の実装をそのまま呼ぶ。**同じ出力を別々に読む経路を2つ作らないため。
  ★ `scripts/cc01_gate.py` は1行も変えていない。

★ この道具は判定を**足さない。**規則の側を動かせないように、閾値はここに
  定数として書き、コマンドラインからは受け取らない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc01_gate import cmd_gate_c, cmd_inputs, parse_check, sd  # noqa: E402

BASE_ACC_BAND = (0.562, 0.658)   # 第0段の帯(pc-04 が凍結)
BASE_UNPARSED = 0.016            # 第0段の解釈不能率の上限
ANOMALY_UNPARSED = 0.50          # 安価な異常検知(★ 判定ではない)
COND_A_DIFF = 0.10               # 合格条件 a(★ 本ランでは関門ではない)
COND_B_ACC = 0.30                # 合格条件 b
T_FACTOR_K3 = 1.686              # t(2) 片側95% / sqrt(3)
LAMBDA_EXPECTED = 0.8            # ★ 本ランの段は L1 の1つだけ
LAMBDA_RELATIVE_TOLERANCE = 1e-6 # 停止条件(pc-06 の規則のまま)


def _groups(g: dict) -> dict:
    """プラセボは判定に入らない(cc-01 の規則をそのまま)。"""
    return {k: v for k, v in g.items() if k != "placebo"}


def cmd_base(args) -> int:
    """G0 —— 第0段。素のベースが pc-04〜ll-01 と同じ帯に入っているか。"""
    arms = parse_check(Path(args.check))
    if len(arms) != 1:
        print(f"★ 第0段は素のベース1本で読む(見つかったアーム {len(arms)} 本)。")
        return 1
    arm, g = next(iter(arms.items()))
    non = g.get("non")
    if non is None:
        print("★ 非注入群の行を読めなかった。")
        return 1
    lo, hi = BASE_ACC_BAND
    print(f"  {arm:26s} 正解率 {non['acc']:.4f}(帯 [{lo}, {hi}]) / "
          f"解釈不能 {non['unparsed']:.2%}(上限 {BASE_UNPARSED:.1%})")
    if not (lo <= non["acc"] <= hi):
        print("★ 停止条件 1(G0): ベースの正解率が帯を外れた。**ここで止める。**")
        return 1
    if non["unparsed"] > BASE_UNPARSED:
        print("★ 停止条件 1(G0): ベースの解釈不能率が上限を超えた。**ここで止める。**")
        return 1
    print("  ★ 第0段を通過(ベースは pc-03 以来の帯の中にある)。")
    return 0


def cmd_anomaly(args) -> int:
    """G3 —— 安価な異常検知。★ 判定ではない(cc-01 から引き継いだ関門)。"""
    arms = parse_check(Path(args.check))
    if not arms:
        print("★ 操作チェックの出力を読めなかった。")
        return 1
    bad = []
    for arm, groups in arms.items():
        for name, g in sorted(_groups(groups).items()):
            print(f"  {arm:26s} {name:8s} 正解率 {g['acc']:.4f} / "
                  f"解釈不能 {g['unparsed']:.2%}")
            if g["unparsed"] > ANOMALY_UNPARSED:
                bad.append(f"{arm} の{name} 解釈不能率 {g['unparsed']:.2%}")
    if bad:
        print(f"★ 安価な異常検知に該当({ANOMALY_UNPARSED:.0%} 超): " + " / ".join(bad))
        return 1
    print(f"  安価な異常検知({ANOMALY_UNPARSED:.0%})には該当しない。")
    return 0


def cmd_lambda(args) -> int:
    """G2 —— 実効 λ が凍結値から相対 1e-6 を超えて離れていないか。"""
    for arm in args.arms:
        p = Path("models") / arm / "scale.json"
        if not p.is_file():
            print(f"★ {p} が無い。")
            return 1
        d = json.loads(p.read_text(encoding="utf-8"))
        eff = float(d["effective_lambda"])
        rel = abs(eff - LAMBDA_EXPECTED) / LAMBDA_EXPECTED
        print(f"  {arm:26s} λ={d['lambda']} 実効 {eff!r}(相対 {rel:.3e})")
        if d["lambda"] != LAMBDA_EXPECTED:
            print(f"★ 停止条件 7(G2): λ が {LAMBDA_EXPECTED} でない。")
            return 1
        if rel > LAMBDA_RELATIVE_TOLERANCE:
            print(f"★ 停止条件(G2): 実効 λ の相対誤差が {LAMBDA_RELATIVE_TOLERANCE} を超えた。")
            return 1
    print("  実効 λ は3本とも凍結値と一致(相対 1e-6 以内)。")
    return 0


def _read_k3(check: Path, key: str, pick) -> list | None:
    """複製3本ぶんの値を取り出す。**3本そろっていなければ読まない。**"""
    arms = parse_check(check)
    if len(arms) != 3:
        print(f"★ 判定は複製3本の分布で行う規則である(見つかったアーム {len(arms)} 本)。")
        return None
    values = []
    for arm, g in sorted(arms.items()):
        v = pick(_groups(g))
        if v is None:
            print(f"★ {arm} から {key} を読めなかった。")
            return None
        print(f"  {arm:26s} {key} {v:.4f}")
        values.append(v)
    return values


def cmd_gate_b(args) -> int:
    """G4 —— 条件 b(非注入群 正解率 >= 0.30)を k=3 の分布で読む。"""
    values = _read_k3(Path(args.check), "非注入群 正解率",
                      lambda g: g["non"]["acc"] if "non" in g else None)
    if values is None:
        return 1
    ok_binary = sum(1 for x in values if x >= COND_B_ACC)
    mean = sum(values) / len(values)
    margin = T_FACTOR_K3 * sd(values)
    bound = mean - margin                       # ★ 下側(b は「以上」が合格)
    print(f"  読み1(二値): {ok_binary}/3 が b(>= {COND_B_ACC})を満たす")
    print(f"  読み2(量)  : 平均 {mean:.4f} / 片側95%下限 {bound:.4f}"
          f"(余裕 {T_FACTOR_K3} × SD = {margin:.4f})")
    passed_1, passed_2 = ok_binary == 3, bound >= COND_B_ACC
    if passed_1 and passed_2:
        print("  ★ 関門 G4 を通過(素の能力は保たれている)。")
        return 0
    if passed_1 != passed_2:
        print("  ★ 読み1と読み2が食い違う → **不明**。★ 合格ではないので関門は通さない。")
    else:
        print("  ★ 関門 G4 に落ちた(fine-tune がベースを壊している)。")
    print("  ★ 壊れたモデルに検出器を当てても、測れるのは配管だけである。**ここで止める。**")
    return 1


def cmd_report_a(args) -> int:
    """★ 条件 a(差 >= 10pt)を k=3 で読む。**報告するだけ。常に成功で返る。**

    ⛔ preregister「★ a を関門にしない」—— 本ランの目的は検出器を1回撃つことであり、
      陽性対照の認定ではない(それは ll-01 で済んでいる)。
    ★ ただし a が頑健に合格しなかった場合、判定 B の結果を
      「検出器が汚染を検出した」とは書かない(測る前に凍結した縛り)。
    """
    arms = parse_check(Path(args.check))
    if len(arms) != 3:
        print(f"★ a は複製3本で読む(見つかったアーム {len(arms)} 本)。★ 報告のみ。")
        return 0
    diffs = []
    for arm, g in sorted(arms.items()):
        gg = _groups(g)
        if "inj" not in gg or "non" not in gg:
            print(f"  {arm:26s} 注入群/非注入群がそろわない。★ 報告のみ。")
            return 0
        diff = gg["inj"]["acc"] - gg["non"]["acc"]
        diffs.append(diff)
        print(f"  {arm:26s} 差 {diff*100:+7.2f}pt "
              f"(注入群 {gg['inj']['acc']:.4f} / 非注入群 {gg['non']['acc']:.4f})")
    ok_binary = sum(1 for x in diffs if x >= COND_A_DIFF)
    mean = sum(diffs) / len(diffs)
    margin = T_FACTOR_K3 * sd(diffs)
    bound = mean - margin
    print(f"  読み1(二値): {ok_binary}/3 が a(>= {COND_A_DIFF*100:.0f}pt)を満たす")
    print(f"  読み2(量)  : 平均 {mean*100:+.2f}pt / 片側95%下限 {bound*100:+.2f}pt"
          f"(余裕 {T_FACTOR_K3} × SD = {margin*100:.2f}pt)")
    passed_1, passed_2 = ok_binary == 3, bound >= COND_A_DIFF
    if passed_1 and passed_2:
        print("  ★ a は頑健に合格 → 判定 B を「検出器が汚染を検出した」と読んでよい。")
    else:
        state = "不明" if passed_1 != passed_2 else "不合格"
        print(f"  ★ a は**{state}** → ★ 判定 B を「検出器が汚染を検出した」とは書かない。")
        print("     書けるのは「注入の効きが確認できなかったモデルで drop がこうだった」まで。")
    print("  ★ a は関門ではない。**測定は止めない**(preregister で測る前に凍結した)。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("base", cmd_base), ("anomaly", cmd_anomaly),
                     ("gate-b", cmd_gate_b), ("gate-c", cmd_gate_c),
                     ("report-a", cmd_report_a)):
        p = sub.add_parser(name)
        p.add_argument("--check", required=True)
        p.set_defaults(fn=fn)
    for name, fn in (("inputs", cmd_inputs), ("lambda", cmd_lambda)):
        p = sub.add_parser(name)
        p.add_argument("--arms", nargs="+", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
