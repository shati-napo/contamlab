#!/usr/bin/env python3
"""scripts/cc01_gate.py — ラン calibration-curve-01 の関門を**機械に守らせる**。

    python3 scripts/cc01_gate.py anomaly --check reports/cc01-check-x00-t1.txt
    python3 scripts/cc01_gate.py stop6   --check reports/cc01-check-x40.txt
    python3 scripts/cc01_gate.py gate-c  --check reports/cc01-check-x00.txt
    python3 scripts/cc01_gate.py inputs  --arms cc1t1-x00 cc1t2-x00 cc1t3-x00
    python3 scripts/cc01_gate.py detect  --tag <env-tag> --rate 40

★ ここに書いてある数字は**1つも新しくない。**すべて preregister
  「## ラン: calibration-curve-01」が測る前に凍結した値である。

    50%   —— 「x00 の1本目で解釈不能率が 50% を超えたら即座に止める」(安価な異常検知)
    5%    —— 合格条件 c(解釈不能率 両群 <= 5%)。pc-02 から1文字も変えていない
    1.686 —— k=3 の片側95%(t(2) = 2.920 -> 2.920 / sqrt(3) = 1.686)
    3/3   —— detected の読み(3/3 検出 / 0/3 不検出 / 1〜2/3 不明)

★ この道具は判定を**足さない。**規則の側を動かせないように、閾値はここに
  定数として書き、コマンドラインからは受け取らない。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANOMALY_UNPARSED = 0.50      # 安価な異常検知(★ 判定ではない)
COND_C_UNPARSED = 0.05       # 合格条件 c
T_FACTOR_K3 = 1.686          # t(2) 片側95% / sqrt(3)

ARM_RE = re.compile(r"^\s+(\S+)\s+\(注入率")
NON_RE = re.compile(r"非注入群 n=\s*(\d+)\s+正解率 ([0-9.]+)\s+解釈不能 ([0-9.]+)%")
INJ_RE = re.compile(r"注入群\s+n=\s*(\d+)\s+正解率 ([0-9.]+)\s+解釈不能 ([0-9.]+)%")
PLA_RE = re.compile(r"プラセボ n=\s*(\d+)\s+正解率 ([0-9.]+)\s+解釈不能 ([0-9.]+)%")


def parse_check(path: Path) -> dict:
    """65-manipulation-check.sh の出力を読む。★ 数字を作らない。読むだけ。"""
    arms: dict = {}
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ARM_RE.match(line)
        if m:
            cur = m.group(1)
            arms[cur] = {}
            continue
        if cur is None:
            continue
        for key, rx in (("placebo", PLA_RE), ("non", NON_RE), ("inj", INJ_RE)):
            m = rx.search(line)
            if m:
                arms[cur][key] = {"n": int(m.group(1)),
                                  "acc": float(m.group(2)),
                                  "unparsed": float(m.group(3)) / 100.0}
                break
    return arms


def sd(xs: list) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def cmd_anomaly(args) -> int:
    arms = parse_check(Path(args.check))
    if not arms:
        print("★ 操作チェックの出力を読めなかった。")
        return 1
    bad = []
    for arm, groups in arms.items():
        for name, g in sorted(groups.items()):
            print(f"  {arm:20s} {name:8s} 正解率 {g['acc']:.4f} / 解釈不能 {g['unparsed']:.2%}")
            if name != "placebo" and g["unparsed"] > ANOMALY_UNPARSED:
                bad.append(f"{arm} の{name} 解釈不能率 {g['unparsed']:.2%}")
    if bad:
        print(f"★ 安価な異常検知に該当({ANOMALY_UNPARSED:.0%} 超): " + " / ".join(bad))
        print("  ★ これは判定ではない。5% の線には触れていない。")
        return 1
    print(f"  ★ 明らかな崩壊は無い(どの群も {ANOMALY_UNPARSED:.0%} 以下)。"
          "★ 合否はまだ判定していない。")
    return 0


def cmd_stop6(args) -> int:
    """停止条件 6: 注入群のほうが正解率が低いアームが出た。"""
    arms = parse_check(Path(args.check))
    if not arms:
        print("★ 操作チェックの出力を読めなかった。")
        return 1
    bad = []
    for arm, g in sorted(arms.items()):
        if "inj" in g and "non" in g:
            diff = g["inj"]["acc"] - g["non"]["acc"]
            print(f"  {arm:20s} 差 {diff*100:+7.2f}pt "
                  f"(注入群 {g['inj']['acc']:.4f} / 非注入群 {g['non']['acc']:.4f})")
            if diff < 0:
                bad.append(f"{arm}: {diff*100:+.2f}pt")
    if bad:
        print("★ 停止条件 6 に該当(注入群のほうが低い): " + " / ".join(bad))
        return 1
    print("  停止条件 6 は該当しない。")
    return 0


def cmd_gate_c(args) -> int:
    """x00 の関門 —— 条件 c(解釈不能率 両群 <= 5%)を k=3 の分布で読む。"""
    arms = parse_check(Path(args.check))
    if len(arms) != 3:
        print(f"★ 関門は複製3本で読む規則である(見つかったアーム {len(arms)} 本)。")
        return 1
    per_arm_max = []
    for arm, g in sorted(arms.items()):
        groups = {k: v for k, v in g.items() if k != "placebo"}   # ★ プラセボは判定に入らない
        worst = max(v["unparsed"] for v in groups.values())
        per_arm_max.append(worst)
        print(f"  {arm:20s} 解釈不能率(両群の最大) {worst:.2%}")
    ok_binary = sum(1 for x in per_arm_max if x <= COND_C_UNPARSED)
    mean = sum(per_arm_max) / len(per_arm_max)
    margin = T_FACTOR_K3 * sd(per_arm_max)
    bound = mean + margin                       # ★ 上側(c は「以下」が合格)
    print(f"  読み1(二値): {ok_binary}/3 が c(<= {COND_C_UNPARSED:.0%})を満たす")
    print(f"  読み2(量)  : 平均 {mean:.2%} / 片側95%上限 {bound:.2%}"
          f"(余裕 {T_FACTOR_K3} × SD = {margin:.2%})")
    passed_1 = ok_binary == 3
    passed_2 = bound <= COND_C_UNPARSED
    if passed_1 and passed_2:
        print("  ★ 関門を通過(読み1・読み2 とも c を満たす)。")
        return 0
    if passed_1 != passed_2:
        print("  ★ 読み1と読み2が食い違う → **不明**。★ 合格ではないので関門は通さない。")
    else:
        print("  ★ 関門に落ちた(停止条件 5)。")
    print("  ★ 埋め草の量を減らして測り直さない。設計が成立しないという結果として報告する。")
    return 1


def cmd_inputs(args) -> int:
    """停止条件 2: 同一アームの3本で入力側が一致するか。"""
    keys = ("target_total_tokens_T", "seed", "n_injected_items", "n_blocks", "steps")
    rows = {}
    for arm in args.arms:
        p = Path("models") / arm / "train.json"
        if not p.is_file():
            print(f"★ {p} が無い。")
            return 1
        d = json.loads(p.read_text(encoding="utf-8"))
        rows[arm] = {k: d.get(k) for k in keys}
        print(f"  {arm:16s} " + " / ".join(f"{k}={d.get(k)}" for k in keys)
              + f" / train_loss={d.get('train_loss')}")
    first = next(iter(rows.values()))
    for arm, r in rows.items():
        if r != first:
            print(f"★ 停止条件 2 に該当: {arm} の入力側が他と一致しない。")
            return 1
    print("  入力側は3本とも一致(★ train_loss の不一致は停止条件ではない —— td-01)。")
    return 0


def cmd_detect(args) -> int:
    """detected を複製3本の分布で読む。3/3 検出 / 0/3 不検出 / 1〜2/3 不明。"""
    found = []
    for k in (1, 2, 3):
        arm = f"cc1L08t{k}-x{args.rate}"
        p = Path("reports") / f"calibration-curve-01.{arm}.{args.tag}.json"
        if not p.is_file():
            print(f"★ {p} が無い。")
            return 1
        d = json.loads(p.read_text(encoding="utf-8"))
        m = next(m for m in d["models"] if m["name"] == arm)
        found.append(m)
        print(f"  {arm:20s} drop {m['drop']*100:+7.2f}pt / 割引後下限 {m['adjusted_lcb']:+.4f} "
              f"/ p {m['p_holm']:.4g} / {'★ 検出' if m['detected'] else '—'}")
    n_det = sum(1 for m in found if m["detected"])
    drops = [m["drop"] for m in found]
    print(f"  平均 drop {sum(drops)/3*100:+.2f}pt / SD {sd(drops)*100:.2f}pt")
    if n_det == 3:
        print(f"  ★ 3/3 → **検出**(注入率 {int(args.rate)}%)。次の段へ下ろす。")
        return 0
    if n_det == 0:
        print(f"  ★ 0/3 → **不検出**(注入率 {int(args.rate)}%)。★ ここで打ち切る。")
        print("  ★ これより下の水準は「未実行」であって「不検出」ではない。")
        return 2
    print(f"  ★ {n_det}/3 → **不明**(検出でも不検出でもない)。"
          "打ち切り規則は 3/3 不検出でしか発動しないので、次の段へ下ろす。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("anomaly", cmd_anomaly), ("stop6", cmd_stop6), ("gate-c", cmd_gate_c)):
        p = sub.add_parser(name)
        p.add_argument("--check", required=True)
        p.set_defaults(fn=fn)
    p = sub.add_parser("inputs")
    p.add_argument("--arms", nargs="+", required=True)
    p.set_defaults(fn=cmd_inputs)
    p = sub.add_parser("detect")
    p.add_argument("--tag", required=True)
    p.add_argument("--rate", required=True, choices=("00", "02", "05", "10", "20", "40"))
    p.set_defaults(fn=cmd_detect)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
