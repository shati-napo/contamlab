#!/usr/bin/env bash
# scripts/65-manipulation-check.sh — ★ 注入が本当に入ったかを、検定とは独立に確かめる。
#
#   bash scripts/65-manipulation-check.sh pc-x40      ← まずこれ1本(最も強い注入)
#   bash scripts/65-manipulation-check.sh             ← 全6アーム
#
# preregister「ラン: positive-control-01」の「★ 操作チェック」節が正。
#
# ★ これが無いと、陰性の結果が「装置が鈍い」のか「注入が入っていない」のか
#   **永久に区別できなくなる。** 測定(56,904 コール)の前に必ず通すこと。
#
# ★ 読むのは**原文条件の正解率だけ**である。drop / p_value / adjusted_lcb は読まない。
#   採用基準の判定に先立って結果を覗くことになるため(preregister の明文の禁止)。
#
# 課金の無駄は出ない —— ここで撃つ原文条件のプロンプトは測定で使うものと同一で、
# キャッシュのキー(モデル名 + プロンプト)が一致するので 70 で再利用される。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format

PER_GROUP="${CONTAMLAB_PER_GROUP:-400}"   # 0 で全件。既定は1アーム約800コール(≒5分)
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || ARMS=(pc-x00 pc-x02 pc-x05 pc-x10 pc-x20 pc-x40)

banner "操作チェック(原文条件のみ・書式 $(prompt_format))"
echo "  1群あたりの問題数: ${PER_GROUP}(0 = 全件)"
echo "  キャッシュ        : $(cache_path)"

CONTAMLAB_ARMS="${ARMS[*]}" \
CONTAMLAB_PER_GROUP="$PER_GROUP" \
CONTAMLAB_CACHE="$(cache_path)" \
CONTAMLAB_FORMAT="$(prompt_format)" \
CONTAMLAB_BENCHMARK="$BENCHMARK" \
CONTAMLAB_BASE_URL="$OLLAMA_BASE_URL" \
$PY - <<'PYEOF'
import os, sys
from pathlib import Path

from contamlab.benchmark import load_jsonl, split_dev_holdout, take_deterministic
from contamlab.clients import CallBudget, ClientOptions
from contamlab.runner import CachedModel, ResponseCache, format_prompt, grade, set_prompt_format
from contamlab.clients import build_api_model

set_prompt_format(os.environ["CONTAMLAB_FORMAT"])
per_group = int(os.environ["CONTAMLAB_PER_GROUP"])
arms = os.environ["CONTAMLAB_ARMS"].split()

dev, _ = split_dev_holdout(load_jsonl(Path(os.environ["CONTAMLAB_BENCHMARK"])))
by_id = {i.id: i for i in dev}
cache = ResponseCache(Path(os.environ["CONTAMLAB_CACHE"]))

def accuracy(model, items):
    ok = parsed = 0
    for item in items:
        r = grade(item, model.answer(format_prompt(item)))
        ok += r.correct
        parsed += r.parsed
    n = len(items)
    return (ok / n if n else 0.0), (1 - parsed / n if n else 0.0), n

failures = []
saw_injection = False   # ★ 表示の分岐にだけ使う。判定には一切入らない
print()
for arm in arms:
    ids_path = Path("data/injection") / f"{arm}.ids"
    injected_ids = set(ids_path.read_text(encoding="utf-8").split()) if ids_path.exists() else set()
    rate = int(arm[-2:]) / 100.0

    injected = [by_id[i] for i in sorted(injected_ids) if i in by_id]
    other = [i for i in dev if i.id not in injected_ids]
    if per_group:
        injected = take_deterministic(injected, min(per_group, len(injected)))
        other = take_deterministic(other, min(per_group, len(other)))

    budget = CallBudget(max_calls=(len(injected) + len(other)) * 2 + 100)
    spec = f"compat:{arm}:{arm}:{os.environ['CONTAMLAB_BASE_URL']}"
    model = CachedModel(build_api_model(spec, ClientOptions(
        budget=budget, temperature=0.0, max_tokens=256)), cache)

    acc_o, unp_o, n_o = accuracy(model, other)
    print(f"  {arm}  (注入率 {rate:.0%})")
    print(f"      非注入群 n={n_o:4d}  正解率 {acc_o:.4f}  解釈不能 {unp_o:.2%}")

    if not injected:
        # pc-x00 / pcbase-x00 には注入群が無い。ここで見るのは
        # 「fine-tune がベースを壊していないか」(pc-01)、
        # 「そもそもこのベースが書式 C で使えるのか」(pc-02 の第0段)。
        print(f"      注入群   —— 無し(陰性対照)")
        if acc_o < 0.30:
            failures.append(f"{arm}: 素の正解率 {acc_o:.4f} < 0.30。fine-tune がベースを壊した。")
        # ★ pc-02 の第0段。閾値 5% は pc-01「解釈不能率が 5% 超のアームは外す」の移植で、
        #   preregister pc-02「合格条件 c」として測る前に凍結されている。
        if unp_o > 0.05:
            failures.append(
                f"{arm}: 解釈不能率 {unp_o:.2%} > 5%。**レシピの問題ではない。**"
                " このベースと書式の組み合わせでは、注入の有無に関わらず全アームが脱落する。")
        continue

    saw_injection = True
    acc_i, unp_i, n_i = accuracy(model, injected)
    diff = acc_i - acc_o
    print(f"      注入群   n={n_i:4d}  正解率 {acc_i:.4f}  解釈不能 {unp_i:.2%}")
    print(f"      差       {diff:+.4f}  ({diff*100:+.2f} pt)")

    # ★ preregister の停止条件をそのまま実装する。閾値は測る前に決まっている。
    # 合格条件 a(差 ≥ 10pt)—— pc-01 から変えていない。
    if diff < 0:
        failures.append(f"{arm}: 注入群のほうが低い({diff:+.4f})。レシピがモデルを壊している。")
    elif rate >= 0.20 and diff < 0.10:
        failures.append(
            f"{arm}: 注入率 {rate:.0%} なのに差が {diff*100:.2f}pt < 10pt。**注入が入っていない。**")

    # ★ 合格条件 b・c —— preregister pc-02「合格条件(各段で同一・測る前に固定)」の実装。
    #   閾値そのものは新しくない(b は pc-01「pc-x00 が 0.30 未満なら停止」、
    #   c は pc-01「解釈不能率が 5% 超のアームは外す」)。**機械判定が無かっただけである。**
    #
    #   ★ a だけで選ぶと「注入は入るがモデルが壊れているレシピ」が勝ちうる。
    #     そのレシピで較正を回すと全アームが c で脱落し、アームが 1 本以下になって
    #     Cochran の Q が計算できない。b・c は本番での実行可能性そのものである。
    if acc_o < 0.30:
        failures.append(
            f"{arm}: 非注入群の正解率 {acc_o:.4f} < 0.30(合格条件 b)。"
            " 注入が入っていてもこのレシピは本番で使えない。")
    if unp_o > 0.05 or unp_i > 0.05:
        failures.append(
            f"{arm}: 解釈不能率 非注入 {unp_o:.2%} / 注入 {unp_i:.2%} のどちらかが 5% 超"
            "(合格条件 c)。pc-01 ならこのアームは脱落、pc-02 ならこの段は不合格。")

print()
if failures:
    print("★ 停止条件に該当した。測定へ進んではいけない。", file=sys.stderr)
    for f in failures:
        print(f"    {f}", file=sys.stderr)
    print("""
  ★ この結果を「装置が鈍い」と読んではいけない。読めるのは「注入が入らなかった」だけである。

  ラン positive-control-01 のアーム(pc-*)の場合:
    レシピ(E・学習率・LoRA rank)を見直して**別のランとして**やり直す。
    ★ 同じランの中で E を変えて撃ち直してはいけない。結果を見てから規則を選ぶことになる。
    → それが positive-control-02 であり、事前登録済みである。

  ラン positive-control-02 のアーム(pcbase-x00 / pcr*-x40)の場合:
    pcbase-x00(第0段)が落ちた → ★停止。レシピの問題ではない。E を触っても解決しない。
                                 ベースの選び直しを positive-control-03 として別に事前登録する。
    pcr*-x40(梯子)が落ちた   → その段は不合格。**事前登録した順序どおり次の段へ進む。**
                                 R4 まで落ちたら「全滅」を結果として報告し、格子は増やさない。

  ラン positive-control-03 のアーム(pcbase-<名前>-x00)の場合:
    候補が関門を落ちた → 事前登録した順序どおり次の候補へ。2本とも落ちたら「全滅」を
                         結果として報告する。**書式も採点規則も 5% の閾値も緩めない。**

  ラン positive-control-04 のアーム(pc4r*-x40)の場合:
    pcbase-*-x00(第0段)が落ちた → ★停止。レシピの問題ではない。
    pc4r*-x40(梯子)が落ちた     → その段は不合格。次の段へ。R4 まで落ちたら「全滅」。
""", file=sys.stderr)
    raise SystemExit(1)

# ★ 文言だけの分岐(2026-08-09・ラン pc-03 の実行時に見つけた不備の修正)。
#   注入率 0% のアームだけを測ったとき、「注入は入っている」は事実に反する
#   —— そのアームには注入群が無く、差も測っていない。**判定は上で確定済みで、
#   ここは表示だけである。閾値も分岐も変えていない。**
if saw_injection:
    print("操作チェック通過。注入は入っている。測定へ進んでよい。")
else:
    print("関門を通過。**注入は測っていない**(注入率 0% のアームのみ)。"
          "確かめたのは「このベースが書式 C でこの器を通せる」ことだけである。")
PYEOF
