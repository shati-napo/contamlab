#!/usr/bin/env bash
# scripts/50-check-determinism.sh — GPU での決定性を**実測**する。
#
#   bash scripts/50-check-determinism.sh        (先に 40-pilot.sh 1 を済ませておく)
#
# ★ なぜ必要か ★
# パイロット①(2026-08-03)の「不一致ペア 0 件 = 完全決定的」は **CPU での測定値**である。
# `temperature 0` は「最も確率の高い選択肢を選ぶ」であって「計算結果が同じになる」では
# ない。確率の計算は大量の浮動小数の加算であり、**加算順序が変われば最下位桁が変わる。**
# CPU と GPU では並列化が違うので順序が違い、1位と2位が僅差の問題で順位が入れ替わりうる。
# そしてそれが起きるのは**モデルが迷っている問題**であり、`shuffle_choices` で答えが
# 変わるのも**モデルが迷っている問題**である。ノイズが乗る集合と測りたい集合が重なる。
#
# ★ なぜ「キャッシュの conflicts を見る」では測れないか ★
# `CachedModel.answer` はキャッシュに当たればモデルを呼ばずに帰る(runner.py:223-229)。
# 同じキャッシュで2回目を回すと**モデルは一度も呼ばれず、古い答えがそのまま返る。**
# `put()` に到達しないので `conflicts` も立たない。つまり同一キャッシュでの再実行は
# 決定性を測っていない。**別のキャッシュファイルに独立に取り直して突き合わせる**しかない。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format
CACHE_A="$(cache_path)"
CACHE_B="$(det_cache_path)"
PROMPT_FORMAT="$(prompt_format)"
RESULT_A="reports/pilot1.$(env_tag).json"
RESULT_B="reports/determinism.$(env_tag).json"

[[ -f "$CACHE_A"   ]] || { echo "本番キャッシュが無い。先に 40-pilot.sh 1。" >&2; exit 1; }
[[ -f "$RESULT_A"  ]] || { echo "$RESULT_A が無い。先に 40-pilot.sh 1。" >&2; exit 1; }

# パイロット①と**完全に同じ設計**で取り直す。違うのはキャッシュの行き先だけ。
SAMPLE_N=70
PERTURBATOR=identity
TARGET_EFFECT=0.20
EXPECTED_PSI=0.405

banner "同じ 70 問を、独立したキャッシュにもう一度取る"
echo "  A(本番) : $CACHE_A"
echo "  B(捨て) : $CACHE_B   ← 本番と混ぜない。混ぜると再生が起きて測定にならない"
if [[ -f "$CACHE_B" ]]; then
  echo
  echo "★ B が既にある($(wc -l < "$CACHE_B") 行)。このまま走らせると B 側も再生になる。" >&2
  echo "  測定をやり直すなら B を別名に退避してから再実行すること(消さない)。" >&2
  exit 1
fi

$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
  --perturbator "$PERTURBATOR" --prompt-format "$PROMPT_FORMAT" \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --yes --cache "$CACHE_B" --json "$RESULT_B" $(model_flags)

banner "突き合わせ"
$PY - "$CACHE_A" "$CACHE_B" "$RESULT_A" "$RESULT_B" <<'PYEOF'
import json, re, sys
from collections import defaultdict

cache_a, cache_b, result_a, result_b = sys.argv[1:5]
WS = re.compile(r"\s+")


def load(path):
    entries, models = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entries.setdefault(rec["key"], rec["response"])
            models.setdefault(rec["key"], rec["model"])
    return entries, models


a, models_a = load(cache_a)
b, _ = load(cache_b)
shared = sorted(set(a) & set(b))

print(f"  A {len(a)} 件 / B {len(b)} 件 / 共通 {len(shared)} 件")
if not shared:
    print("  ★ 共通のキーが無い。設計が食い違っている。", file=sys.stderr)
    raise SystemExit(1)

exact = defaultdict(int)
normalized = defaultdict(int)
total = defaultdict(int)
examples = []
for key in shared:
    model = models_a[key]
    total[model] += 1
    if a[key] != b[key]:
        exact[model] += 1
        if WS.sub(" ", a[key].strip()) != WS.sub(" ", b[key].strip()):
            normalized[model] += 1
            if len(examples) < 3:
                examples.append((model, a[key][:80], b[key][:80]))

print()
print(f"  {'モデル':<16} {'共通':>6} {'生の文字列が不一致':>20} {'正規化後も不一致':>18}")
for model in sorted(total):
    print(f"  {model:<16} {total[model]:>6} {exact[model]:>20} {normalized[model]:>18}")

for model, sa, sb in examples:
    print(f"\n  例({model}):\n    A: {sa!r}\n    B: {sb!r}")

# ★ 生の文字列の一致は採点の一致より厳しい条件である。文字列が違っても
#   同じ選択肢に読める場合があるので、**採点結果そのもの**も突き合わせる。
#   こちらが本命(runner.py の採点をそのまま使うので、採点規則を再実装しない)。
def tables(path):
    data = json.load(open(path, encoding="utf-8"))
    return {m["name"]: (m["table"], m["accuracy_original"], m["accuracy_perturbed"],
                        m["unparsed_original"], m["unparsed_perturbed"])
            for m in data["models"]}


ta, tb = tables(result_a), tables(result_b)
print()
same_scoring = True
for name in sorted(set(ta) | set(tb)):
    ok = ta.get(name) == tb.get(name)
    same_scoring &= ok
    print(f"  採点結果 {name:<16} {'一致' if ok else '★不一致'}")
    if not ok:
        print(f"      A: {ta.get(name)}")
        print(f"      B: {tb.get(name)}")

print()
strict_ok = sum(exact.values()) == 0
if strict_ok and same_scoring:
    print("  ★ 完全決定的。GPU でも temperature 0 で同じ答えが返る。")
    print("     → ψ は非決定性で膨らまない。パイロット②の値をそのまま使ってよい。")
elif same_scoring:
    print("  ▲ 生の文字列は揺れたが、**採点結果は一致**した。")
    print("     → 測っているもの(正解率と分割表)は再現している。揺れの件数を")
    print("       preregister に記録し、『採点は決定的、生成は非決定的』と書くこと。")
else:
    print("  ★ 採点結果が一致しない。**モデルが非決定的である。**")
    print("     preregister の停止条件に該当する。ψ が非決定性で膨らむので、")
    print("     この状態の数字は使わない。原因(バックエンド / 並列度 / 量子化)を潰すこと。")
    raise SystemExit(1)
PYEOF

banner "記録"
echo "  reports/determinism.$(env_tag).json"
echo "  ★ この結果を preregister.md に書く。CPU 時代の『不一致 0 件』は"
echo "    別環境の測定値なので、GPU の値で置き換えるのではなく**併記**すること。"
