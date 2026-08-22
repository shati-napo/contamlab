#!/usr/bin/env bash
# scripts/81-pf1-pilot.sh — ラン perturbation-floor-01 の段 3(パイロット)。
#
#   bash scripts/81-pf1-pilot.sh
#
# preregister「ラン: perturbation-floor-01」→「段と関門」の **G2** の実装。
#
# ★ **何を測るのか** —— 各モデルが書式 C で「記号だけ答える」指示に従えるか。
#   **解釈不能率が 5% を超えたモデルは本番から外す。**
#   前例: jmmlu-shuffle-02 で `llmjp3-13b` が書式 A のとき 37.1% で脱落した。
#
# ★ **なぜパイロットを置くのか —— 費用である。**
#   崩壊したモデルは推論が **12.6 倍遅い**(cc-01 実測 1.29 秒/コール)。
#   n=4,742 でそれを踏むと 1 本あたり **+3.4h ≒ +$6.8**。
#   **150 問なら 1 本 20 秒で分かる。**
#
# ⛔ **contamlab run は使わない。**n=150 では検出力ゲートが UnderpoweredError で
#   止める(README が謳う正しい挙動)。ここで測りたいのは解釈不能率だけなので、
#   runner の内部 API を直接呼ぶ(65-manipulation-check.sh と同じ作法)。
#   ★ **測定条件(書式・temperature・max_tokens)は本番と同一である。**
#
# ⛔ **原文条件のみ。**摂動は当てない —— 生存の判定に摂動後の値は要らないし、
#   当てれば本番と同じ問題を先に消費することになる。
#
# 出力: $PF1_SURVIVORS_FILE(生存モデル名。本番 82 が機械的に読む)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format

TAG="$(env_tag)"
PROMPT_FORMAT="$(prompt_format)"
CACHE="$(pf1_pilot_cache_path)"
OUT="reports/pf1-pilot.$TAG.json"
mkdir -p reports data/cache

[[ -f "$OUT" ]] && { echo "★ 既に $OUT がある。上書きしない。" >&2; exit 1; }

banner "パイロット(preregister 段 3 / 関門 G2)"
cat <<EOF
  問題数        : $PF1_PILOT_N(DEV から決定論的に抽出・原文条件のみ)
  出力書式      : $PROMPT_FORMAT
  線            : 解釈不能率 > $(awk "BEGIN{printf \"%.0f\", $PF1_MAX_UNPARSABLE*100}")% のモデルは**本番から外す**
  停止条件 2    : 生存が $PF1_MIN_SURVIVORS 本未満なら止める
  キャッシュ    : $CACHE(★ 本番と別ファイル)
  出力          : $OUT / $PF1_SURVIVORS_FILE

  ⛔ 読むのは解釈不能率と素の正解率だけ。drop / p_value は**測らない**。
EOF

CONTAMLAB_FORMAT="$PROMPT_FORMAT" \
CONTAMLAB_PILOT_N="$PF1_PILOT_N" \
CONTAMLAB_BENCHMARK="$BENCHMARK" \
CONTAMLAB_CACHE="$CACHE" \
CONTAMLAB_BASE_URL="$OLLAMA_BASE_URL" \
CONTAMLAB_MAX_UNPARSABLE="$PF1_MAX_UNPARSABLE" \
CONTAMLAB_MIN_SURVIVORS="$PF1_MIN_SURVIVORS" \
CONTAMLAB_SURVIVORS_FILE="$PF1_SURVIVORS_FILE" \
CONTAMLAB_OUT="$OUT" \
CONTAMLAB_MODELS="$(for e in "${PF1_ROSTER[@]}"; do echo -n "${e%%|*} "; done)" \
$PY - <<'PYEOF'
import json, os
from pathlib import Path

from contamlab.benchmark import load_jsonl, split_dev_holdout, take_deterministic
from contamlab.clients import CallBudget, ClientOptions, build_api_model
from contamlab.runner import CachedModel, ResponseCache, format_prompt, grade, set_prompt_format

set_prompt_format(os.environ["CONTAMLAB_FORMAT"])
pilot_n = int(os.environ["CONTAMLAB_PILOT_N"])
threshold = float(os.environ["CONTAMLAB_MAX_UNPARSABLE"])
min_survivors = int(os.environ["CONTAMLAB_MIN_SURVIVORS"])
names = os.environ["CONTAMLAB_MODELS"].split()

dev, _ = split_dev_holdout(load_jsonl(Path(os.environ["CONTAMLAB_BENCHMARK"])))
# ★ 決定論的に取る。乱数を使わないので、誰が走らせても同じ 150 問になる。
items = take_deterministic(dev, min(pilot_n, len(dev)))
cache = ResponseCache(Path(os.environ["CONTAMLAB_CACHE"]))

rows, survivors = [], []
print()
for name in names:
    budget = CallBudget(max_calls=len(items) * 2 + 100)
    spec = f"compat:{name}:{name}:{os.environ['CONTAMLAB_BASE_URL']}"
    model = CachedModel(build_api_model(spec, ClientOptions(
        budget=budget, temperature=0.0, max_tokens=256)), cache)

    ok = parsed = 0
    for item in items:
        r = grade(item, model.answer(format_prompt(item)))
        ok += r.correct
        parsed += r.parsed
    n = len(items)
    acc = ok / n
    unparsable = 1 - parsed / n
    alive = unparsable <= threshold

    mark = "✅ 生存" if alive else "❌ 脱落"
    print(f"  {name:<22} 正解率 {acc:.4f}  解釈不能 {unparsable:6.2%}  {mark}")
    rows.append({"name": name, "n": n, "accuracy_original": acc,
                 "unparsable_rate": unparsable, "survived": alive})
    if alive:
        survivors.append(name)

Path(os.environ["CONTAMLAB_OUT"]).write_text(json.dumps({
    "run": "perturbation-floor-01", "stage": "pilot",
    "pilot_n": pilot_n, "threshold_unparsable": threshold,
    "prompt_format": os.environ["CONTAMLAB_FORMAT"],
    "models": rows, "survivors": survivors,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print()
print(f"  生存 {len(survivors)} / {len(names)} 本: {' '.join(survivors) or '(なし)'}")

# ⛔ 停止条件 2 —— 生存が 3 本未満なら止める。
#   ★ 「床」は分布として読むものなので、2 本では帯を描けない。
if len(survivors) < min_survivors:
    print()
    print(f"  ★ 停止条件 2 に該当 —— 生存 {len(survivors)} 本 < {min_survivors} 本。")
    print("  ⛔ 線を下げて測り直さない。**別のランとして事前登録する。**")
    raise SystemExit(2)

Path(os.environ["CONTAMLAB_SURVIVORS_FILE"]).write_text(
    "\n".join(survivors) + "\n", encoding="utf-8")
print(f"  ★ 生存モデルを書いた: {os.environ['CONTAMLAB_SURVIVORS_FILE']}")
PYEOF

echo
echo "★ 記録: $OUT"
echo "★ 次は段 4(本番): bash scripts/82-pf1-production.sh"
