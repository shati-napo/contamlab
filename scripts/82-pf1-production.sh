#!/usr/bin/env bash
# scripts/82-pf1-production.sh — ラン perturbation-floor-01 の段 4(本番測定)。
#
#   bash scripts/82-pf1-production.sh
#
# preregister「ラン: perturbation-floor-01」→「測定条件」「段と関門」段 4 の実装。
#
# ★ **測るのは「摂動そのものが課す drop の床」である。**
#   ロースターは**こちらが1度も学習に使っていないモデル**だけ(パイロット生存分)。
#
# ⛔ **72-detector-firstlight.sh は1行も触っていない。**あちらは自作アームを測る
#   スクリプトで、本ランとはロースターも目的も違う。
#
# ⛔ **--alpha は渡さない。**既定 0.05 のまま。df1 と同じ経路である
#   (2026-08-22 に実測で確認: 72:104 の --alpha は contamlab power にだけ渡っており、
#    contamlab run には渡っていない)。
#
# ★ HOLDOUT は開かない(--split dev)。K も消費しない(shuffle_choices / K=1)。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format

TAG="$(env_tag)"
CACHE="$(cache_path)"
PROMPT_FORMAT="$(prompt_format)"
OUT="reports/perturbation-floor-01.$TAG.json"
TARGET_EFFECT=0.05     # df1 と同一
EXPECTED_PSI=0.4050    # df1 と同一(pc-01 のパイロット②の採用値)
mkdir -p reports data/cache

[[ -f "$OUT" ]] && { echo "★ 既に $OUT がある。上書きしない。" >&2; exit 1; }

# ★ パイロットを通っていないと本番に進めない。飛ばすと、崩壊したモデルを
#   n=4,742 で踏んで 1 本 +$6.8 を払うことになる。
if [[ ! -s "$PF1_SURVIVORS_FILE" ]]; then
  echo "★ $PF1_SURVIVORS_FILE が無い。段 3(81-pf1-pilot.sh)を先に通すこと。" >&2
  exit 1
fi

SURVIVORS="$(tr '\n' ' ' < "$PF1_SURVIVORS_FILE")"
N_SURVIVORS="$(grep -c . "$PF1_SURVIVORS_FILE")"

if [[ "$N_SURVIVORS" -lt "$PF1_MIN_SURVIVORS" ]]; then
  echo "★ 停止条件 2 —— 生存 $N_SURVIVORS 本 < $PF1_MIN_SURVIVORS 本。" >&2
  exit 2
fi

banner "0. 測定条件(⛔ preregister の凍結値。1文字も動かさない)"
cat <<EOF
  ベンチマーク  : $BENCHMARK
  分割          : dev(⛔ HOLDOUT は開けない)
  n             : $PF1_SAMPLE_N(DEV 全量)
  摂動器        : shuffle_choices / K=1 / seed $DEV_SEED
  出力書式      : $PROMPT_FORMAT
  狙う効果量    : $TARGET_EFFECT / 想定 ψ: $EXPECTED_PSI
  α             : 0.05(既定。⛔ --alpha は渡さない)
  ロースター    : $SURVIVORS(M=$N_SURVIVORS)
  キャッシュ    : $CACHE($( [[ -f "$CACHE" ]] && wc -l < "$CACHE" || echo 0 ) 行)
  出力          : $OUT
EOF

banner "1. 検出力(★ 参考。⛔ 判定には使わない)"
# ⛔ ここに出るのは**正規近似**検定の検出力である(欠陥 P-1)。
#   判定に使う mcnemar.py は厳密条件付き検定なので、実際はこれより悪い。
$PY -m contamlab power --n "$PF1_SAMPLE_N" --effect "$TARGET_EFFECT" \
  --discordant-rate "$EXPECTED_PSI"
echo "  ★ 上は正規近似の値である(欠陥 P-1)。厳密検定では悪い側に読むこと。"

banner "2. 本番測定"
$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$PF1_SAMPLE_N" \
  --perturbator shuffle_choices --prompt-format "$PROMPT_FORMAT" \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --yes --cache "$CACHE" --json "$OUT" $(pf1_model_flags)

banner "3. 判定 D1 / D2 / D3(★ 測る前に凍結した規則)"
$PY scripts/pf1_judge.py "$OUT"
