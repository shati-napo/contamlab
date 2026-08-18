#!/usr/bin/env bash
# scripts/71-calibration-curve.sh — ラン calibration-curve-01 の測定。
#
#   bash scripts/71-calibration-curve.sh cc1L08t1-x00 cc1L08t2-x00 cc1L08t3-x00
#   bash scripts/71-calibration-curve.sh --heterogeneity 1   # 複製1の測定済みアームで Q を出す
#
# ★ なぜ 70-positive-control.sh をそのまま使えないか(実装の穴。規則ではない)
#   70 は pc-01 の6アーム(pc-x00〜pc-x40)を**ハードコード**しており、出力名も
#   positive-control-01.<tag>.json に固定されている。本ランのアームは
#   cc1L08t<k>-x<rr> の 18 本で、しかも**複製ごとに判定する**(replicate-judge-01)。
#
# ★ Holm の扱い(★ 規則を緩めないための設計。測定値を1つも見ずに決めた)
#   preregister は「M=6 → 実効 α = 0.05/6 = 0.008333」を凍結している。
#   harness は「そのランに渡したモデルの本数」で Holm を掛けるので、
#   打ち切りで3アームしか測らないと m=3 になり、**凍結値より甘くなる。**
#   → **1アームずつ単独で走らせ、α に実効値 0.008333 を直接渡す**(m=1 なので
#     p_holm = p となり、閾値は凍結値そのもの)。⛔ これは Holm の第一段そのもので、
#     どのアームに対しても**凍結値より甘くならない。**
#
# ★ Cochran の Q(判定項目 C)はアームが2本以上要る。--heterogeneity <複製番号> で、
#   測定済みのアームを1回のランにまとめて出す。**応答はキャッシュ済みなので追加課金ゼロ。**
#
# ⛔ contamlab/ 配下(harness / stats / runner の採点規則)には1行も触っていない。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format

SAMPLE_N=4742          # DEV 全量(preregister の凍結値)
TARGET_EFFECT=0.05     # 5pt
EXPECTED_PSI=0.4050
EFFECTIVE_ALPHA=0.008333   # = 0.05 / 6(M=6 の実効 α。凍結値)
TAG="$(env_tag)"
CACHE="$(cache_path)"
PROMPT_FORMAT="$(prompt_format)"

if [[ ! -f "reports/manipulation-check.$TAG.ok" ]]; then
  echo "★ 操作チェックの通過印が無い(reports/manipulation-check.$TAG.ok)。" >&2
  echo "  65-manipulation-check.sh を通してから測定に進むこと。" >&2
  exit 1
fi

if [[ "${1:-}" == "--heterogeneity" ]]; then
  REP="$2"
  [[ -n "$REP" ]] || { echo "--heterogeneity には複製番号(1/2/3)が要る" >&2; exit 1; }
  shift 2
  ARMS=("$@")
  [[ ${#ARMS[@]} -ge 2 ]] || { echo "Q には2アーム以上が要る" >&2; exit 1; }
  OUT="reports/calibration-curve-01.q-t${REP}.$TAG.json"
  [[ -f "$OUT" ]] && { echo "★ 既に $OUT がある。上書きしない。" >&2; exit 1; }
  MODEL_FLAGS=()
  for arm in "${ARMS[@]}"; do MODEL_FLAGS+=(--model "compat:$arm:$arm:$OLLAMA_BASE_URL"); done
  banner "Cochran の Q(複製 $REP・${ARMS[*]})★ キャッシュ済みなので追加課金は無い"
  $PY -m contamlab run \
    --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
    --perturbator shuffle_choices --prompt-format "$PROMPT_FORMAT" \
    --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
    --k 1 --yes --cache "$CACHE" --json "$OUT" "${MODEL_FLAGS[@]}"
  exit 0
fi

ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || { echo "アーム名を渡すこと(例: cc1L08t1-x40)" >&2; exit 1; }

for arm in "${ARMS[@]}"; do
  ollama show "$arm" >/dev/null 2>&1 || { echo "★ Ollama に $arm が無い。" >&2; exit 1; }
done

banner "測定(1アームずつ・α = $EFFECTIVE_ALPHA = 0.05/6)"
cat <<EOF
  分割          : dev(全量 $SAMPLE_N 問)
  アーム        : ${ARMS[*]}
  摂動器        : shuffle_choices / シード $DEV_SEED(K = 1/10。★ HOLDOUT は開かない)
  出力書式      : $PROMPT_FORMAT
  α             : $EFFECTIVE_ALPHA(★ M=6 の実効値。1アーム単独なので p_holm = p)
  キャッシュ    : $CACHE
EOF

for arm in "${ARMS[@]}"; do
  OUT="reports/calibration-curve-01.$arm.$TAG.json"
  if [[ -f "$OUT" ]]; then
    echo "★ 既にある(飛ばす): $OUT"
    continue
  fi
  banner "アーム $arm"
  $PY -m contamlab run \
    --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
    --perturbator shuffle_choices --prompt-format "$PROMPT_FORMAT" \
    --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
    --alpha "$EFFECTIVE_ALPHA" \
    --k 1 --yes --cache "$CACHE" --json "$OUT" \
    --model "compat:$arm:$arm:$OLLAMA_BASE_URL"
done

banner "ここまでの結果(★ 判定は preregister の表が正)"
$PY - "$TAG" <<'PYEOF'
import json, sys
from pathlib import Path

tag = sys.argv[1]
rows = []
for p in sorted(Path("reports").glob(f"calibration-curve-01.cc1L08t*.{tag}.json")):
    d = json.load(open(p, encoding="utf-8"))
    for m in d["models"]:
        name = m["name"]
        rate = int(name.split("-x")[1]) / 100.0
        rep = int(name.split("t")[1].split("-")[0])
        rows.append((rate, rep, name, m["drop"], m["adjusted_lcb"], m["p_holm"], m["detected"]))

print(f"  {'アーム':18s} {'注入率':>6s} {'複製':>4s} {'drop':>9s} {'割引後下限':>11s} {'p':>10s}  検出")
for rate, rep, name, drop, lcb, p, det in sorted(rows):
    print(f"  {name:18s} {rate:6.0%} {rep:4d} {drop*100:+8.2f}pt {lcb:+11.4f} {p:10.4g}  "
          f"{'★ 検出' if det else '—'}")

print()
by_rate = {}
for rate, rep, name, drop, lcb, p, det in rows:
    by_rate.setdefault(rate, []).append(det)
print("  ★ 複製の分布(3/3 = 検出 / 0/3 = 不検出 / 1〜2/3 = 不明)")
for rate in sorted(by_rate):
    d = by_rate[rate]
    n = sum(1 for x in d if x)
    verdict = "検出" if len(d) == 3 and n == 3 else ("不検出" if len(d) == 3 and n == 0 else "不明/未了")
    print(f"    {rate:6.0%}  {n}/{len(d)}  {verdict}")
PYEOF
