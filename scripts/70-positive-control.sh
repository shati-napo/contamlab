#!/usr/bin/env bash
# scripts/70-positive-control.sh — ラン positive-control-01 の本番測定。
#
#   bash scripts/70-positive-control.sh
#
# ★ 60-production.sh は使えない。あちらは M=2 専用の「ψ → 必要問題数」の写像表を引くが、
#   本ランは **n が DEV 全量 4,742 に固定済み**(preregister で確定)で、パイロット②も
#   写像表も通らない。表を無理に流用すると、値を見てから規則を選ぶことになる。
#
# ★ HOLDOUT は開かない。K も消費しない(摂動器は shuffle_choices のまま)。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format

# preregister「設計(2026-08-08 に固定。以後変更しない)」の表そのまま。
PC_ARMS=(pc-x00 pc-x02 pc-x05 pc-x10 pc-x20 pc-x40)
SAMPLE_N=4742          # DEV 全量。分割は凍結済みなので増やせないし、増やす必要も無い
TARGET_EFFECT=0.05     # 5pt
EXPECTED_PSI=0.4050    # パイロット②の採用値。検出力表もこの値で作ってある
TAG="$(env_tag)"
CACHE="$(cache_path)"
PROMPT_FORMAT="$(prompt_format)"
OUT="reports/positive-control-01.$TAG.json"

banner "0. 事前条件"

[[ -f "$OUT" ]] && { echo "★ 既に $OUT がある。上書きしない。" >&2; exit 1; }

# ★ 操作チェックを通っていないと測定に進めない。飛ばすと、陰性の結果が
#   「装置が鈍い」のか「注入が入っていない」のか永久に区別できなくなる。
if [[ ! -f "reports/manipulation-check.$TAG.ok" ]]; then
  cat >&2 <<'EOF'
★ 操作チェックの通過印が無い。

  bash scripts/65-manipulation-check.sh
  # 通ったら: touch reports/manipulation-check.$(cat reports/env-tag).ok

  preregister の「★ 操作チェック」は測定の**前提条件**である。ここを飛ばすと、
  仮に全アームが陰性でも「装置が鈍い」のか「注入が入っていない」のか区別できない。
EOF
  exit 1
fi

for arm in "${PC_ARMS[@]}"; do
  ollama show "$arm" >/dev/null 2>&1 || { echo "★ Ollama に $arm が無い。" >&2; exit 1; }
done
echo "  6アームすべて Ollama にある"
echo "  出力: $OUT"

banner "1. 設計(preregister で凍結済み。ここでは引くだけ)"
cat <<EOF
  分割          : dev(全量)
  問題数        : $SAMPLE_N
  アーム        : ${PC_ARMS[*]}
  摂動器        : shuffle_choices(K = 1 / 10。**新しい摂動器を足さないので K は増えない**)
  摂動シード    : $DEV_SEED
  出力書式      : $PROMPT_FORMAT
  狙う効果量    : $TARGET_EFFECT
  想定 ψ        : $EXPECTED_PSI
  α             : 0.05(M=6 → Holm の実効 α = 0.00833)
  推論回数      : 6 × $SAMPLE_N × 2 = $((6 * SAMPLE_N * 2)) 回
  キャッシュ    : $CACHE($( [[ -f "$CACHE" ]] && wc -l < "$CACHE" || echo 0 ) 行)
EOF

banner "2. 検出力ゲート"
# ★ α は M=6 の実効値(0.05 / 6)を渡す。既定の 0.05 で引くと Holm 補正を無視した
#   甘い検出力が出る。preregister の検出力表(2.99pt / 必要 1,694)はこの値で作ってある。
$PY -m contamlab power --n "$SAMPLE_N" --effect "$TARGET_EFFECT" \
  --discordant-rate "$EXPECTED_PSI" --alpha 0.008333

banner "3. 判定規則(★ 測る前に固定済み。結果を見てから選ばない)"
cat <<'EOF'
  A. 偽陽性を出さない  —— pc-x00(0%)が detected = false
  B. 陽性を出せる      —— pc-x40(40%)が detected = true
  C. 区別できている    —— Cochran の Q が有意

  較正曲線 = 各アームの (X, drop, adjusted_lcb, detected) の組。
  検出下限 = detected が true になった最小の X。水準は離散なので**帯**で報告する。
  ★ 中間の水準を後から足さない(足せば「見えるまで刻む」ことになる)。
EOF

read -r -p "実行する? [y/N] " reply
[[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "中止。"; exit 0; }

banner "4. 実行"
MODEL_FLAGS=()
for arm in "${PC_ARMS[@]}"; do
  MODEL_FLAGS+=(--model "compat:$arm:$arm:$OLLAMA_BASE_URL")
done

$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
  --perturbator shuffle_choices --prompt-format "$PROMPT_FORMAT" \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --yes --cache "$CACHE" --json "$OUT" "${MODEL_FLAGS[@]}"

banner "5. 較正曲線"
$PY - "$OUT" <<'PYEOF'
import json, sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
RATES = {"pc-x00": 0.00, "pc-x02": 0.02, "pc-x05": 0.05,
         "pc-x10": 0.10, "pc-x20": 0.20, "pc-x40": 0.40}
# preregister「★ 事前予測」。**当たり外れをそのまま報告する。**
PRED = {"pc-x00": 0.0000, "pc-x02": 0.0115, "pc-x05": 0.0288,
        "pc-x10": 0.0575, "pc-x20": 0.1150, "pc-x40": 0.2300}

s = data["sample"]
print(f"  n={s['n_items']} 実測 ψ̂={s['observed_discordant_rate']:.4f} "
      f"最小検出可能={s['min_detectable_effect']:.4f}\n")
print(f"  {'アーム':10s} {'注入率':>6s} {'予測drop':>9s} {'実測drop':>9s} "
      f"{'割引後下限':>11s} {'p_holm':>9s}  検出")
detected_rates = []
for m in sorted(data["models"], key=lambda m: RATES.get(m["name"], 0)):
    name = m["name"]
    print(f"  {name:10s} {RATES.get(name, 0):6.0%} {PRED.get(name, 0)*100:8.2f}pt "
          f"{m['drop']*100:+8.2f}pt {m['adjusted_lcb']:+11.4f} {m['p_holm']:9.4g}  "
          f"{'★ 検出' if m['detected'] else '—'}")
    if m["detected"]:
        detected_rates.append(RATES.get(name, 0))

print()
x00 = next((m for m in data["models"] if m["name"] == "pc-x00"), None)
x40 = next((m for m in data["models"] if m["name"] == "pc-x40"), None)
h = data.get("heterogeneity")
print("  ★ 判定(preregister で測る前に固定した3項目)")
print(f"    A. pc-x00 が detected=false : "
      f"{'○ 通過' if x00 and not x00['detected'] else '× ★偽陽性。装置の欠陥の発見である'}")
print(f"    B. pc-x40 が detected=true  : "
      f"{'○ 通過' if x40 and x40['detected'] else '× ★40%の逐語リークすら検出できない'}")
if h is None:
    print("    C. Cochran の Q             : × null(アームが足りない)")
else:
    print(f"    C. Cochran の Q             : "
          f"{'○ 有意' if h['heterogeneous'] else '× 有意でない'}  "
          f"(Q={h['q']:.4f} df={h['df']} p={h['p_value']:.4g})")

print()
if detected_rates:
    floor = min(r for r in detected_rates if r > 0) if any(detected_rates) else 0.0
    below = [r for r in RATES.values() if r < floor]
    print(f"  検出下限の帯: {max(below) if below else 0:.0%} では検出せず、{floor:.0%} で検出")
else:
    print("  ★ どのアームも検出しなかった。**注入率 40% でも見えない**という実測である。")
for w in data.get("warnings", []):
    print(f"  ▲ {w}")
PYEOF
