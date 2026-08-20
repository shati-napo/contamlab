#!/usr/bin/env bash
# scripts/72-detector-firstlight.sh — ラン detector-firstlight-01 の本番測定。
#
#   bash scripts/72-detector-firstlight.sh
#
# ★ **本プロジェクトで初めて、検出器を実モデルに当てるスクリプトである。**
#   12 ラン通じて `drop` / `p_value` / `adjusted_lcb` は1つも計算されたことがない。
#
# ---------------------------------------------------------------------------
# ★ なぜ 70-positive-control.sh ではないのか(★ 先に正直に書く)
# ---------------------------------------------------------------------------
#   `70` は 6 アーム(pc-x00〜pc-x40)が Ollama に在ることを起動時に要求し、
#   実効 α も M=6 の 0.008333 で固定されている。**本ランは汚染モデルを1本しか
#   作らない**ので、`70` はそのままでは起動できない。
#
#   ★ **`70` からの差分は2つだけである**(preregister「★ `70-positive-control.sh` は
#     そのままでは通らない」の表):
#
#       ロースター : 6 アーム → **M=2**(素のベース + df1L08t1-x40)
#                    ★ 本ランが作るモデルが1本だからである。
#                    ⛔ 結果を見てから選んだ組ではない。
#       実効 α     : 0.008333(M=6)→ **0.0250**(M=2)
#                    ★ Holm の規則 `0.05 / M` の機械的な帰結であり、
#                    jmmlu-shuffle-02 / 03 が M=2 で使った値と同一である。
#                    ⛔ 私が選んだ数字ではない。
#
#   ⛔ **`n`・摂動器・効果量・想定 ψ・`detected` の定義・判定項目 A と B は
#     `70` から1文字も変えていない。**⛔ **`70` そのものは1行も書き換えていない。**
#   ⛔ **結果を書くときに「70 を通した」とは書かない。**書けるのは
#     「70 と同じ contamlab run の経路を、M=2 で初めて通した」までである。
#
# ★ HOLDOUT は開かない。K も消費しない(摂動器は shuffle_choices のまま)。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag
require_prompt_format

# preregister「ラン: detector-firstlight-01」→「凍結した設計」の表そのまま。
BASE_ARM=pcbase-swallow31-8b-x00     # 素のベース(0% アーム。pc-03 以来ビット一致)
DIRTY_ARM=df1L08t1-x40               # ★ 複製は t1 に**事前固定**(結果を見て選ばない)
DF1_ARMS=("$BASE_ARM" "$DIRTY_ARM")
SAMPLE_N=4742          # DEV 全量。分割は凍結済みなので増やせないし、増やす必要も無い
TARGET_EFFECT=0.05     # 5pt
EXPECTED_PSI=0.4050    # パイロット②の採用値(pc-01)。検出力表もこの値で作ってある
ALPHA_EFFECTIVE=0.025  # ★ Holm の実効 α = 0.05 / M。M=2 なので 0.0250
TAG="$(env_tag)"
CACHE="$(cache_path)"
PROMPT_FORMAT="$(prompt_format)"
OUT="reports/detector-firstlight-01.$TAG.json"

banner "0. 事前条件"

[[ -f "$OUT" ]] && { echo "★ 既に $OUT がある。上書きしない。" >&2; exit 1; }

# ★ 操作チェックを通っていないと測定に進めない。飛ばすと、陰性の結果が
#   「装置が鈍い」のか「注入が入っていない」のか永久に区別できなくなる。
if [[ ! -f "reports/manipulation-check.$TAG.ok" ]]; then
  cat >&2 <<'EOF'
★ 操作チェックの通過印が無い。

  bash scripts/65-manipulation-check.sh df1L08t1-x40 df1L08t2-x40 df1L08t3-x40
  # 関門 G4(b)・G5(c)を k=3 で判定し、通ったら:
  #   touch reports/manipulation-check.$(cat reports/env-tag).ok
  # ⛔ a(差 ≥ 10pt)は関門ではない —— preregister「★ a を関門にしない」。
  #    ただし a が k=3 で頑健に合格しなければ、判定 B を
  #    「検出器が汚染を検出した」とは書かない(測る前に凍結した縛り)。

  preregister の「関門」は測定の**前提条件**である。ここを飛ばすと、
  仮に陰性でも「装置が鈍い」のか「注入が入っていない」のか区別できない。
EOF
  exit 1
fi

for arm in "${DF1_ARMS[@]}"; do
  ollama show "$arm" >/dev/null 2>&1 || { echo "★ Ollama に $arm が無い。" >&2; exit 1; }
done
echo "  2アームとも Ollama にある"
echo "  出力: $OUT"

banner "1. 設計(preregister で凍結済み。ここでは引くだけ)"
cat <<EOF
  分割          : dev(全量)
  問題数        : $SAMPLE_N
  アーム        : ${DF1_ARMS[*]}(M = ${#DF1_ARMS[@]})
  摂動器        : shuffle_choices(K = 1 / 10。**新しい摂動器を足さないので K は増えない**)
  摂動シード    : $DEV_SEED
  出力書式      : $PROMPT_FORMAT
  狙う効果量    : $TARGET_EFFECT
  想定 ψ        : $EXPECTED_PSI
  α             : 0.05(M=${#DF1_ARMS[@]} → Holm の実効 α = $ALPHA_EFFECTIVE)
  推論回数      : ${#DF1_ARMS[@]} × $SAMPLE_N × 2 = $(( ${#DF1_ARMS[@]} * SAMPLE_N * 2 )) 回
  キャッシュ    : $CACHE($( [[ -f "$CACHE" ]] && wc -l < "$CACHE" || echo 0 ) 行)
EOF

banner "2. 検出力ゲート"
# ★ α は M=2 の実効値(0.05 / 2)を渡す。既定の 0.05 で引くと Holm 補正を無視した
#   甘い検出力が出る。preregister の検出力表(2.5885pt / 必要 1,270)はこの値で作ってある。
# ⛔ ここに出る数字は**正規近似**検定の検出力である。判定に使う mcnemar.py は
#   厳密条件付き検定なので、実際はこれより悪い(欠陥 P-1)。
#   ★ 最小検出可能効果は 2.59pt ではなく**概ね 2.7pt**と読むこと。
$PY -m contamlab power --n "$SAMPLE_N" --effect "$TARGET_EFFECT" \
  --discordant-rate "$EXPECTED_PSI" --alpha "$ALPHA_EFFECTIVE"
echo
echo "  ★ 上は正規近似の値である(欠陥 P-1)。厳密検定では悪い側 —— 概ね 2.7pt と読む。"

banner "3. 判定規則(★ 測る前に固定済み。結果を見てから選ばない)"
cat <<'EOF'
  A. 偽陽性を出さない  —— pcbase-swallow31-8b-x00(素のベース)が detected = false
  B. 陽性を出せる      —— df1L08t1-x40(注入率 40% / λ=0.8)が detected = true
  C. Cochran の Q      —— ★ 報告のみ。**判定に入れない**(M=2 では
                          「アーム間で区別できている」を問う設計になっていない)

  detected = 割引後下限 adjusted_lcb > 0 かつ Holm 補正後 p < α。★ 定義は 70 のまま。

  ★ 較正曲線ではない(注入率は1点だけ)。**検出下限については何も言えない。**
  ★ 陰性対照は「fine-tune を経ていない素のベース」しかない。よって B が通っても
     「注入で検出された」と「fine-tune 一般で検出された」は分離できない
     (preregister「主張範囲 3」)。
EOF

read -r -p "実行する? [y/N] " reply
[[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "中止。"; exit 0; }

banner "4. 実行"
MODEL_FLAGS=()
for arm in "${DF1_ARMS[@]}"; do
  MODEL_FLAGS+=(--model "compat:$arm:$arm:$OLLAMA_BASE_URL")
done

$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
  --perturbator shuffle_choices --prompt-format "$PROMPT_FORMAT" \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --yes --cache "$CACHE" --json "$OUT" "${MODEL_FLAGS[@]}"

banner "5. 検出器の初撃(★ 実モデルに対する最初の drop / p_value / adjusted_lcb)"
CONTAMLAB_BASE_ARM="$BASE_ARM" CONTAMLAB_DIRTY_ARM="$DIRTY_ARM" \
$PY - "$OUT" <<'PYEOF'
import json, os, sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
base_arm = os.environ["CONTAMLAB_BASE_ARM"]
dirty_arm = os.environ["CONTAMLAB_DIRTY_ARM"]
RATES = {base_arm: 0.00, dirty_arm: 0.40}
# preregister「★ 事前予測」。**当たり外れをそのまま報告する。**
# ★ 汚染アームの 0.0523 は cc-01 が測る前に凍結した式(1問あたりの寄与 0.1308 × 0.40)
#   による**超過** drop である。ベースの drop はこれに含まれない。
PRED = {base_arm: 0.0000, dirty_arm: 0.0523}

s = data["sample"]
print(f"  n={s['n_items']} 実測 ψ̂={s['observed_discordant_rate']:.4f} "
      f"最小検出可能={s['min_detectable_effect']:.4f}"
      f"(★ 正規近似。厳密検定では悪い側 = 欠陥 P-1)\n")
print(f"  {'アーム':26s} {'注入率':>6s} {'予測drop':>9s} {'実測drop':>9s} "
      f"{'割引後下限':>11s} {'p_holm':>9s}  検出")
for m in sorted(data["models"], key=lambda m: RATES.get(m["name"], 0)):
    name = m["name"]
    print(f"  {name:26s} {RATES.get(name, 0):6.0%} {PRED.get(name, 0)*100:8.2f}pt "
          f"{m['drop']*100:+8.2f}pt {m['adjusted_lcb']:+11.4f} {m['p_holm']:9.4g}  "
          f"{'★ 検出' if m['detected'] else '—'}")

print()
base = next((m for m in data["models"] if m["name"] == base_arm), None)
dirty = next((m for m in data["models"] if m["name"] == dirty_arm), None)
h = data.get("heterogeneity")
print("  ★ 判定(preregister で測る前に固定した2項目 + 報告1項目)")
print(f"    A. 素のベースが detected=false : "
      f"{'○ 通過' if base and not base['detected'] else '× ★偽陽性。装置の欠陥の発見である'}")
print(f"    B. x40 が detected=true        : "
      f"{'○ 通過' if dirty and dirty['detected'] else '× ★40%の逐語リークすら検出できない'}")
if h is None:
    print("    C. Cochran の Q(報告のみ)     : null")
else:
    print(f"    C. Cochran の Q(報告のみ)     : "
          f"Q={h['q']:.4f} df={h['df']} p={h['p_value']:.4g}"
          f"  ★ 判定には入れない")

print()
print("  ★ 較正曲線ではない(注入率は1点)。検出下限については何も言えない。")
print("  ★ 陰性対照は素のベースだけである —— 注入の効果と fine-tune 一般の効果は分離できない。")
print("  ★ 副次の読み(注入済み 1,896 / 非注入 2,846 に分けた drop)は")
print("     python tools/split_drop_by_injection.py  で出す(★ 追加課金ゼロ・★ 報告のみ)。")
for w in data.get("warnings", []):
    print(f"  ▲ {w}")
PYEOF
