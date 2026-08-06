#!/usr/bin/env bash
# scripts/40-pilot.sh — パイロット①(採点の健全性)と②(ψ の実測)。
#
#   bash scripts/40-pilot.sh 1        床効果と採点の健全性(identity・70問)
#   bash scripts/40-pilot.sh 2        ψ の実測(shuffle_choices・250問)
#
# ★ n を 01 の 50 / 200 から 70 / 250 に上げてある。理由は事後正当化を避けるため ★
#
# 01 のパイロットは ψ=0.30 を想定して n を決めた。しかしパイロット②で ψ̂=0.335、
# その Clopper-Pearson 上側限界 0.4050 を**既に知っている**。知っていながら 0.30 と
# 宣言し直すのは、通るように想定を選ぶことである。0.4050 で宣言し直すと:
#
#   ①: 目標 20pt / ψ=0.4050 → 必要 61 問。n=50 では足りない → **70 問**
#   ②: 目標 10pt / ψ=0.4050 → 必要 249 問。n=200 では足りない → **250 問**
#
# どちらも `--force-underpowered` を使わずにゲートを通る。`take_deterministic` は
# prefix 安定(benchmark.py:227)なので **70 ⊂ 250 ⊂ 本番** となり、応答キャッシュは
# 1問も無駄にならない。n=70 は preregister が既に宣言しているパイロット③の設計値と同じ。
#
# ★ --alpha は既定の 0.05 のまま渡す ★
# 0.025(M=2 の実効水準)を渡すと信頼区間と Cochran の Q まで二重補正になる。
# Holm 補正は harness が判定側で行う。問題数はこちらが ψ→問題数の表から決める。
# (preregister「ψ → 必要問題数の写像」および program.md 2026-08-03 の学び)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAGE="${1:-}"
case "$STAGE" in
  1) SAMPLE_N=70;  PERTURBATOR=identity;        TARGET_EFFECT=0.20; LABEL="パイロット①(採点の健全性・床効果)" ;;
  2) SAMPLE_N=250; PERTURBATOR=shuffle_choices; TARGET_EFFECT=0.10; LABEL="パイロット②(ψ の実測)" ;;
  *) echo "使い方: bash scripts/40-pilot.sh {1|2}" >&2; exit 1 ;;
esac

EXPECTED_PSI=0.405   # preregister: パイロット②(01)の ψ̂=0.335 の CP 両側95%上側

require_ollama
require_env_tag
[[ -f "$BENCHMARK" ]] || { echo "ベンチマークが無い。先に 20-rebuild-benchmark.sh。" >&2; exit 1; }

CACHE="$(cache_path)"
OUT="reports/pilot${STAGE}.$(env_tag).json"
mkdir -p reports data/cache

banner "$LABEL"
cat <<EOF
  問題数        : $SAMPLE_N(DEV から決定論的に抽出)
  摂動器        : $PERTURBATOR
  狙う効果量    : $TARGET_EFFECT
  想定 ψ        : $EXPECTED_PSI
  α             : 0.05(既定のまま。Holm は harness が判定側で行う)
  モデル        : $(printf '%s ' $(for e in "${ROSTER[@]}"; do echo "${e%%|*}"; done))
  キャッシュ    : $CACHE($( [[ -f "$CACHE" ]] && wc -l < "$CACHE" || echo 0 ) 行)
  出力          : $OUT
EOF

banner "検出力ゲート(問題を投げる前に確認する)"
# cmd_run は --yes が無いと**検出力ゲートに到達する前に**終了コード 3 で止まる
# (cli.py:121-128)。つまり見積もりだけではゲートの可否が分からない。
# program.md:「まず検出力。何問必要かを計算してから問題を集める」に従い、先に見る。
$PY -m contamlab power --n "$SAMPLE_N" --effect "$TARGET_EFFECT" \
  --discordant-rate "$EXPECTED_PSI"

# 見積もりだけ(--yes を付けないと CLI は回数を出して止まる)
banner "呼び出し回数の見積もり"
set +e
$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
  --perturbator "$PERTURBATOR" \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --cache "$CACHE" $(model_flags)
status=$?
set -e
# 終了コード 3 = 承認待ち(EXIT_NEEDS_CONFIRMATION)。それ以外の非ゼロは本当の失敗。
if [[ $status -ne 0 && $status -ne 3 ]]; then
  echo "★ 見積もりの段階で失敗した(exit $status)。実行しない。" >&2
  exit $status
fi

if [[ "${2:-}" != "-y" ]]; then
  read -r -p "実行する? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "中止。"; exit 0; }
fi

banner "実行"
# 進捗は別端末で: watch -n 30 "wc -l $CACHE"(追記専用なので行数が代理指標になる)
$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
  --perturbator "$PERTURBATOR" \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --yes --cache "$CACHE" --json "$OUT" $(model_flags)

banner "読む値 / 読まない値"
cat <<'EOF'
  読む   : 解釈不能率(モデル別・条件別)/ 素の正解率 / モデル別の ψ / conflicts
  読まない: drop / p_value / p_holm / adjusted_lcb
           ★ 汚染の判定は HOLDOUT でのみ行う。パイロットの効果量は見ない。

  ★ ①(identity)の ψ=0 を「決定的である証拠」と読まないこと。identity は摂動後の
    プロンプトが原文と同一なので、**2回目はキャッシュに当たり、同じ文字列が返る。**
    ψ=0 は構成上そうなるだけで、モデルを2回呼んだ結果ではない。
    決定性は 50-check-determinism.sh が**別のキャッシュに取り直して**測る。
EOF

$PY - "$OUT" <<'PYEOF'
import json, sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
sample = data["sample"]
n = sample["n_items"]
# ★ observed_power は None になりうる(2026-08-06 実機で判明)。identity 摂動では
#   不一致ペアが1件も出ないので、harness は「不一致率が狙う効果量を下回った」として
#   検出力を算出不能にする。**これは異常ではなくパイロット①の正常な結果**である
#   (何も変えなければ差はちょうど0)。ここで :.3f を当てて落ちると、
#   **本文の数字を印字する前に要約が死ぬ。** 読むべき値は解釈不能率なので、
#   整形の都合で読めなくなるのは本末転倒だった。
power = sample["observed_power"]
power_text = f"{power:.3f}" if power is not None else "算出不能(不一致ペアが0件)"
print()
print(f"  n = {n} / 全体の ψ̂ = {sample['observed_discordant_rate']:.4f}"
      f" / 達成検出力 = {power_text}")
print()
print(f"  {'モデル':<16} {'素の正解率':>10} {'摂動後':>8} {'ψ̂(モデル別)':>14}"
      f" {'解釈不能(原/摂)':>18}")
for m in data["models"]:
    t = m["table"]
    # ★ モデル別の ψ は JSON に無いので分割表から出す。採用値 0.4050 は別のモデル
    #   1本の実測から来ており、ψ はモデル依存(正解率が違えば不一致の出方も違う)。
    psi = (t["only_original"] + t["only_perturbed"]) / n if n else float("nan")
    print(f"  {m['name']:<16} {m['accuracy_original']:>10.4f}"
          f" {m['accuracy_perturbed']:>8.4f} {psi:>14.4f}"
          f" {m['unparsed_original']:>8} / {m['unparsed_perturbed']:<8}")

if data.get("warnings"):
    print()
    for w in data["warnings"]:
        print(f"  ▲ {w}")

# 解釈不能率が条件間でずれたら「落ちたのは能力ではなく採点」(program.md の警告)。
for m in data["models"]:
    lo, hi = m["unparsed_original"], m["unparsed_perturbed"]
    if n and max(lo, hi) / n > 0.05:
        print(f"\n  ★ {m['name']}: 解釈不能率が 5% を超えている({lo}/{hi} of {n})。"
              f"停止条件に該当しないか確認すること。")
    elif abs(lo - hi) / max(n, 1) > 0.02:
        print(f"\n  ★ {m['name']}: 解釈不能率が条件間でずれている({lo} vs {hi})。"
              f"落ちたのは能力ではなく採点かもしれない。")
PYEOF

echo
echo "次: bash scripts/50-check-determinism.sh   ← GPU での決定性を実測する"
