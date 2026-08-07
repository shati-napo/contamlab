#!/usr/bin/env bash
# scripts/35-select-format.sh — パイロット⓪(出力書式の選定)。
#
#   bash scripts/35-select-format.sh [-y]
#
# preregister「ラン: jmmlu-shuffle-03」。候補3書式を DEV 150 問・identity で回し、
# **解釈不能率だけ**を読んで1つ選び、reports/prompt-format に焼く。
# 以降の段(40-pilot.sh / 60-production.sh)はそのファイルを読む。
#
# ★ この段が「落ちたモデルが通る条件を、落ちたのを見てから選ぶ」ことにならないための
#   装置は4つある。preregister に全部書いてあるが、コード側の担保はここ:
#
#   1. 摂動器は identity 固定。**摂動後の応答を1件も見ずに書式が決まる**
#   2. 下の要約は **unparsed_* と n_items しか読まない。** accuracy も drop も
#      p 値も**参照しない**(読まないのではなく、コードに出てこない)
#   3. --split dev 固定。HOLDOUT には触れない
#   4. 候補は下の FORMATS で凍結。**合格ゼロなら4つ目を作らず本番を実行しない**
#
# ★ --target-effect 0.20 / --expected-discordant-rate 0.405 は**検出力ゲートを
#   通すための形式値**である。この段の目的は効果量の推定ではなく解釈不能率の測定で、
#   低下幅は読みもしない。n=150 は ψ=0.405・20pt に必要な 61 問を十分上回る。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

FORMATS=(A B C)          # ★ preregister で凍結した列挙順(介入の小さい順)。増やさない
SAMPLE_N=150
PERTURBATOR=identity
TARGET_EFFECT=0.20
EXPECTED_PSI=0.405
THRESHOLD=0.05           # 解釈不能率の事前条件(02 の停止条件と同じ)

require_ollama
require_env_tag
[[ -f "$BENCHMARK" ]] || { echo "ベンチマークが無い。先に 20-rebuild-benchmark.sh。" >&2; exit 1; }

CACHE="$(cache_path)"
mkdir -p reports data/cache

banner "パイロット⓪(出力書式の選定)"
cat <<EOF
  候補書式      : ${FORMATS[*]}(preregister で凍結・列挙順が優先順位)
  問題数        : $SAMPLE_N(DEV から決定論的に抽出)
  摂動器        : $PERTURBATOR ★摂動後の応答は見ない
  判定          : 2本とも解釈不能率 < $(python3 -c "print($THRESHOLD*100)")% なら合格。
                  合格が複数なら**列挙順が最も早いもの**。合格ゼロなら本番を実行しない
  モデル        : $(printf '%s ' $(for e in "${ROSTER[@]}"; do echo "${e%%|*}"; done))
  キャッシュ    : $CACHE($( [[ -f "$CACHE" ]] && wc -l < "$CACHE" || echo 0 ) 行)
  呼び出し      : 2 モデル × ${#FORMATS[@]} 書式 × $SAMPLE_N 問 = $((2 * ${#FORMATS[@]} * SAMPLE_N)) 回
                  ★選ばれた書式ぶんはパイロット①②でそのまま再利用される
EOF

banner "検出力ゲート(形式確認)"
$PY -m contamlab power --n "$SAMPLE_N" --effect "$TARGET_EFFECT" \
  --discordant-rate "$EXPECTED_PSI"

if [[ "${1:-}" != "-y" ]]; then
  read -r -p "実行する? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "中止。"; exit 0; }
fi

for FMT in "${FORMATS[@]}"; do
  banner "書式 $FMT を測る"
  $PY -m contamlab run \
    --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split dev --sample-n "$SAMPLE_N" \
    --perturbator "$PERTURBATOR" --prompt-format "$FMT" \
    --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
    --k 1 --yes --cache "$CACHE" \
    --json "reports/pilot0-${FMT}.$(env_tag).json" $(model_flags) \
    > "reports/pilot0-${FMT}.$(env_tag).log" 2>&1
  echo "  完了 → reports/pilot0-${FMT}.$(env_tag).json"
done

banner "選定(解釈不能率だけを読む)"
$PY - "$THRESHOLD" "$PROMPT_FORMAT_FILE" \
     $(for FMT in "${FORMATS[@]}"; do echo "${FMT}=reports/pilot0-${FMT}.$(env_tag).json"; done) <<'PYEOF'
import json, sys

threshold = float(sys.argv[1])
out_path = sys.argv[2]

# ★ ここで読むのは unparsed_original / unparsed_perturbed / n_items の3つだけ。
#   accuracy_* も drop も p_value も adjusted_lcb も**このスクリプトには出てこない。**
#   「見ない」という運用規律ではなく、**構造的に見られない**ようにしてある
#   (2026-08-07 に解釈不能 26 件を分類したときと同じ手)。
rows = []
for arg in sys.argv[3:]:
    name, _, path = arg.partition("=")
    data = json.load(open(path, encoding="utf-8"))
    n = data["sample"]["n_items"]
    models = {}
    for m in data["models"]:
        worst = max(m["unparsed_original"], m["unparsed_perturbed"])
        models[m["name"]] = (worst, worst / n if n else 1.0)
    rows.append((name, n, models))

names = sorted({k for _, _, ms in rows for k in ms})
width = max(len(k) for k in names)

print()
print(f"  {'書式':<6}" + "".join(f"{k:>{width + 12}}" for k in names) + "   判定")
for name, n, models in rows:
    cells = ""
    passed = True
    for k in names:
        count, rate = models.get(k, (None, 1.0))
        cells += f"{f'{count}/{n} ({rate:.1%})':>{width + 12}}"
        if rate >= threshold:
            passed = False
    print(f"  {name:<6}{cells}   {'✅ 合格' if passed else '❌ 不合格'}")

winners = [
    name
    for name, n, models in rows
    if all(models.get(k, (None, 1.0))[1] < threshold for k in names)
]

print()
if winners:
    chosen = winners[0]   # ★ 列挙順が最も早いもの。最小値では選ばない(preregister)
    print(f"  ★ 採用: 書式 {chosen}"
          + (f"(合格 {len(winners)} 件のうち列挙順が最も早い)" if len(winners) > 1 else ""))
    open(out_path, "w", encoding="utf-8").write(chosen + "\n")
    print(f"  {out_path} に書いた。次: bash scripts/40-pilot.sh 1")
else:
    print("  ★ 合格ゼロ。**本番を実行しない。**")
    print("    preregister「合格ゼロのときにやらないこと」のとおり、")
    print("    4つ目の書式を作らない / 採点器を緩めない / 母集団を緩めない /")
    print("    max_tokens を上げない。02 と同じ形(中止・K=0・HOLDOUT 未使用)で終える。")
    sys.exit(1)
PYEOF
