#!/usr/bin/env bash
# scripts/60-production.sh — 本番。DEV は何度でも、**HOLDOUT は1構成・1回だけ。**
#
#   bash scripts/60-production.sh dev
#   bash scripts/60-production.sh holdout      ← ★ 取り返しがつかない
#
# 問題数はパイロット②の実測 ψ から **preregister の写像表**で決まる。表は値を見る前に
# 固定してある(preregister「ψ → 必要問題数の写像」)。このスクリプトは表を引くだけで、
# 表そのものは書き換えない。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SPLIT="${1:-}"
case "$SPLIT" in
  dev|holdout) ;;
  *) echo "使い方: bash scripts/60-production.sh {dev|holdout}" >&2; exit 1 ;;
esac

require_ollama
TAG="$(env_tag)"
PILOT2="reports/pilot2.$TAG.json"
DETERMINISM="reports/determinism.$TAG.json"

banner "0. 事前条件"
[[ -f "$PILOT2"      ]] || { echo "★ パイロット② の結果が無い($PILOT2)。ψ が決まらない。" >&2; exit 1; }
[[ -f "$DETERMINISM" ]] || { echo "★ 決定性の実測が無い($DETERMINISM)。50-check-determinism.sh を先に。" >&2; exit 1; }
echo "  パイロット②   : $PILOT2"
echo "  決定性の実測   : $DETERMINISM"

banner "1. ψ → 必要問題数(preregister の写像表を引く)"
read -r PSI_HAT PSI_UPPER SAMPLE_N <<< "$(
  $PY - "$PILOT2" <<'PYEOF'
import json, sys
from contamlab.stats.distributions import clopper_pearson_upper

data = json.load(open(sys.argv[1], encoding="utf-8"))
n = data["sample"]["n_items"]
psi_hat = data["sample"]["observed_discordant_rate"]
discordant = round(psi_hat * n)

# ★ 使う ψ は実測値そのものではなく Clopper-Pearson の**上側限界**である。
#   下側に誤ると検出力を過大評価する。パイロット②(01)自身がそれで
#   達成検出力 0.791 < 目標 0.80 に終わった(program.md 2026-08-03 の学び)。
#   両側95%の上側 = alpha 0.05 を両側に割る。
psi_upper = clopper_pearson_upper(discordant, n, 0.05 / 2)

# preregister「ψ → 必要問題数の写像」。**値を見る前に固定した表**。
# 表に無い ψ は直近上位の行に丸める(保守側に倒す)。
TABLE = [(0.2000, 626), (0.2500, 783), (0.3000, 940), (0.3500, 1097),
         (0.4050, 1270), (0.4500, 1411), (0.5000, 1568)]

for threshold, required in TABLE:
    if psi_upper <= threshold:
        print(f"{psi_hat:.4f} {psi_upper:.4f} {required}")
        break
else:
    print(f"{psi_hat:.4f} {psi_upper:.4f} 0")
PYEOF
)"

echo "  実測 ψ̂                        : $PSI_HAT"
echo "  Clopper-Pearson 両側95%上側    : $PSI_UPPER   ← ★ 設計にはこちらを使う"
if [[ "$SAMPLE_N" == "0" ]]; then
  cat >&2 <<EOF

★ ψ の上側限界が 0.50 を超えた。preregister の写像表は
  「> 0.5000 → **本番を実行しない**(HOLDOUT 1,922 問では 5pt に検出力が足りない)」
  と事前に宣言している。**表を書き換えず、ここで止まること。**
EOF
  exit 1
fi
echo "  → 必要問題数                   : $SAMPLE_N"

# 設計に使う ψ は表の行の値(丸めた保守側)ではなく実測の上側限界。harness には
# こちらを渡す。問題数だけが表で決まる。
EXPECTED_PSI="$PSI_UPPER"
TARGET_EFFECT=0.05
CACHE="$(cache_path)"
OUT="reports/production-$SPLIT.$TAG.json"

banner "2. 設計"
cat <<EOF
  分割          : $SPLIT
  問題数        : $SAMPLE_N
  摂動器        : shuffle_choices(K = 1 / 10。摂動器は1種類のみ)
  狙う効果量    : $TARGET_EFFECT(5 ポイント)
  想定 ψ        : $EXPECTED_PSI
  α             : 0.05(既定。Holm は harness が判定側で行う。M=2 → 実効 0.0250)
  モデル        : $(printf '%s ' $(for e in "${ROSTER[@]}"; do echo "${e%%|*}"; done))
  キャッシュ    : $CACHE($( [[ -f "$CACHE" ]] && wc -l < "$CACHE" || echo 0 ) 行)
  出力          : $OUT
EOF

banner "2b. 検出力ゲート"
# cmd_run は --yes が無いと検出力ゲートに到達する前に止まる(cli.py:121-128)ので、
# ゲートの可否はここで別に見る。**通らないなら問題を投げる前に分かる。**
$PY -m contamlab power --n "$SAMPLE_N" --effect "$TARGET_EFFECT" \
  --discordant-rate "$EXPECTED_PSI"

if [[ "$SPLIT" == "holdout" ]]; then
  banner "★ HOLDOUT — 1構成・1回だけ"
  cat <<'EOF'
  HOLDOUT 1,922 問は「まだ誰にも見られていない」ことに全価値がある。
  一度使えば、その構成で得た数字が最終結果であり、**やり直しは効かない。**
  条件を変えて引き直すのは、有意になるまで回すのと同じである。

  進む前に確認すること:
    - DEV での構成が確定しているか(摂動器・シード・問題数・モデル)
    - 決定性の実測が通っているか
    - preregister の jmmlu-shuffle-02 節に、この構成が**実行前に**書かれているか
EOF
  if [[ -f "$OUT" ]]; then
    echo >&2
    echo "★ 既に $OUT がある。HOLDOUT は既に開封済みである。2回目は実行しない。" >&2
    exit 1
  fi
  echo
  read -r -p "続けるなら 'open holdout' と入力: " reply
  [[ "$reply" == "open holdout" ]] || { echo "中止。"; exit 0; }
else
  read -r -p "実行する? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "中止。"; exit 0; }
fi

banner "3. 実行"
$PY -m contamlab run \
  --benchmark "$BENCHMARK" --seed "$DEV_SEED" --split "$SPLIT" --sample-n "$SAMPLE_N" \
  --perturbator shuffle_choices \
  --target-effect "$TARGET_EFFECT" --expected-discordant-rate "$EXPECTED_PSI" \
  --k 1 --yes --cache "$CACHE" --json "$OUT" $(model_flags)

banner "4. 判定の材料"
$PY - "$OUT" <<'PYEOF'
import json, sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
s = data["sample"]
print(f"  n={s['n_items']} 必要={s['required_n']} 最小検出可能={s['min_detectable_effect']:.4f}")
print(f"  実測 ψ̂={s['observed_discordant_rate']:.4f} 達成検出力={s['observed_power']:.3f}")
print()
for m in data["models"]:
    print(f"  {m['name']}")
    print(f"      drop={m['drop']:+.4f}  adjusted_lcb={m['adjusted_lcb']:+.4f}"
          f"  p_holm={m['p_holm']:.4g}  detected={m['detected']}")

h = data.get("heterogeneity")
print()
if h is None:
    print("  ★ heterogeneity が null。モデル1本では判定できない = 採用基準4を満たせない。")
else:
    print(f"  不均一さ Q={h['q']:.4f} df={h['df']} p={h['p_value']:.4g} I²={h['i_squared']:.3f}")
    print(f"      {h['interpretation']}")
    if not h["heterogeneous"]:
        print("  ★ 不均一さが有意でない。**一律に落ちたのは「摂動が難しくなっただけ」と")
        print("    区別できない。** その場合はそう報告する(program.md の警告)。")

for w in data.get("warnings", []):
    print(f"  ▲ {w}")
PYEOF
