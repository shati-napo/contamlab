#!/usr/bin/env bash
# scripts/cc01-orchestrate.sh — ラン calibration-curve-01 を**インスタンスの上で**最後まで回す。
#
#   nohup bash scripts/cc01-orchestrate.sh > reports/cc01-orchestrate.log 2>&1 &
#
# ★ なぜこれが要るか(2026-08-18 に足した。★ 規則は1つも変えていない)
#   lambda-ladder-01 の $47.82 の事故から「**課金を止める仕掛けはリモートに置く**」を
#   学んで cost_watchdog.py を作った。だが**ランを進める仕掛け**は手元のエージェントの
#   ままだった。手元の PC が落ちると学習は続くが**次の段が始まらず**、GPU は空回りした
#   まま $90 のハード期限まで課金される。**同じ形の穴である。**
#   → 段の順序と関門を、**判定の線を1つも動かさずに**スクリプトへ移した。
#
# ★ 判定はすべて scripts/cc01_gate.py が持つ(閾値はそちらに定数として書いてある)。
#   このスクリプトは**順序と分岐だけ**を担う。⛔ 数字を1つも持たない。
#
# ★ 冪等である —— 既に在るものは作り直さない(学習・マージ・GGUF・測定)。
#   途中で落ちても、もう一度起動すれば続きから進む。

set -uo pipefail
cd "$HOME/contamlab"
export CONTAMLAB_ENV_TAG=lambda-a100-cc01-20260818
export PYTHONIOENCODING=utf-8
export LC_ALL="${LC_ALL:-C.UTF-8}"
TAG="$CONTAMLAB_ENV_TAG"
PY_VENV=finetune/.venv/bin/python
RATES="00 40 20 10 05 02"     # ★ preregister の凍結した順序。上から下ろす
MEASURED=""

ts() { date -u +%FT%TZ; }
say() { echo "=== $* $(ts)"; }

# ★ 終わったら(正常でも異常でも)ハード期限を手前に寄せる。
#   ⛔ 期限を**後ろへ動かすことはしない**(緩和になる)。手元が死んでいても、
#   成果物を回収する猶予を残しつつ、空回りを 3h で止める。
GRACE_HOURS=8            # ★ 成果物を回収する猶予。8h × $1.99 = $15.92
grace() {
  local cur_epoch new_epoch
  cur_epoch=$(python3 -c "import json;print(int(json.load(open('reports/cost-watchdog.json'))['deadline_epoch']))" 2>/dev/null || echo 0)
  new_epoch=$(( $(date -u +%s) + GRACE_HOURS * 3600 ))
  # ★ 期限は**手前にしか動かさない。**後ろへ動かせば規則の緩和になる。
  if [ "$cur_epoch" -gt 0 ] && [ "$new_epoch" -ge "$cur_epoch" ]; then
    say "[grace] いまの期限のほうが早い。1秒も後ろへ動かさない"
    return 0
  fi
  say "[grace] ハード期限を現在 +${GRACE_HOURS}h に寄せ直す(★ 厳しい側にしか動かさない)"
  python3 scripts/cost_watchdog.py arm --name contamlab-cc01 \
      --price-usd-per-hour 1.99 --hard-usd 15.92 --budget-usd 95 \
      --started-at-utc "$(date -u +%FT%TZ)" --interval 300 || return 0
  sudo systemctl restart contamlab-watchdog || true
  python3 scripts/cost_watchdog.py status || true
}

stop_run() {
  say "[STOP] $*"
  { echo "$(ts)  $*"; } >> reports/cc01-STOPPED.txt
  grace
  exit 1
}

train_one() {   # $1=rate $2=replicate
  local rate="$1" t="$2" arm="cc1t${2}-x${1}" lam="cc1L08t${2}-x${1}"
  if [ ! -f "models/${arm}/train.json" ]; then
    say "[T] 学習 ${arm}"
    sudo systemctl stop ollama || true
    $PY_VENV finetune/train_lora.py --run calibration-curve-01 --recipe R1 \
        --rate "$rate" --replicate "$t" || stop_run "学習に失敗: ${arm}"
  else
    say "[T] 既にある(飛ばす) ${arm}"
  fi
  if [ ! -d "models/${lam}" ]; then
    say "[L] lambda=0.8 でマージ ${lam}"
    sudo systemctl stop ollama || true
    $PY_VENV finetune/scale_adapter.py --run calibration-curve-01 --lambda-step L1 \
        --rate "$rate" --replicate "$t" || stop_run "マージに失敗: ${lam}"
  else
    say "[L] 既にある(飛ばす) ${lam}"
  fi
  sudo systemctl start ollama; sleep 5
  if ! ollama show "$lam" >/dev/null 2>&1; then
    say "[G] GGUF 変換と登録 ${lam}"
    bash finetune/to_gguf.sh "$lam" || stop_run "GGUF 変換に失敗: ${lam}"
  else
    say "[G] 既に Ollama にある(飛ばす) ${lam}"
  fi
  df -h / | tail -1
}

check_arms() {  # $1=out-file $2..=arms
  local out="$1"; shift
  local rate_suffix="${1##*-x}"
  say "[C] 操作チェック $* -> $out"
  sudo systemctl start ollama; sleep 3
  if [ "$rate_suffix" = "00" ]; then
    CONTAMLAB_PLACEBO_IDS=data/injection/pc-x40.ids \
      bash scripts/65-manipulation-check.sh "$@" > "$out" 2>&1 || true
  else
    bash scripts/65-manipulation-check.sh "$@" > "$out" 2>&1 || true
  fi
  cat "$out"
}

say "[START] calibration-curve-01 のオーケストレータ"

# ★ 二重起動を機械が拒む。同じアームを2本同時に学習すると、どちらが残ったのか
#   分からなくなる(複製の同一性が壊れる)。
exec 9>"reports/.cc01-orchestrate.lock"
if ! flock -n 9; then
  say "[LOCK] 既にオーケストレータが走っている。何もしない"
  exit 0
fi

# ★ 手元のエージェントが先に起動していた学習があれば、それが終わるまで待つ。
#   ⛔ 二重に学習を始めない(冪等の判定は train.json の有無で行うため、
#      走っている最中は「無い」に見える)。
while pgrep -f "finetune/train_lora.py" > /dev/null; do
  say "[WAIT] 先に走っている学習の終了を待つ(120s)"
  sleep 120
done

for rate in $RATES; do
  say "[RATE] x${rate} に入る"

  # ---- 複製1本目。x00 だけは、ここで安価な異常検知を通す ----
  train_one "$rate" 1
  if [ "$rate" = "00" ]; then
    check_arms "reports/cc01-check-x00-t1.txt" "cc1L08t1-x00"
    python3 scripts/cc01_gate.py anomaly --check reports/cc01-check-x00-t1.txt \
      || stop_run "x00 の1本目で解釈不能率が 50% を超えた(安価な異常検知)"
  fi

  # ---- 残りの複製 ----
  train_one "$rate" 2
  train_one "$rate" 3

  # ---- 停止条件 2(入力側の一致) ----
  say "[I] 入力側の一致(停止条件 2)x${rate}"
  python3 scripts/cc01_gate.py inputs \
      --arms "cc1t1-x${rate}" "cc1t2-x${rate}" "cc1t3-x${rate}" \
    || stop_run "停止条件 2: x${rate} の3本で入力側が一致しない"

  # ---- 操作チェック(3本ぶん) ----
  CHECK="reports/cc01-check-x${rate}.txt"
  check_arms "$CHECK" "cc1L08t1-x${rate}" "cc1L08t2-x${rate}" "cc1L08t3-x${rate}"

  python3 scripts/cc01_gate.py stop6 --check "$CHECK" \
    || stop_run "停止条件 6: x${rate} で注入群のほうが正解率が低い"

  if [ "$rate" = "00" ]; then
    say "[GATE] x00 の関門(条件 c を k=3 で読む)"
    python3 scripts/cc01_gate.py gate-c --check "$CHECK" \
      || stop_run "停止条件 5: x00 が関門(c <= 5%)を通らなかった。埋め草を使う較正曲線の設計は、この条件では成立しない"
  fi

  # ---- 測定(n=4,742 × 2条件 × 3複製)----
  touch "reports/manipulation-check.${TAG}.ok"
  say "[M] 測定 x${rate}"
  bash scripts/71-calibration-curve.sh \
      "cc1L08t1-x${rate}" "cc1L08t2-x${rate}" "cc1L08t3-x${rate}" \
    || stop_run "測定に失敗: x${rate}"

  MEASURED="$MEASURED $rate"

  # ---- 検出の判定と打ち切り ----
  say "[D] 検出の判定 x${rate}"
  python3 scripts/cc01_gate.py detect --tag "$TAG" --rate "$rate"
  rc=$?
  if [ "$rc" -eq 2 ]; then
    say "[打ち切り] x${rate} が 3複製とも不検出。★ これより下の水準は測らない"
    break
  elif [ "$rc" -ne 0 ]; then
    stop_run "検出の判定に失敗: x${rate}"
  fi
done

# ---- Cochran の Q(判定項目 C)。★ キャッシュ済みなので追加課金ゼロ ----
NRATES=$(echo $MEASURED | wc -w)
if [ "$NRATES" -ge 2 ]; then
  for t in 1 2 3; do
    ARMS=""
    for r in $MEASURED; do ARMS="$ARMS cc1L08t${t}-x${r}"; done
    say "[Q] Cochran の Q 複製${t}:$ARMS"
    bash scripts/71-calibration-curve.sh --heterogeneity "$t" $ARMS || true
  done
else
  say "[Q] 測定したアームが1水準しかないので Q は計算しない"
fi

say "[DONE] 測定した水準:$MEASURED"
grace
