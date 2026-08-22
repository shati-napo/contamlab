#!/usr/bin/env bash
# scripts/pf1-orchestrate.sh — ラン perturbation-floor-01 の段の順序。
#
#   nohup bash scripts/pf1-orchestrate.sh >> reports/pf1-orchestrate.log 2>&1 &
#
# ★ **段の順序と分岐をインスタンス側に置く。**手元の PC が落ちてもランは自分で
#   進み、止まる(cc-01 で停電を越えた仕掛け・df1 で実戦を通った)。
#
# ★ **df1-orchestrate.sh から3つの修理をそのまま引き継いでいる**(2026-08-21 の実機で
#   1回ずつ通ったもの):
#     ① 依存を作る段を手順書ではなくオーケストレータに置く(飛ばされないように)
#     ② grace() はインスタンス名を直書きせず、見張りが確定させた instance_id を読む
#     ③ 常駐同期に flock の fd を相続させない(9>&-)
#   ⛔ **複製せずに書き直すと3つとも復活する。**
#
# ★ **本ランは学習を1回もしない。**推論だけである(停止条件 6)。
#   よって finetune/.venv も GPU メモリの奪い合いも無い。

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -s reports/env-tag ] || { echo "★ reports/env-tag が無い。30-record-environment.sh を先に。"; exit 1; }
TAG="$(cat reports/env-tag)"

ts() { date -u +%FT%TZ; }

# ★ 段の境目ごとに成果物をホストの外へ出す(2026-08-20 の $17.90 の損失から)。
#   ⛔ 同期の失敗ではランを止めない。
SYNC=scripts/df1_sync.py
sync_now() { timeout 300 python3 "$SYNC" once --why "$1" >> reports/pf1-sync.log 2>&1 || true; }
say() { echo "=== $* $(ts)"; sync_now "$*"; }

# ★ 終わったらハード期限を手前に寄せる。⛔ 後ろへは動かさない。
GRACE_HOURS=1
grace() {
  local cur_epoch new_epoch iid
  cur_epoch=$(python3 -c "import json;print(int(json.load(open('reports/cost-watchdog.json'))['deadline_epoch']))" 2>/dev/null || echo 0)
  new_epoch=$(( $(date -u +%s) + GRACE_HOURS * 3600 ))
  if [ "$cur_epoch" -gt 0 ] && [ "$new_epoch" -ge "$cur_epoch" ]; then
    say "[grace] いまの期限のほうが早い。1秒も後ろへ動かさない"
    return 0
  fi
  say "[grace] ハード期限を現在 +${GRACE_HOURS}h に寄せ直す(★ 厳しい側にしか動かさない)"
  # ★ 対象は名前で探さない。見張りが arm 時に確定させた instance_id を使う(修理②)。
  iid=$(python3 -c "import json;print(json.load(open('reports/cost-watchdog.json'))['instance_id'])" 2>/dev/null || echo "")
  [ -n "$iid" ] || { say "[grace] 見張りの状態が読めない。⛔ 期限は触らない"; return 0; }
  # ⛔ --budget-usd は preregister「停止条件 5」の $15。直書きの置き去りを避けるため変数から。
  python3 scripts/cost_watchdog.py arm --instance-id "$iid" \
      --price-usd-per-hour 1.99 \
      --hard-usd "$(python3 -c "print(round(1.99*${GRACE_HOURS},2))")" --budget-usd 15 \
      --started-at-utc "$(date -u +%FT%TZ)" --interval 300 || return 0
  sudo systemctl restart contamlab-watchdog || true
  python3 scripts/cost_watchdog.py status || true
}

stop_run() {
  echo "=== [STOP] $* $(ts)"
  { echo "$(ts)  $*"; } >> reports/pf1-STOPPED.txt
  sync_now "[STOP] $*"
  grace
  exit 1
}

say "[START] perturbation-floor-01 のオーケストレータ(env-tag ${TAG})"

exec 9>"reports/.pf1-orchestrate.lock"
if ! flock -n 9; then
  say "[LOCK] 既にオーケストレータが走っている。何もしない"
  exit 0
fi

# ---- 保全(★ 測定より先に通す)-----------------------------------------------
# ⛔ 外へ出せないまま GPU を回さない —— 2026-08-20 に成果物を失った形である。
echo "=== [S] 成果物の保全: 置き先へ push できることを確かめる $(ts)"
python3 "$SYNC" init || stop_run "保全の導通確認に失敗(⛔ 成果物を外に出せない状態で測定を始めない)"
if ! pgrep -f "df1_sync.p[y] daemon" > /dev/null; then
  # ★ 9>&- で錠を相続させない(修理③)。
  setsid nohup python3 "$SYNC" daemon --interval 900 >> reports/pf1-sync.log 2>&1 < /dev/null 9>&- &
  echo "=== [S] 常駐の同期を始めた(900秒ごと)$(ts)"
fi

# ---- G0: ベンチマークの同一性(停止条件 1)------------------------------------
say "[G0] ベンチマークの pin を確かめる"
python3 - <<'PYEOF' || exit 1
import hashlib, json, sys
from pathlib import Path
EXPECT = "8aa877e57335daca61a9aa4e676e78e1da5a7608806f6e959e1e67bc56317745"
p = Path("data/jmmlu.jsonl")
if not p.exists():
    print("★ 停止条件 1 —— data/jmmlu.jsonl が無い"); sys.exit(1)
got = hashlib.sha256(p.read_bytes()).hexdigest()
rows = sum(1 for _ in p.open(encoding="utf-8"))
print(f"  sha256 {got[:16]}…  rows {rows}")
if got != EXPECT or rows != 6664:
    print(f"★ 停止条件 1 —— pin が合わない(期待 {EXPECT[:16]}… / 6664 行)"); sys.exit(1)
print("  ★ 一致。pc-03 以来の凍結値どおり")
PYEOF
[ $? -eq 0 ] || stop_run "G0: ベンチマークの pin が合わない(停止条件 1)"

# ---- 段 2: モデルの取得(G1)---------------------------------------------------
say "[P] 5 本を取得して疎通を見る(80-pf1-pull.sh)"
bash scripts/80-pf1-pull.sh || stop_run "G1: モデルの取得か疎通に失敗"

# ---- 段 3: パイロット(G2 / 停止条件 2)----------------------------------------
say "[C] パイロット n=150 —— 書式 C に従えないモデルを本番から外す(81-pf1-pilot.sh)"
bash scripts/81-pf1-pilot.sh
rc=$?
if [ $rc -eq 2 ]; then
  stop_run "停止条件 2: パイロットの生存が 3 本未満(⛔ 線を下げて測り直さない)"
elif [ $rc -ne 0 ]; then
  stop_run "パイロットが異常終了した(exit $rc)"
fi

# ---- 段 4: 本番 + 判定 ---------------------------------------------------------
say "[M] 本番 n=4,742(82-pf1-production.sh)—— ★ 摂動の床を初めて測る"
bash scripts/82-pf1-production.sh || stop_run "本番測定に失敗"

say "[DONE] 測定と判定を終えた。結果: reports/perturbation-floor-01.${TAG}.json"
sync_now "[DONE] 最終"
grace
