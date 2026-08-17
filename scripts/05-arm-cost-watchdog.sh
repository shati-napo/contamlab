#!/usr/bin/env bash
# scripts/05-arm-cost-watchdog.sh — 費用ウォッチドッグを**インスタンスの上で**常駐させる。
#
#   bash scripts/05-arm-cost-watchdog.sh \
#       --name contamlab-ll02 --price 1.99 --budget 20 --hard 18
#
# ★ 番号が 05 なのは、**10-bootstrap.sh より先に走らせる**からである。bootstrap 自体が
#   数十分の GPU 時間を使い、しかも止まることがある。見張りは最初に立てる。
#
# ★ 何をするか:
#     ① cost_watchdog.py arm  —— 対象を1台に決め、ハード期限を**絶対時刻で凍結**する
#     ② systemd のユニットとして常駐させる(無ければ setsid で切り離して起動)
#     ③ ★ **鼓動が出るまで待って、出なければ失敗する。**
#        ⛔ 「起動した」で終わらせない —— ラン lambda-ladder-01 の事故は
#           「立てたつもりの見張りが死んでいた」ことに気付けなかった事故である
#
# ⛔ 手元(Windows)から実行しても、cost_watchdog.py が拒む。それが仕様である。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=""
INSTANCE_ID=""
PRICE=""
BUDGET=""
HARD=""
INTERVAL=300
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)        NAME="$2"; shift 2 ;;
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --price)       PRICE="$2"; shift 2 ;;      # 1時間あたりの単価(USD)
    --budget)      BUDGET="$2"; shift 2 ;;     # 事前登録の停止条件(USD)
    --hard)        HARD="$2"; shift 2 ;;       # ここで切る(USD)。budget より小さいこと
    --interval)    INTERVAL="$2"; shift 2 ;;
    --dry-run)     EXTRA+=(--dry-run); shift ;;
    --started-at-utc) EXTRA+=(--started-at-utc "$2"); shift 2 ;;
    *) echo "不明な引数: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$PRICE" ]] || { echo "--price が要る(例: A100 40GB SXM4 なら 1.99)" >&2; exit 1; }
[[ -n "$HARD"  ]] || { echo "--hard が要る(この額で切る)" >&2; exit 1; }
[[ -n "$NAME" || -n "$INSTANCE_ID" ]] || { echo "--name か --instance-id が要る" >&2; exit 1; }

WATCHDOG="$REPO_ROOT/scripts/cost_watchdog.py"
STATE="$REPO_ROOT/reports/cost-watchdog.json"
LOG="$REPO_ROOT/reports/cost-watchdog.log"

banner "① 発火の経路を確かめる(課金ゼロ・terminate ゼロ)"
"$PY" "$WATCHDOG" selftest

banner "② 対象を決めてハード期限を凍結する"
ARM_ARGS=(arm --price-usd-per-hour "$PRICE" --hard-usd "$HARD" --interval "$INTERVAL")
[[ -n "$NAME"        ]] && ARM_ARGS+=(--name "$NAME")
[[ -n "$INSTANCE_ID" ]] && ARM_ARGS+=(--instance-id "$INSTANCE_ID")
[[ -n "$BUDGET"      ]] && ARM_ARGS+=(--budget-usd "$BUDGET")
"$PY" "$WATCHDOG" "${ARM_ARGS[@]}" "${EXTRA[@]}"

banner "③ 常駐させる"
if has_systemd; then
  # ★ API キーは EnvironmentFile で渡す。ユニットファイルに直接書くと
  #   /etc/systemd/system が世界可読なのでキーが漏れる。
  if [[ ! -s "$REPO_ROOT/.env" ]]; then
    echo "⛔ .env が無い。LAMBDA_API_KEY=... を書いてから実行すること。" >&2
    exit 1
  fi
  chmod 600 "$REPO_ROOT/.env"
  {
    echo "[Unit]"
    echo "Description=contamlab cost watchdog (hard deadline terminate)"
    echo "After=network-online.target"
    echo ""
    echo "[Service]"
    echo "Type=simple"
    echo "User=$(id -un)"
    echo "WorkingDirectory=$REPO_ROOT"
    echo "EnvironmentFile=$REPO_ROOT/.env"
    echo "ExecStart=$(command -v "$PY") $WATCHDOG run"
    # ★ 落ちたら必ず上げ直す。期限は状態ファイルに絶対時刻で入っているので、
    #   再起動しても寿命は伸びない(cost_watchdog.py の「状態」の節)。
    echo "Restart=always"
    echo "RestartSec=15"
    echo ""
    echo "[Install]"
    echo "WantedBy=multi-user.target"
  } | $SUDO tee /etc/systemd/system/contamlab-watchdog.service >/dev/null
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable contamlab-watchdog   # ★ 再起動しても復活する
  $SUDO systemctl restart contamlab-watchdog
  echo "systemd ユニット contamlab-watchdog を起動した。"
else
  # 貸しコンテナ型など。setsid で端末から切り離す —— ⛔ ssh が切れても死なせない。
  pkill -f "cost_watchdog.py run" || true
  setsid nohup "$PY" "$WATCHDOG" run >> "$LOG" 2>&1 &
  echo "systemd が無い。setsid で起動した(PID $!)。"
fi

banner "④ ★ 鼓動が出るまで待つ(出なければ失敗にする)"
# 立てたつもりで死んでいる、を許さない。20 秒で1回も鼓動しなければ設置は失敗である。
for _ in $(seq 1 20); do
  if "$PY" "$WATCHDOG" status >/dev/null 2>&1; then
    "$PY" "$WATCHDOG" status
    echo ""
    echo "★ 撤収時は disarm してから terminate すること:"
    echo "    python3 scripts/cost_watchdog.py disarm"
    echo "★ ログ: $LOG   状態: $STATE"
    exit 0
  fi
  sleep 1
done

echo "" >&2
echo "⛔ 鼓動が出ない。ウォッチドッグは設置できていない。" >&2
echo "   ログを見ること: $LOG" >&2
if has_systemd; then
  echo "   systemctl status contamlab-watchdog --no-pager" >&2
fi
echo "⛔ この状態で測定に進まないこと(見張りの無いまま GPU を回すことになる)。" >&2
exit 1
