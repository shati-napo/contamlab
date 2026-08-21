#!/usr/bin/env bash
# scripts/df1-orchestrate.sh — ラン detector-firstlight-01 を**インスタンスの上で**最後まで回す。
#
#   nohup bash scripts/df1-orchestrate.sh > reports/df1-orchestrate.log 2>&1 &
#
# ★ なぜこれが要るか(cc-01 の作法をそのまま引き継ぐ)
#   ll-01 の $47.82 の事故から「**課金を止める仕掛けはリモートに置く**」を学んで
#   cost_watchdog.py を作り、cc-01 で「**ランを進める仕掛け**もリモートに置く」に広げた。
#   手元の PC が落ちると学習は続くが**次の段が始まらず**、GPU は空回りしたまま
#   ハード期限まで課金される。**同じ形の穴である。**
#
# ★ 判定はすべて scripts/df1_gate.py が持つ(閾値はそちらに定数として書いてある)。
#   このスクリプトは**順序と分岐だけ**を担う。⛔ 数字を1つも持たない。
#
# ★ 冪等である —— 既に在るものは作り直さない(学習・マージ・GGUF・測定)。
#   途中で落ちても、もう一度起動すれば続きから進む。
#
# ⛔ a(差 >= 10pt)は関門ではない。`report-a` は印字するだけで、ここを止めない
#   (preregister「★ a を関門にしない」)。

set -uo pipefail
cd "$HOME/contamlab"
export PYTHONIOENCODING=utf-8
export LC_ALL="${LC_ALL:-C.UTF-8}"
TAG="$(cat reports/env-tag)"
PY_VENV=finetune/.venv/bin/python
BASE_ARM=pcbase-swallow31-8b-x00
DIRTY_ARM=df1L08t1-x40          # ★ 検出器に通す複製は t1 に**事前固定**

ts() { date -u +%FT%TZ; }

# ★ 段の境目ごとに成果物を**ホストの外**へ出す(2026-08-20 の $17.90 の損失から)。
#   前回は「作ったのに外に出せないまま terminate でディスクごと消えた」。
#   ⛔ 同期の失敗ではランを止めない —— 止めると GPU の空回りが増えるだけで、
#      失敗自体は reports/df1-sync.json と常駐側が拾う。
SYNC=scripts/df1_sync.py
sync_now() { timeout 300 python3 "$SYNC" once --why "$1" >> reports/df1-sync.log 2>&1 || true; }
say() { echo "=== $* $(ts)"; sync_now "$*"; }

# ★ 終わったら(正常でも異常でも)ハード期限を手前に寄せる。
#   ⛔ 期限を**後ろへ動かすことはしない**(緩和になる)。
# ★ 2026-08-21 に 6h から 1h へ縮めた。**理由: 猶予の目的が消えたため。**
#   6h は「人が起きてきて手で回収する」ための時間だった。いまは df1_sync.py が
#   段ごと・900秒ごとに外へ出しているので、終わった時点で成果物はもう手元にある。
#   ⛔ 前回はこの猶予のあいだ(19:30Z 完走 -> 21:37Z 期限)に $4 を空回りで払った。
GRACE_HOURS=1            # ★ 中を覗くための余白だけ残す。1h × $1.99 = $1.99
grace() {
  local cur_epoch new_epoch
  cur_epoch=$(python3 -c "import json;print(int(json.load(open('reports/cost-watchdog.json'))['deadline_epoch']))" 2>/dev/null || echo 0)
  new_epoch=$(( $(date -u +%s) + GRACE_HOURS * 3600 ))
  if [ "$cur_epoch" -gt 0 ] && [ "$new_epoch" -ge "$cur_epoch" ]; then
    say "[grace] いまの期限のほうが早い。1秒も後ろへ動かさない"
    return 0
  fi
  say "[grace] ハード期限を現在 +${GRACE_HOURS}h に寄せ直す(★ 厳しい側にしか動かさない)"
  # ★ 金額は GRACE_HOURS から出す。⛔ 直書きすると猶予を縮めたときに置き去りになる
  #   (2026-08-21: 6h -> 1h に縮めた。ここが 11.94 のままなら期限は 6h 先のままだった)
  python3 scripts/cost_watchdog.py arm --name contamlab-df1 \
      --price-usd-per-hour 1.99 \
      --hard-usd "$(python3 -c "print(round(1.99*${GRACE_HOURS},2))")" --budget-usd 20 \
      --started-at-utc "$(date -u +%FT%TZ)" --interval 300 || return 0
  sudo systemctl restart contamlab-watchdog || true
  python3 scripts/cost_watchdog.py status || true
}

stop_run() {
  echo "=== [STOP] $* $(ts)"
  { echo "$(ts)  $*"; } >> reports/df1-STOPPED.txt
  sync_now "[STOP] $*"      # ★ 落ちた理由と、そこまでの成果物を外へ出してから終わる
  grace
  exit 1
}

check_arms() {  # $1=out-file $2..=arms
  local out="$1"; shift
  say "[C] 操作チェック $* -> $out"
  sudo systemctl start ollama; sleep 3
  # ⛔ `|| exit 1` を付けない —— 65 は解釈不能率 5% 超で非ゼロ終了するが、
  #    本ランではそれが測定対象である(td-01・ll-01 で踏んだ穴)。判定は df1_gate.py が行う。
  bash scripts/65-manipulation-check.sh "$@" > "$out" 2>&1 || true
  cat "$out"
}

train_one() {   # $1=replicate
  local t="$1" arm="df1t${1}-x40" lam="df1L08t${1}-x40"
  if [ ! -f "models/${arm}/train.json" ]; then
    say "[T] 学習 ${arm}"
    sudo systemctl stop ollama || true
    $PY_VENV finetune/train_lora.py --run detector-firstlight-01 --recipe R1 \
        --replicate "$t" || stop_run "学習に失敗: ${arm}"
  else
    say "[T] 既にある(飛ばす) ${arm}"
  fi
  if [ ! -d "models/${lam}" ]; then
    say "[L] lambda=0.8 でマージ ${lam}"
    sudo systemctl stop ollama || true
    $PY_VENV finetune/scale_adapter.py --run detector-firstlight-01 \
        --lambda-step L1 --replicate "$t" || stop_run "マージに失敗: ${lam}"
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

say "[START] detector-firstlight-01 のオーケストレータ(env-tag ${TAG})"

exec 9>"reports/.df1-orchestrate.lock"
if ! flock -n 9; then
  say "[LOCK] 既にオーケストレータが走っている。何もしない"
  exit 0
fi

while pgrep -f "finetune/train_lora.py" > /dev/null; do
  say "[WAIT] 先に走っている学習の終了を待つ(120s)"
  sleep 120
done

# ---- 保全(★ 学習より先に通す)-----------------------------------------------
# ⛔ 外へ出せないまま GPU を回さない —— それが 2026-08-20 に成果物を失った形である。
#   ここは「読めるか」ではなく「**実際に1コミット push できるか**」を見る。
echo "=== [S] 成果物の保全: 置き先へ push できることを確かめる $(ts)"
python3 "$SYNC" init   || stop_run "保全の導通確認に失敗(⛔ 成果物を外に出せない状態で学習を始めない)"
# ★ 段の境目だけでは、検出器(2.4時間)の途中で落ちたときに失う。常駐も置く。
if ! pgrep -f "df1_sync.p[y] daemon" > /dev/null; then
  setsid nohup python3 "$SYNC" daemon --interval 900 >> reports/df1-sync.log 2>&1 < /dev/null &
  echo "=== [S] 常駐の同期を始めた(900秒ごと)$(ts)"
fi

# ---- 事前条件 --------------------------------------------------------------
# ★ ベンチマークと注入集合は .gitignore 対象(問題文そのものなので配らない)。
#   借りたホストでは作り直しになる。⛔ 中身は再生成だが、prepare_df1_arms.py が
#   pc-01 の manifest と sha256 で照合するので、1バイトでも違えば書き込みを残さず止まる。
[ -f data/jmmlu.jsonl ] || stop_run "data/jmmlu.jsonl が無い。先に 20-rebuild-benchmark.sh を通すこと"
[ -f reports/prompt-format ] || stop_run "reports/prompt-format が無い(書式 C を置くこと)"
if [ ! -f data/injection/pc-x40.ids ]; then
  say "[P] 注入集合を作り直す(pc-01 の規則。salt と pin から決まる決定論的な量)"
  python3 tools/build_injection_sets.py || stop_run "注入集合の生成に失敗"
fi
if [ ! -f data/injection/df1L08t1-x40.ids ]; then
  say "[P] 本ランのアームへ複製(sha256 を pc-01 の manifest と照合する)"
  python3 finetune/prepare_df1_arms.py || stop_run "注入集合の複製に失敗(sha256 が pc-01 と違う)"
fi
if [ ! -f reports/micro-batch ]; then
  # ★ probe は**学習と同じ VRAM の条件**で走らせる。Ollama が載ったままだと
  #   9GB ほど掴まれた状態で「載らない」と判定し、実効バッチの内訳が
  #   ll-01(micro 4 × grad_accum 4)と変わってしまう。⛔ 規則ではなく実測条件の問題である。
  say "[P] micro-batch を probe で決める(★ 人が決めない。Ollama を止めてから測る)"
  sudo systemctl stop ollama || true
  sleep 3
  $PY_VENV finetune/probe_micro_batch.py --run detector-firstlight-01 --recipe R1     || stop_run "probe に失敗"
fi

# ---- G0: 第0段(素のベース)-------------------------------------------------
# ★ probe の段で Ollama を止めているので、**先に起こしてから在庫を見る。**
#   止まったまま `ollama show` を撃つと「無い」と読み、作り直しに 6 分払う
#   (2026-08-20 に実際に払った。⛔ 規則ではなく手順の穴である)。
sudo systemctl start ollama || true
sleep 5
if ! ollama show "$BASE_ARM" >/dev/null 2>&1; then
  say "[B] ベースを書き出して GGUF にする"
  sudo systemctl stop ollama || true
  $PY_VENV finetune/export_base.py --candidate 1 || stop_run "ベースの書き出しに失敗"
  sudo systemctl start ollama; sleep 5
  bash finetune/to_gguf.sh "$BASE_ARM" || stop_run "ベースの GGUF 変換に失敗"
fi
check_arms "reports/df1-check-base.txt" "$BASE_ARM"
python3 scripts/df1_gate.py base --check reports/df1-check-base.txt \
  || stop_run "G0: 第0段の帯を外れた(ベースが pc-03 以来の帯の中にない)"

# ---- 学習3本と λ=0.8 --------------------------------------------------------
train_one 1
check_arms "reports/df1-check-t1.txt" "df1L08t1-x40"
python3 scripts/df1_gate.py anomaly --check reports/df1-check-t1.txt \
  || stop_run "G3: 1本目で解釈不能率が 50% を超えた(安価な異常検知)"

train_one 2
train_one 3

# ---- G1: 入力側の一致 -------------------------------------------------------
say "[I] 入力側の一致(G1)"
python3 scripts/df1_gate.py inputs --arms df1t1-x40 df1t2-x40 df1t3-x40 \
  || stop_run "G1: 3本で入力側が一致しない"

# ---- G2: 実効 λ -------------------------------------------------------------
say "[L] 実効 λ の相対誤差(G2)"
python3 scripts/df1_gate.py lambda --arms df1L08t1-x40 df1L08t2-x40 df1L08t3-x40 \
  || stop_run "G2: 実効 λ が凍結値から離れた"

# ---- G4・G5(関門)と a(報告のみ)-------------------------------------------
CHECK="reports/df1-check.txt"
check_arms "$CHECK" df1L08t1-x40 df1L08t2-x40 df1L08t3-x40

say "[A] 条件 a の読み(★ 報告のみ・⛔ ここでは止めない)"
python3 scripts/df1_gate.py report-a --check "$CHECK" || true

say "[G4] 条件 b(非注入群 正解率 >= 0.30)を k=3 で読む"
python3 scripts/df1_gate.py gate-b --check "$CHECK" \
  || stop_run "G4: b が k=3 で通らなかった(fine-tune がベースを壊している)"

say "[G5] 条件 c(解釈不能率 両群 <= 5%)を k=3 で読む"
python3 scripts/df1_gate.py gate-c --check "$CHECK" \
  || stop_run "G5: c が k=3 で通らなかった(指示追従が壊れている)"

# ---- 検出器の初撃 -----------------------------------------------------------
touch "reports/manipulation-check.${TAG}.ok"
say "[M] ★ 検出器(72-detector-firstlight.sh)—— 本プロジェクトで初めて実モデルに当てる"
printf 'y\n' | bash scripts/72-detector-firstlight.sh || stop_run "検出器の測定に失敗"

# ---- 副次の読み(★ 追加課金ゼロ・⛔ 報告のみ)-------------------------------
say "[S] 副次: 注入済み / 非注入に分けた drop(推論は1回も増えない)"
python3 tools/split_drop_by_injection.py --arms "$BASE_ARM" "$DIRTY_ARM" \
    --json "reports/df1-split-drop.${TAG}.json" || true

say "[DONE] 検出器を通した。結果: reports/detector-firstlight-01.${TAG}.json"
sync_now "[DONE] 最終"     # ★ 最後の1回。ここまで来たら何としても外へ出す
python3 "$SYNC" status || true
grace
