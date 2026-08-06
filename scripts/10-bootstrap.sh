#!/usr/bin/env bash
# scripts/10-bootstrap.sh — GPU ホスト側で最初に1回だけ走らせる。
#
#   bash scripts/10-bootstrap.sh
#   OLLAMA_VERSION=v0.x.y bash scripts/10-bootstrap.sh   # 版を固定して再現する
#
# やること: NVIDIA ドライバの確認 → Ollama の導入 → **決定性のための環境変数を固定**
# → ロースター2本の GGUF 取得 → コロン無しの別名付与。
#
# ★ このスクリプトは Bedrock を含む一切のマネージド推論 API に触れない。
#   ここで用意するのは「自分のプロセスの中で GGUF を読む」実行系だけである
#   (CLAUDE.md「絶対禁止: Amazon Bedrock」)。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

banner "0. GPU の確認"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi が無い。ドライバを入れる(DLAMI や GPU 貸し出し業者のイメージなら飛ばせる)。"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ubuntu-drivers-common
  $SUDO ubuntu-drivers install
  echo "★ ドライバを入れた。**再起動してからこのスクリプトを再実行すること。**"
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 13B の Q4_K_M が約 8.4GB。モデルは harness が逐次評価する(harness.py:170)ので
# ピークは最大の1本ぶん。L4(24GB)で足りる。
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if (( VRAM_MIB < 12000 )); then
  echo "★ VRAM ${VRAM_MIB}MiB は 13B Q4_K_M(約8.4GB)に対して余裕が無い。" >&2
  echo "  インスタンス種別を見直すこと。" >&2
  exit 1
fi

banner "1. 依存(OS 側の道具だけ)"
# contamlab は**標準ライブラリのみ**なので pip install も venv も要らない。
# リポジトリのルートから `python3 -m contamlab` を叩けば動く。
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git curl python3

banner "2. Ollama"
if command -v ollama >/dev/null 2>&1 && [[ -z "${OLLAMA_VERSION:-}" ]]; then
  echo "既にある: $(ollama --version 2>&1 | head -1)"
elif [[ -n "${OLLAMA_VERSION:-}" ]]; then
  # 版を固定して入れる。**AMI 化して再利用するときはこちらを使う。**
  echo "版を固定して導入: ${OLLAMA_VERSION}"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/ollama.tgz" \
    "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz"
  $SUDO tar -C /usr -xzf "$tmp/ollama.tgz"
  rm -rf "$tmp"
  $SUDO useradd -r -s /bin/false -m -d /usr/share/ollama ollama 2>/dev/null || true
else
  curl -fsSL https://ollama.com/install.sh | sh
  echo
  echo "★ 版を指定せずに入れた。**解決された版を preregister.md に記録し、"
  echo "  以後は OLLAMA_VERSION で固定すること**(再現性は版の固定で担保される)。"
fi

banner "3. 決定性と秘匿のための環境変数を固定"
# OLLAMA_NUM_PARALLEL=1 が本命。並列実行はリクエストをバッチにまとめるので、
# **バッチの組み方で浮動小数の加算順序が変わりうる。** temperature 0 は
# 「最も確率の高い選択肢を選ぶ」であって「計算結果が同じになる」ではない
# (CLAUDE.md「temperature 0 は決定性を保証しない」)。contamlab は逐次に問い合わせる
# ので、並列度を1に落としても速度は落ちない。
#
# OLLAMA_HOST=127.0.0.1 も必須。0.0.0.0 で待つと HOLDOUT の問題文を投げる口が
# インスタンスの外に開く。**問題文が自分の管理下のプロセスの外に出るか**が判定の一行。
OLLAMA_ENV_KV=(
  "OLLAMA_HOST=127.0.0.1:11434"
  "OLLAMA_NUM_PARALLEL=1"
  "OLLAMA_MAX_LOADED_MODELS=1"
  "OLLAMA_KEEP_ALIVE=30m"
  "OLLAMA_MODELS=/opt/ollama/models"
)

$SUDO mkdir -p /opt/ollama/models
$SUDO chown -R ollama:ollama /opt/ollama 2>/dev/null || $SUDO chown -R "$(id -un)" /opt/ollama

# 記録は systemd の有無に関わらず残す。30-record-environment.sh は systemd が無ければ
# こちらを読む(**来歴が環境によって欠けることを許さない**)。
mkdir -p reports
printf '%s\n' "${OLLAMA_ENV_KV[@]}" > "$OLLAMA_ENV_FILE"

if has_systemd; then
  $SUDO mkdir -p /etc/systemd/system/ollama.service.d
  {
    echo "[Service]"
    for kv in "${OLLAMA_ENV_KV[@]}"; do printf 'Environment="%s"\n' "$kv"; done
  } | $SUDO tee /etc/systemd/system/ollama.service.d/contamlab.conf >/dev/null
  $SUDO systemctl daemon-reload
  # ★ `enable --now` では足りない(2026-08-06 実機で判明)。--now は「停止していれば
  #   起動する」であって、**既に走っているユニットは再起動しない。** Ollama の
  #   install.sh は導入時にサービスを起動してしまうので、drop-in を書いた直後の
  #   `enable --now` は何もせず、プロセスは古い環境のまま生き続ける。
  #   下の /proc/<pid>/environ 照合はまさにこれを捕まえた。**restart は省略できない。**
  $SUDO systemctl enable ollama
  $SUDO systemctl restart ollama
else
  # systemd が無いホスト(貸しコンテナ型など)。同じ環境変数を渡して直に起動する。
  # ★ 環境変数を渡さずに起動すると OLLAMA_NUM_PARALLEL の既定は 1 ではないので、
  #   **決定性の前提が静かに崩れる。** ここは省略できない。
  echo "systemd が無い。ollama serve を直接起動する。"
  pkill -f 'ollama serve' 2>/dev/null || true
  env "${OLLAMA_ENV_KV[@]}" nohup ollama serve > /tmp/ollama.log 2>&1 &
fi
sleep 3
require_ollama
echo "Ollama 応答あり。"

# ★ 設定を書いただけでは「効いている」ことにならない —— daemon-reload の忘れや、
#   前から生き残っていた ollama serve が居座っている場合、ファイルは正しいのに
#   **プロセスは古い環境で動いている。** Ollama は API で並列度を返さないので、
#   実際に走っているプロセスの environ を読んで確かめる。
serve_pid="$(pgrep -f 'ollama serve' | head -1 || true)"
if [[ -n "$serve_pid" ]] && { [[ -r "/proc/$serve_pid/environ" ]] || $SUDO test -r "/proc/$serve_pid/environ"; }; then
  live_env="$( ($SUDO cat "/proc/$serve_pid/environ" 2>/dev/null || cat "/proc/$serve_pid/environ") | tr '\0' '\n')"
  for kv in "${OLLAMA_ENV_KV[@]}"; do
    if grep -qxF "$kv" <<< "$live_env"; then
      printf '  ✓ %s\n' "$kv"
    else
      echo "  ★ 反映されていない: $kv(pid $serve_pid)" >&2
      echo "    古い ollama serve が生き残っている可能性がある。止めてから再実行すること。" >&2
      exit 1
    fi
  done
else
  echo "  ★ ollama serve の環境を読めなかった。OLLAMA_NUM_PARALLEL=1 を手で確認すること。" >&2
fi

banner "4. GGUF の取得(mmnga / Q4_K_M 統一を維持)"
for entry in "${ROSTER[@]}"; do
  IFS='|' read -r name alias repo <<< "$entry"
  echo "--- $name ← $repo"
  ollama pull "$repo"
  # Ollama のモデル名のコロンが compat spec の解析を壊す(clients.py:344)ので
  # コロン無しの別名を付ける。clients.py は編集しない。
  ollama cp "$repo" "$alias"
done

banner "5. 疎通(モデルが実際に答えるか)"
for entry in "${ROSTER[@]}"; do
  IFS='|' read -r name alias _ <<< "$entry"
  reply="$(curl -sf "$OLLAMA_BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1は？数字だけ答えて。\"}],\"temperature\":0,\"max_tokens\":16}" \
    | $PY -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:40])')"
  printf '%-16s → %s\n' "$alias" "$reply"
done

banner "6. 測定装置の健全性チェック"
# program.md:「これが落ちたら実験を止める(測定装置が壊れていれば全実験が無価値)」。
# モデルを一切使わない合成データの検査なので、ここで落ちるなら環境の問題である。
$PY -m contamlab verify

banner "完了"
cat <<'EOF'
次:
  bash scripts/20-rebuild-benchmark.sh    JMMLU を pin した SHA から作り直して照合
  bash scripts/30-record-environment.sh   版と SHA256 を記録(preregister に貼る)
EOF
