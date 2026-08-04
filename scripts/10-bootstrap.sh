#!/usr/bin/env bash
# scripts/10-bootstrap.sh — EC2 インスタンス側で最初に1回だけ走らせる。
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
  echo "nvidia-smi が無い。ドライバを入れる(DLAMI を使えばここは飛ばせる)。"
  sudo apt-get update -qq
  sudo apt-get install -y -qq ubuntu-drivers-common
  sudo ubuntu-drivers install
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
sudo apt-get update -qq
sudo apt-get install -y -qq git curl python3

banner "2. Ollama"
if command -v ollama >/dev/null 2>&1 && [[ -z "${OLLAMA_VERSION:-}" ]]; then
  echo "既にある: $(ollama --version 2>&1 | head -1)"
elif [[ -n "${OLLAMA_VERSION:-}" ]]; then
  # 版を固定して入れる。**AMI 化して再利用するときはこちらを使う。**
  echo "版を固定して導入: ${OLLAMA_VERSION}"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/ollama.tgz" \
    "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz"
  sudo tar -C /usr -xzf "$tmp/ollama.tgz"
  rm -rf "$tmp"
  sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama 2>/dev/null || true
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
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/contamlab.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_MODELS=/opt/ollama/models"
EOF
sudo mkdir -p /opt/ollama/models
sudo chown -R ollama:ollama /opt/ollama 2>/dev/null || sudo chown -R "$USER" /opt/ollama
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sleep 3
require_ollama
echo "Ollama 応答あり。"

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
