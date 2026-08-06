#!/usr/bin/env bash
# scripts/lib.sh — 全スクリプト共通の定数と関数。単体では実行しない。
#
# ここに置いてよいのは「実験の設定ではなく、実験を回す道具の設定」だけである。
# 効果量・α・分割・摂動器といった**測定条件は preregister.md が正**であり、
# ここに書き写した値は必ず preregister の該当行を引用すること。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# ロースター(preregister.md「ラン: jmmlu-shuffle-02」の対象モデル表)
# ---------------------------------------------------------------------------
# NAME は**応答キャッシュのキーに入る**(runner.py:180 が model_name を使う)。
# 一度決めたら二度と変えない。変えるとパイロットのキャッシュが本番で効かなくなる。
#
#   NAME|OLLAMA_ALIAS|HF_REPO_TAG
ROSTER=(
  "llmjp3-13b|llmjp3-13b|hf.co/mmnga/llm-jp-3-13b-instruct3-gguf:Q4_K_M"
  "swallow31-8b|swallow31-8b|hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:Q4_K_M"
)

OLLAMA_BASE_URL="http://localhost:11434/v1"

# ---------------------------------------------------------------------------
# ベンチマークの pin(data/jmmlu.manifest.json と一致していなければならない)
# ---------------------------------------------------------------------------
JMMLU_COMMIT="762cbf192c9e4588b574d718f95afd64c96fcbf4"
JMMLU_EXPECTED_ACCEPTED=6664
JMMLU_EXPECTED_DEV=4742
JMMLU_EXPECTED_HOLDOUT=1922

BENCHMARK="data/jmmlu.jsonl"
DEV_SEED="dev-seed"

# ---------------------------------------------------------------------------
# 環境タグ — キャッシュを環境ごとに分けるための識別子
# ---------------------------------------------------------------------------
# CLAUDE.md「応答キャッシュに環境が入っていない」への対応。キャッシュのキーは
# モデル名とプロンプトだけなので、**バックエンドが変わったらファイルを分ける**
# 以外にキャッシュ混入を防ぐ手段が無い。
#
# reports/ は .gitignore 済みなので、タグの置き場所として .gitignore を触らずに済む。
ENV_TAG_FILE="reports/env-tag"

# ★ これは **サブシェルの外**から呼ぶこと。
#   env_tag は $(...) の中で使われるので、そこで exit しても死ぬのは副シェルだけで、
#   呼び出し元は "data/cache/responses..jsonl" という**もっともらしい別ファイル**を
#   受け取ったまま先へ進んでしまう。環境と結び付かないキャッシュに書き込むのは、
#   このプロジェクトが最も嫌う「静かに間違う」型の失敗である。
require_env_tag() {
  if [[ ! -s "$ENV_TAG_FILE" ]]; then
    echo "環境タグが無い: $ENV_TAG_FILE" >&2
    echo "先に scripts/30-record-environment.sh を実行すること。" >&2
    exit 1
  fi
}

env_tag() {
  require_env_tag
  tr -d '[:space:]' < "$ENV_TAG_FILE"
}

cache_path() { echo "data/cache/responses.$(env_tag).jsonl"; }

# 決定性測定用の**捨てキャッシュ**。本番キャッシュと混ぜない。
# 混ぜると2回目の実行が1回目の答えを再生してしまい(runner.py:223 の短絡)、
# 「不一致 0 件」が自明に出る。測定として無意味になる。
det_cache_path() { echo "data/cache/responses.$(env_tag).determinism-b.jsonl"; }

model_flags() {
  local entry name alias_unused spec
  for entry in "${ROSTER[@]}"; do
    IFS='|' read -r name alias_unused _ <<< "$entry"
    printf -- "--model compat:%s:%s:%s " "$name" "$name" "$OLLAMA_BASE_URL"
  done
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "見つからない: $1" >&2; exit 1; }
}

# ---------------------------------------------------------------------------
# 特権と init —— 借りる先によって違う(EC2 だけを想定しない)
# ---------------------------------------------------------------------------
# EC2 / Lambda Labs 等の「素の VM」は ubuntu ユーザ + sudo + systemd。
# 一方、貸しコンテナ型(RunPod 等)は **root で sudo が無く、systemd も無い**。
# どちらでも同じ手順が通るように、ここで吸収する。
if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

# systemd が実際に PID 1 として動いているか。`systemctl` の有無では判定できない
# (コンテナにはバイナリだけ残っていることがある)。
has_systemd() { [[ -d /run/systemd/system ]] && command -v systemctl >/dev/null 2>&1; }

# systemd が無いホストで Ollama に渡した環境変数の控え。
# 30-record-environment.sh が来歴を書くときの読み元になる。
OLLAMA_ENV_FILE="reports/ollama-env"

# Ollama が生きているか。CLI ではなく API で見る(program.md の環境メモ)。
require_ollama() {
  curl -sf "http://localhost:11434/api/tags" >/dev/null \
    || { echo "Ollama が応答しない(http://localhost:11434)。" >&2; exit 1; }
}

# python の実体。ローカル Windows では `python` が別プロジェクトの venv を指すので
# `py` を使う運用だが、EC2(Linux)では python3 が正。
PY="${CONTAMLAB_PYTHON:-python3}"

banner() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
