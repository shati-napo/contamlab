#!/usr/bin/env bash
# finetune/to_gguf.sh — マージ済み HF 重みを GGUF Q4_K_M にして Ollama に載せる。
#
#   bash finetune/to_gguf.sh pc-x40
#
# preregister「量子化 —— 測定条件からの逸脱を先に申告する」の実装。
# 測定条件は「Q4_K_M・提供元 mmnga に統一」だが自作モデルに mmnga 版は無い。
# **6アームすべてが同じこのパイプラインを通る**ので、ロースター内は完全に揃う。
# 記録すべきは llama.cpp の commit と各アームの GGUF の sha256(下で出す)。

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARM="${1:-}"
[[ -n "$ARM" ]] || { echo "使い方: bash finetune/to_gguf.sh pc-x40" >&2; exit 1; }
[[ -d "models/$ARM" ]] || { echo "モデルが無い: models/$ARM(先に train_lora.py)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# ★ pc-01 で詰まった 2 点を、走り出す前に潰す(preregister pc-02「実装」)。
# ---------------------------------------------------------------------------
# (1) python の実体を venv に固定する。
#     素の `python` は transformers の入っていない system python を指すことがあり、
#     pc-01 では変換の途中で ModuleNotFoundError で落ちた。**変換は数分かかるので、
#     落ちるなら最初の1秒で落ちるべきである。**
PY="${CONTAMLAB_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "finetune/.venv/bin/python" ]]; then
    PY="finetune/.venv/bin/python"          # finetune/README.md:22 の作法
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
  else
    echo "★ venv の python が見つからない: finetune/.venv/bin/python" >&2
    echo "  finetune/README.md:22 の手順で venv を作るか、CONTAMLAB_PYTHON で指すこと。" >&2
    exit 1
  fi
fi
"$PY" -c 'import transformers, torch' 2>/dev/null || {
  echo "★ $PY に transformers / torch が無い。変換は途中で落ちる。" >&2
  echo "  pip install -r finetune/requirements.txt を先に。" >&2
  exit 1; }

# (2) Ollama が起動しているか。**変換の前に**確かめる。
#     学習中は VRAM のため Ollama を止める運用なので、止めたまま変換に入りやすい。
#     pc-01 では f16→Q4 の数分を費やした最後の `ollama create` で落ちた。
require_cmd_ollama() { command -v ollama >/dev/null 2>&1; }
require_cmd_ollama || { echo "★ ollama が見つからない。" >&2; exit 1; }
curl -sf "http://localhost:11434/api/tags" >/dev/null || {
  echo "★ Ollama が応答しない(http://localhost:11434)。" >&2
  echo "  最後の登録で落ちるので、変換に入らない。先に Ollama を起動すること。" >&2
  echo "  (学習中は VRAM のために止める運用なので、止めたままになりやすい)" >&2
  exit 1; }

echo "python : $PY"
echo "Ollama : 応答あり"

# ★ pin。リリースタグなので CI を通った状態である(master の HEAD は使わない)。
LLAMA_TAG="b10327"
LLAMA_COMMIT="69bf6437914596fbbc4caf09a7ac16f2acdd1a94"
LLAMA_DIR="data/raw/llama.cpp"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --quiet https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --quiet --tags origin
git -C "$LLAMA_DIR" checkout --quiet --detach "$LLAMA_TAG"
actual="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
[[ "$actual" == "$LLAMA_COMMIT" ]] || {
  echo "★ llama.cpp の commit が pin と違う: $actual != $LLAMA_COMMIT" >&2; exit 1; }

if [[ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
  echo "llama.cpp をビルド中(初回のみ)"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$LLAMA_DIR/build" --target llama-quantize -j"$(nproc)" >/dev/null
fi

mkdir -p models/gguf
F16="models/gguf/$ARM.f16.gguf"
Q4="models/gguf/$ARM.Q4_K_M.gguf"

echo "1. HF → GGUF f16"
"$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "models/$ARM" --outfile "$F16" --outtype f16

echo "2. f16 → Q4_K_M"
"$LLAMA_DIR/build/bin/llama-quantize" "$F16" "$Q4" Q4_K_M

echo "3. Ollama に登録"
printf 'FROM %s\nPARAMETER temperature 0\n' "$(realpath "$Q4")" > "models/gguf/$ARM.Modelfile"
ollama create "$ARM" -f "models/gguf/$ARM.Modelfile"

SHA="$(sha256sum "$Q4" | cut -d' ' -f1)"
echo "$ARM  $SHA" >> reports/gguf-sha256.txt
echo
echo "★ preregister の「6アームの GGUF SHA256」に転記する値:"
echo "   $ARM = $SHA"
rm -f "$F16"   # f16 は中間物。Q4 が正
