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
python "$LLAMA_DIR/convert_hf_to_gguf.py" "models/$ARM" --outfile "$F16" --outtype f16

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
