#!/usr/bin/env bash
# scripts/80-pf1-pull.sh — ラン perturbation-floor-01 の段 2(モデルの取得)。
#
#   bash scripts/80-pf1-pull.sh
#
# preregister「ラン: perturbation-floor-01」→「段と関門」の段 2 の実装。
#
# ★ **本ランは学習を1回もしない。**ここで引く 5 本は、いずれも
#   **本プロジェクトが1度も学習に使っていないモデル**である(選定基準 4)。
#
# ⛔ **10-bootstrap.sh は1行も触っていない。**あちらは既存ランの ROSTER(2本)を
#   引く段であり、本ランは PF1_ROSTER(5本)を別に引く。
#   既存ランの pull 対象・環境記録・キャッシュのキーは1つも動いていない。
#
# ★ 関門 G1: 5 本とも `ollama show` が通り、GGUF の digest が記録できること。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_ollama
require_env_tag

TAG="$(env_tag)"
OUT="reports/pf1-models.$TAG.json"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"

banner "1. ロースター(preregister の凍結表。⛔ ここで選び直さない)"
for entry in "${PF1_ROSTER[@]}"; do
  IFS='|' read -r name alias repo <<< "$entry"
  printf '  %-22s ← %s\n' "$name" "$repo"
done

banner "2. GGUF の取得(mmnga / Q4_K_M 統一を維持)"
# ⛔ 停止条件 3: mmnga 以外の提供元、または Q4_K_M 以外の量子化を引いたら止める。
for entry in "${PF1_ROSTER[@]}"; do
  IFS='|' read -r name alias repo <<< "$entry"
  case "$repo" in
    hf.co/mmnga/*:Q4_K_M) ;;
    *) echo "★ 停止条件 3 —— 提供元か量子化が凍結表から外れている: $repo" >&2; exit 1 ;;
  esac
  echo "--- $name ← $repo"
  ollama pull "$repo"
  # Ollama のモデル名のコロンが compat spec の解析を壊す(clients.py:344)ので
  # コロン無しの別名を付ける。clients.py は編集しない。
  ollama cp "$repo" "$alias"
done

banner "3. 疎通(モデルが実際に答えるか)"
for entry in "${PF1_ROSTER[@]}"; do
  IFS='|' read -r name alias _ <<< "$entry"
  reply="$(curl -sf "$OLLAMA_BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1は？数字だけ答えて。\"}],\"temperature\":0,\"max_tokens\":16}" \
    | $PY -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:40])')" || {
      echo "★ 関門 G1 —— $alias が応答しない。" >&2; exit 1; }
  printf '  %-22s → %s\n' "$alias" "$reply"
done

banner "4. GGUF の SHA256(★ 後から「同じ重みを読んだ」ことを検証するため)"
# Ollama の manifest に載っている model レイヤの digest は GGUF そのものの sha256。
# 30-record-environment.sh と同じ読み方をする(あちらは ROSTER、こちらは PF1_ROSTER)。
{
  echo "{"
  echo "  \"run\": \"perturbation-floor-01\","
  echo "  \"env_tag\": \"$TAG\","
  echo "  \"models\": ["
  first=1
  for entry in "${PF1_ROSTER[@]}"; do
    IFS='|' read -r name alias repo <<< "$entry"
    mpath="$OLLAMA_MODELS_DIR/manifests/${repo%:*}/${repo##*:}"
    digest=""
    if [[ -r "$mpath" ]] || $SUDO test -r "$mpath" 2>/dev/null; then
      digest="$( (cat "$mpath" 2>/dev/null || $SUDO cat "$mpath") | $PY -c '
import sys, json
layers = json.load(sys.stdin).get("layers", [])
for layer in layers:
    if layer.get("mediaType", "").endswith(".model"):
        print(layer["digest"].split(":", 1)[-1]); break
' 2>/dev/null || true)"
    fi
    [[ $first -eq 1 ]] || echo ","
    first=0
    printf '    {"name": "%s", "repo": "%s", "gguf_sha256": "%s"}' "$name" "$repo" "$digest"
  done
  echo
  echo "  ]"
  echo "}"
} > "$OUT"

$PY -c "
import json
d = json.load(open('$OUT', encoding='utf-8'))
missing = [m['name'] for m in d['models'] if not m['gguf_sha256']]
for m in d['models']:
    print(f\"  {m['name']:<22} {m['gguf_sha256'][:16] or '(読めなかった)'}\")
if missing:
    print()
    print('  ★ digest を読めなかったモデルがある: ' + ', '.join(missing))
    print('  ⛔ 記録は残るが、後から重みの同一性を検証できない。')
"

echo
echo "★ 記録: $OUT"
echo "★ 次は段 3(パイロット): bash scripts/81-pf1-pilot.sh"
