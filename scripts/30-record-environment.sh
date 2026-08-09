#!/usr/bin/env bash
# scripts/30-record-environment.sh — 実行環境を記録し、環境タグを確定する。
#
#   bash scripts/30-record-environment.sh
#   CONTAMLAB_ENV_TAG=ec2-l4-20260805 bash scripts/30-record-environment.sh
#
# 出すもの:
#   reports/env-tag                     以後のキャッシュ名を決める識別子
#   reports/environment.<tag>.json      機械可読の来歴
#   reports/environment.<tag>.md        ★ preregister.md にそのまま貼れる節
#
# ★ なぜ記録が要るか ★
# 応答キャッシュのキーは**モデル名とプロンプトだけ**で、ハードウェアもバックエンドも
# 量子化も入っていない(runner.py:180)。つまり「この数字がどの環境で出たか」は
# キャッシュの中には残らない。**残す場所はここしかない。**

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

mkdir -p reports

# --- インスタンス種別(EC2 でなければ空。ローカルでも動くようにしておく) -------
imds() {
  local token
  token="$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)" || return 1
  curl -sf -H "X-aws-ec2-metadata-token: $token" \
    "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null
}
INSTANCE_TYPE="$(imds instance-type || echo "")"
AVAILABILITY_ZONE="$(imds placement/availability-zone || echo "")"
# Spot かどうかは中断の可能性に直結するので記録する(キャッシュは追記専用なので再開できる)。
LIFECYCLE="$(imds instance-life-cycle || echo "")"

TAG="${CONTAMLAB_ENV_TAG:-}"
if [[ -z "$TAG" ]]; then
  if [[ -n "$INSTANCE_TYPE" ]]; then
    TAG="ec2-${INSTANCE_TYPE}-$(date -u +%Y%m%d)"
  else
    # ★ EC2 の外(借りた GPU ホスト・手元の機械)。**"ec2-" を名乗らせない。**
    #   環境タグはキャッシュ名であると同時に来歴の識別子であり、これがそのまま
    #   preregister の「実行環境」に載る。嘘のタグは嘘の事前確約になる。
    gpu_short="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
      | head -1 | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
    TAG="host-${gpu_short:-$(uname -m)}-$(date -u +%Y%m%d)"
    echo "★ EC2 の外で走っている。タグを '$TAG' にした。" >&2
    echo "  提供者を含めたいなら CONTAMLAB_ENV_TAG で明示する(例: lambda-a10-20260806)。" >&2
  fi
  TAG="${TAG//[^A-Za-z0-9._-]/-}"
fi

if [[ -f "$ENV_TAG_FILE" ]]; then
  existing="$(tr -d '[:space:]' < "$ENV_TAG_FILE")"
  if [[ "$existing" != "$TAG" ]]; then
    echo "★ 既存の環境タグ '$existing' と違うタグ '$TAG' を作ろうとしている。" >&2
    echo "  タグを変えるとキャッシュが別ファイルになり、**取得済みの応答が使われなくなる。**" >&2
    echo "  同じ機械で作業を続けるなら既存タグのままにすること。" >&2
    echo "  意図的に環境を変えたのなら CONTAMLAB_ENV_TAG で明示的に上書きすること。" >&2
    [[ -n "${CONTAMLAB_ENV_TAG:-}" ]] || exit 1
  fi
fi
printf '%s\n' "$TAG" > "$ENV_TAG_FILE"
banner "環境タグ = $TAG"
echo "応答キャッシュ: $(cache_path)"

require_ollama
OLLAMA_VER="$(ollama --version 2>&1 | head -1 | tr -d '\r')"
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")"
CUDA_VER="$(nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 && nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1 || echo "")"
# systemd が無いホストでは 10-bootstrap.sh が控えを残している。
# **どちらの経路でも来歴が空にならないようにする**(空欄のまま preregister に貼ると、
# 「並列度を固定した」という主張の裏付けが消える)。
if has_systemd; then
  OLLAMA_ENV="$(systemctl show ollama -p Environment --value 2>/dev/null || echo "")"
else
  OLLAMA_ENV="$(tr '\n' ' ' < "$OLLAMA_ENV_FILE" 2>/dev/null || echo "")"
fi
if [[ -z "${OLLAMA_ENV// /}" ]]; then
  echo "★ Ollama の環境変数を復元できなかった。10-bootstrap.sh を先に走らせること。" >&2
fi
OLLAMA_MODELS_DIR="$(sed -n 's/.*OLLAMA_MODELS=\([^ ]*\).*/\1/p' <<< "$OLLAMA_ENV")"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/usr/share/ollama/.ollama/models}"

banner "GGUF の SHA256"
# Ollama の manifest に載っている model レイヤの digest は、**GGUF ファイルそのものの
# sha256** である。これを固定すれば「同じ重みを読んだ」ことが後から検証できる。
declare -A GGUF_SHA=()
for entry in "${ROSTER[@]}"; do
  IFS='|' read -r name alias repo <<< "$entry"
  # hf.co/mmnga/foo-gguf:Q4_K_M → manifests/hf.co/mmnga/foo-gguf/Q4_K_M
  mpath="$OLLAMA_MODELS_DIR/manifests/${repo%:*}/${repo##*:}"
  digest=""
  if [[ -r "$mpath" ]] || sudo test -r "$mpath" 2>/dev/null; then
    digest="$( (cat "$mpath" 2>/dev/null || sudo cat "$mpath") | $PY -c '
import sys, json
layers = json.load(sys.stdin).get("layers", [])
for layer in layers:
    if layer.get("mediaType", "").endswith(".model"):
        print(layer["digest"].split(":", 1)[-1]); break
' 2>/dev/null || true)"
  fi
  if [[ -z "$digest" ]]; then
    echo "  ★ $name: SHA256 を読めなかった($mpath)。手で確認して記録すること。" >&2
    digest="UNKNOWN"
  else
    printf '  %-16s %s\n' "$name" "$digest"
  fi
  GGUF_SHA["$name"]="$digest"
done

banner "ベンチマークの同一性(生の sha256)"
# ★ 2026-08-09、ラン positive-control-03 で新設した欄。
#   preregister「ラン: positive-control-03」→「実行環境」が正。
#
#   ★ なぜ manifest 照合だけでは足りないのか ★
#   pc-02 の実行中に判明した —— **manifest が完全一致してもバイトは一致しないことがある。**
#   `$BENCHMARK` は .gitignore 済みで各環境で作り直すため、作った OS によって
#   問題文中の改行が `\r\n`(Windows)と `\n`(Linux)に割れる(1,801 箇所)。
#   **件数も id も一致するので、20-rebuild-benchmark.sh の照合はこの差を検出しない。**
#   影響は「応答キャッシュが OS 間で移植できない」「再現が静かにずれる」ことである。
#   **静かに間違う型の失敗なので、生の sha256 を来歴に残す。**
BENCH_SHA=""; BENCH_LINES=""; BENCH_CRLF=""
if [[ -r "$BENCHMARK" ]]; then
  BENCH_SHA="$(sha256sum "$BENCHMARK" | cut -d' ' -f1)"
  BENCH_LINES="$(wc -l < "$BENCHMARK" | tr -d '[:space:]')"
  # JSON 文字列の中に埋まった改行(`\r\n` の 4 バイト)を数える。ここが割れる。
  BENCH_CRLF="$($PY -c '
import sys
data = open(sys.argv[1], "rb").read()
print(data.count(rb"\r\n") + data.count(b"\x0d\x0a"))
' "$BENCHMARK")"
  printf '  %-16s %s\n' "sha256" "$BENCH_SHA"
  printf '  %-16s %s 行 / CRLF %s 箇所\n' "内訳" "$BENCH_LINES" "$BENCH_CRLF"
  if [[ "$BENCH_SHA" != "$JMMLU_MEASURED_SHA256_PREFIX"* ]]; then
    echo "  ★ pc-01 / pc-02 が測った版(${JMMLU_MEASURED_SHA256_PREFIX}…)と違う。" >&2
    echo "    CRLF が ${BENCH_CRLF} 箇所あるなら Windows で作った版である。" >&2
    echo "    preregister pc-03 の停止条件に該当しうる。**先に確かめてから測ること。**" >&2
  fi
  # 完全一致を機械的に強制したいときだけ使う(既定では警告に留める)。
  if [[ -n "${CONTAMLAB_EXPECT_BENCHMARK_SHA256:-}" \
        && "$BENCH_SHA" != "$CONTAMLAB_EXPECT_BENCHMARK_SHA256" ]]; then
    echo "★ ベンチマークの sha256 が指定と違う。測定に進んではいけない。" >&2
    echo "    実測 $BENCH_SHA" >&2
    echo "    指定 $CONTAMLAB_EXPECT_BENCHMARK_SHA256" >&2
    exit 1
  fi
else
  echo "  ★ $BENCHMARK が無い。先に 20-rebuild-benchmark.sh を走らせること。" >&2
fi

banner "書き出し"
JSON_OUT="reports/environment.$TAG.json"
MD_OUT="reports/environment.$TAG.md"

{
  printf '{\n'
  printf '  "env_tag": "%s",\n' "$TAG"
  printf '  "recorded_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "instance_type": "%s",\n' "$INSTANCE_TYPE"
  printf '  "availability_zone": "%s",\n' "$AVAILABILITY_ZONE"
  printf '  "instance_life_cycle": "%s",\n' "$LIFECYCLE"
  printf '  "gpu": "%s",\n' "$GPU_INFO"
  printf '  "cuda": "%s",\n' "$CUDA_VER"
  printf '  "os": "%s",\n' "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
  printf '  "kernel": "%s",\n' "$(uname -r)"
  printf '  "python": "%s",\n' "$($PY --version 2>&1)"
  printf '  "ollama_version": "%s",\n' "$OLLAMA_VER"
  printf '  "ollama_env": "%s",\n' "$(tr '\n' ' ' <<< "$OLLAMA_ENV" | sed 's/"/\\"/g')"
  printf '  "contamlab_commit": "%s",\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '  "contamlab_dirty": %s,\n' "$([[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo true || echo false)"
  printf '  "cache_path": "%s",\n' "$(cache_path)"
  printf '  "benchmark_path": "%s",\n' "$BENCHMARK"
  printf '  "benchmark_sha256": "%s",\n' "$BENCH_SHA"
  printf '  "benchmark_lines": "%s",\n' "$BENCH_LINES"
  printf '  "benchmark_crlf_count": "%s",\n' "$BENCH_CRLF"
  printf '  "models": [\n'
  first=1
  for entry in "${ROSTER[@]}"; do
    IFS='|' read -r name alias repo <<< "$entry"
    [[ $first -eq 1 ]] || printf ',\n'
    first=0
    printf '    {"name": "%s", "source": "%s", "gguf_sha256": "%s"}' \
      "$name" "$repo" "${GGUF_SHA[$name]}"
  done
  printf '\n  ]\n}\n'
} > "$JSON_OUT"

{
  echo "#### 実行環境($TAG・$(date -u +%Y-%m-%d) 記録)"
  echo
  echo "| 項目 | 値 |"
  echo "|---|---|"
  echo "| インスタンス | \`${INSTANCE_TYPE:-(EC2 外)}\` / ${AVAILABILITY_ZONE:-—} / ${LIFECYCLE:-—} |"
  echo "| GPU | ${GPU_INFO:-なし} |"
  echo "| CUDA | ${CUDA_VER:-—} |"
  echo "| OS / カーネル | $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") / $(uname -r) |"
  echo "| Ollama | \`$OLLAMA_VER\` |"
  echo "| Ollama の環境変数 | \`$(tr '\n' ' ' <<< "$OLLAMA_ENV")\` |"
  echo "| contamlab | \`$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\` |"
  echo "| 応答キャッシュ | \`$(cache_path)\`(**この環境専用。CPU 時代のものは持ち込まない**) |"
  echo "| ベンチマークの生 sha256 | \`${BENCH_SHA:-—}\` / ${BENCH_LINES:-—} 行 / CRLF ${BENCH_CRLF:-—} 箇所$( \
       [[ -n "$BENCH_SHA" && "$BENCH_SHA" != "$JMMLU_MEASURED_SHA256_PREFIX"* ]] \
       && echo " ← **★ pc-01 / pc-02 が測った版(${JMMLU_MEASURED_SHA256_PREFIX}…)と違う**" )|"
  echo
  echo "| モデル | 取得元 | GGUF の SHA256 |"
  echo "|---|---|---|"
  for entry in "${ROSTER[@]}"; do
    IFS='|' read -r name alias repo <<< "$entry"
    echo "| \`$name\` | \`$repo\` | \`${GGUF_SHA[$name]}\` |"
  done
  echo
  echo "> \`OLLAMA_NUM_PARALLEL=1\` は決定性のための設定である。並列実行はリクエストを"
  echo "> バッチにまとめるので、バッチの組み方で浮動小数の加算順序が変わりうる。"
  echo "> \`temperature 0\` は「最も確率の高い選択肢を選ぶ」であって「計算結果が同じになる」"
  echo "> ではない。決定性は宣言ではなく **scripts/50-check-determinism.sh の実測**で示す。"
} > "$MD_OUT"

echo "  $JSON_OUT"
echo "  $MD_OUT   ← ★ この中身を preregister.md の**今のラン**の「実行環境」節に貼る"
