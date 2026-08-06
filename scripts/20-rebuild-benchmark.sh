#!/usr/bin/env bash
# scripts/20-rebuild-benchmark.sh — JMMLU を pin した commit から作り直し、
# **コミット済みの manifest と一致するか照合する。**
#
#   bash scripts/20-rebuild-benchmark.sh
#
# ★ このスクリプトが存在する理由 ★
#
# 問題インスタンス(data/jmmlu.jsonl)は .gitignore されているので、新しい機械では
# 作り直すしかない。ところが tools/ingest_jmmlu.py の ensure_clone は
# **`git clone` を HEAD に対して行う**(pin した SHA を指定しない。既存クローンが
# あれば触らないだけ)。JMMLU 側に1コミットでも追加されていれば、
#
#   問題の件数が変わる → item.id が変わる → split_dev_holdout の帰属が変わる
#   → **DEV と HOLDOUT の分割が変わる**
#
# 分割は preregister の「変えてはいけないもの」である。しかも HOLDOUT に DEV の問題が
# 流れ込んでも**エラーは出ない。静かに壊れる。** よってここで明示的に SHA を固定し、
# 生成物が manifest と一致することを確かめる。

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_cmd git
COMMITTED_MANIFEST="data/jmmlu.manifest.json"
GENERATED_MANIFEST="reports/jmmlu.generated.manifest.json"
REPO_DIR="data/raw/JMMLU"

[[ -f "$COMMITTED_MANIFEST" ]] || { echo "manifest が無い: $COMMITTED_MANIFEST" >&2; exit 1; }
mkdir -p reports

banner "1. JMMLU を pin した commit で取得"
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "既存のクローンを使う: $REPO_DIR"
else
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --quiet https://github.com/nlp-waseda/JMMLU.git "$REPO_DIR"
fi

# ★ ここが肝。ingest は HEAD をそのまま使うので、**呼ぶ前に SHA を固定する。**
current="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ "$current" != "$JMMLU_COMMIT" ]]; then
  echo "HEAD が pin と違う($current)。$JMMLU_COMMIT に固定する。"
  git -C "$REPO_DIR" fetch --quiet origin "$JMMLU_COMMIT" 2>/dev/null || git -C "$REPO_DIR" fetch --quiet origin
  git -C "$REPO_DIR" checkout --quiet --detach "$JMMLU_COMMIT"
fi
echo "commit = $(git -C "$REPO_DIR" rev-parse HEAD)"

banner "2. 変換(manifest は別名に出して、コミット済みのものを上書きしない)"
$PY tools/ingest_jmmlu.py \
  --out "$BENCHMARK" \
  --repo-dir "$REPO_DIR" \
  --manifest "$GENERATED_MANIFEST"

banner "3. 照合 — コミット済み manifest と完全一致するか"
$PY - "$COMMITTED_MANIFEST" "$GENERATED_MANIFEST" "$REPO_DIR" <<'PYEOF'
import hashlib, json, pathlib, sys

committed, generated = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
repo_dir = pathlib.Path(sys.argv[3])

if committed == generated:
    print("一致。取得元・件数・除外内訳・各 CSV の sha256 まで同一。")
    raise SystemExit(0)

# ★ ここから先は「不一致を見逃す」処理ではなく、「不一致に**説明を要求する**」処理である。
#
#   2026-08-06、Lambda の Linux ホストで最初にここが落ちた。差分は subjects だけ、しかも
#   各科目の rows / accepted / rejected は全53科目一致していて、**違うのは CSV の sha256
#   だけ**だった。原因は git の改行変換 —— manifest を作ったのは Windows で、autocrlf が
#   チェックアウト時に LF を CRLF へ書き換えていた。記録されていたのは**変換後**のハッシュ
#   であり、上流リポジトリのバイト列は LF のままである。
#
#   だからといって「ハッシュが違っても件数が合えばよい」に緩めるのは、この段の存在意義
#   (静かに壊れる事故を捕まえる)を捨てることになる。**改行差だと主張するなら、
#   committed のハッシュを CRLF 変換でバイト単位に再現できなければならない。**
#   再現できない科目が1つでもあれば、従来どおり止める。
def explains_as_line_endings():
    if set(committed) != set(generated):
        return False, "manifest のキー集合が違う"
    for key in committed:
        if key != "subjects" and committed[key] != generated[key]:
            return False, f"{key} が違う(改行では説明できない)"

    cs, gs = committed["subjects"], generated["subjects"]
    if len(cs) != len(gs):
        return False, "科目数が違う"

    for c, g in zip(cs, gs):
        if c["subject"] != g["subject"]:
            return False, "科目の並びが違う"
        # sha256 **以外**が1つでも違えば改行では説明できない。
        if {k: v for k, v in c.items() if k != "sha256"} != \
           {k: v for k, v in g.items() if k != "sha256"}:
            return False, f"{c['subject']}: 件数か除外内訳が違う"
        if c["sha256"] == g["sha256"]:
            continue

        path = repo_dir / "JMMLU" / f"{c['subject']}.csv"
        if not path.is_file():
            return False, f"{c['subject']}: CSV が見つからない({path})"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != g["sha256"]:
            return False, f"{c['subject']}: 手元の CSV が生成 manifest と一致しない"
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if hashlib.sha256(crlf).hexdigest() != c["sha256"]:
            return False, f"{c['subject']}: 改行変換では committed の sha256 を再現できない"
    return True, ""


ok, why = explains_as_line_endings()
if not ok:
    print("★ 不一致。ベンチマークが再現していない。実験を進めてはいけない。", file=sys.stderr)
    print(f"  {why}", file=sys.stderr)
    for key in sorted(set(committed) | set(generated)):
        if committed.get(key) != generated.get(key):
            print(f"  差分のあるキー: {key}", file=sys.stderr)
    if committed.get("totals") != generated.get("totals"):
        print(f"  committed totals = {committed.get('totals')}", file=sys.stderr)
        print(f"  generated totals = {generated.get('totals')}", file=sys.stderr)
    raise SystemExit(1)

print("▲ 各 CSV の sha256 が committed と違う。**差は改行だけであることを実測で確認した。**")
print("  committed 側は Windows のチェックアウト(CRLF)を記録している。手元の CSV を")
print("  LF→CRLF に変換すると committed のハッシュを**バイト単位で再現できた**(全科目)。")
print("  件数・除外内訳・科目の並び・取得元 commit はすべて一致している。")
print()
print("  ★ 問題の id は内容ではなく**位置**で決まる(ingest_jmmlu.py:296 の")
print("    `jmmlu/{subject}/{index:04d}`)。よって改行差は id を動かさず、")
print("    DEV/HOLDOUT の分割にも影響しない —— それを次の段で件数で確かめる。")
print()
print("  ★ ただし ingest は newline=\"\" で読む(csv モジュールの作法)ので、")
print("    **引用フィールド内部の改行は問題文にそのまま残る。** Windows 版の問題文には")
print("    CR が紛れており、**この Linux 版のほうが上流のバイト列に忠実**である。")
print("    プロンプトが CR の分だけ違うので、**Windows 時代のキャッシュとは混ぜない。**")
print("    (環境タグでキャッシュを分けているので既に満たされている)")
PYEOF

banner "4. 分割の同一性 — DEV / HOLDOUT の件数が凍結値と一致するか"
# manifest が一致していれば id 集合も一致し、split_dev_holdout は id だけで決まる
# (benchmark.py:201)ので分割も一致する。それでも確認する。**静かに壊れる種類の
# 事故なので、確認は重ねる価値がある。**
CONTAMLAB_EXPECT_DEV="$JMMLU_EXPECTED_DEV" \
CONTAMLAB_EXPECT_HOLDOUT="$JMMLU_EXPECTED_HOLDOUT" \
CONTAMLAB_EXPECT_ACCEPTED="$JMMLU_EXPECTED_ACCEPTED" \
CONTAMLAB_BENCHMARK="$BENCHMARK" \
$PY - <<'PYEOF'
import os, sys
from pathlib import Path
from contamlab.benchmark import load_jsonl, split_dev_holdout

items = load_jsonl(Path(os.environ["CONTAMLAB_BENCHMARK"]))
dev, holdout = split_dev_holdout(items)
want = {
    "全体": (len(items), int(os.environ["CONTAMLAB_EXPECT_ACCEPTED"])),
    "DEV": (len(dev), int(os.environ["CONTAMLAB_EXPECT_DEV"])),
    "HOLDOUT": (len(holdout), int(os.environ["CONTAMLAB_EXPECT_HOLDOUT"])),
}
ok = True
for label, (got, expected) in want.items():
    mark = "OK " if got == expected else "★NG"
    print(f"  {mark} {label}: {got}(凍結値 {expected})")
    ok &= got == expected
if not ok:
    print("★ 分割が凍結値と違う。preregister の「変えてはいけないもの」に該当する。",
          file=sys.stderr)
    raise SystemExit(1)
PYEOF

banner "完了"
echo "ベンチマークは pin どおり再現した: $BENCHMARK"
