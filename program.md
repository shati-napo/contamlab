# program.md — contamlab の実験指示書

**人間が育てるファイル。** エージェントはこれを読んで実験を回し、書き換えない
(書き換えるのは `contamlab/perturb.py` だけ)。ランが終わったら、学んだことを人間が
「前回までの学び」に反映する。次のランはここから始まる。

jstock-analyzer-v2 の `research/program.md` と同じ構造。対象だけが違う。

## 対象

- 目的: **汚染の効果量下限を測る**(`adjusted_lcb` = 割引後の片側信頼下限)
- 固定評価系: [contamlab/harness.py](contamlab/harness.py) — **編集禁止**
- 可変ファイル: [contamlab/perturb.py](contamlab/perturb.py) — ここだけ書き換える
- メトリクス: `adjusted_lcb`。**高いほど汚染の証拠が強い**
- 1実験の所要: モデルへの問い合わせ回数 = 問題数 × 2 × モデル本数。
  応答キャッシュが効くので、同じ問題・同じシードなら2回目以降はゼロ

## 実行方法

```powershell
# ① まず検出力。何問必要かを計算してから問題を集める
python -m contamlab power --effect 0.05 --discordant-rate 0.30

# ② 手元の問題数で何ポイントまで見えるか
python -m contamlab power --n 800 --discordant-rate 0.30 --effect 0.05

# ③ 摂動を目で確かめる(正解が壊れていないこと)
python -m contamlab perturb --benchmark data/bench.jsonl --seed dev-seed --limit 3

# ④ 本番
python -m contamlab run --benchmark data/bench.jsonl --seed dev-seed \
  --target-effect 0.05 --expected-discordant-rate 0.30 --k 1 \
  --model fake:demo:0.6 --json reports/run.json
```

**測定装置の健全性チェック**(`perturb.py` を触った後、およびランの開始時に1回):

```powershell
python -m contamlab verify
```

3項目(汚染ありを検出する / 汚染なしを検出しない / 何も変えなければ差はちょうど0)を
合成データと `FakeModel` で確かめる。**これが落ちたら実験を止める**
(測定装置が壊れていれば全実験が無価値)。

## 変えてよいもの / いけないもの

**変えてよい(`contamlab/perturb.py` の中だけ):**

- 新しい摂動器の追加(**1つ追加 = 事前確約の K を1つ消費する**)
- 既存摂動器のパラメータ

**変えてはいけない:**

- `contamlab/harness.py`、`contamlab/stats/` 配下、`contamlab/runner.py` の採点規則
- 指標の定義、DEV/HOLDOUT の分割、非公開シード
- 依存パッケージの追加(**標準ライブラリのみ**を維持する)
- 応答キャッシュ(追記専用。実験の都合で消さない)
- **測定条件**(`--temperature` / `--max-tokens` / 拡張思考の有無)。
  変えると測っているものが変わる。事前確約に書いてからにする

## 実 API を使うときの手順

```powershell
# ① 見積もりだけ出す(--yes が無ければ課金されない)
python -m contamlab run --benchmark data/bench.jsonl --seed dev-seed `
  --target-effect 0.05 --expected-discordant-rate 0.30 `
  --model anthropic:claude:claude-opus-5 --model openai:gpt:gpt-4o

# ② 回数を確認してから実行
python -m contamlab run ... --yes --rate-limit 50
```

- API キーは `.env` に置く。**モデル指定文字列に書かない**(シェル履歴とログに残る)
- 応答キャッシュが効くので、同じ問題・同じシードでの再実行は課金ゼロ。**消さない**
- 実行後に「同じ問いに違う応答が N 件」と出たら、**モデルが非決定的**。
  temperature を確認する。この状態の数字は使わない

## keep / discard

- `adjusted_lcb` が改善し、かつ **改善幅 > drop_se** なら keep
- そうでなければ `git reset --hard <直前のkeep commit>` で戻す
- 同程度なら単純な方を採る。摂動器を削って同等以上なら明確な勝ち
- **1実験 = 1摂動器。** 2つ同時に試さない。組み合わせるのは単体で keep になったもの同士だけ

## 記録

`reports/results.tsv`(タブ区切り・git 管理外)に毎実験1行:

```
commit	adjusted_lcb	drop_se	observed_psi	status	description
```

## 停止条件

- **事前確約した K に到達したら停止**し、Phase 2(HOLDOUT 開封と報告)へ。
  K は [preregister.md](preregister.md) に書く。**この対象では 10 以下**
- 人間に止められたら停止
- `contamlab verify` が落ちたら**即停止して報告**
- 固定評価系のバグ・採点の非対称の疑いが出たら**即停止して報告**

それ以外では止まらない。「続けますか」と聞かない。

## この対象に固有の注意

- **非公開シードはループ中に絶対使わない。** 1構成・1回だけ、Phase 2 で使う
- **不一致率 ψ が高いほど検出力は下がる。** 摂動を強くすれば見えやすくなる、は誤り。
  強い摂動は ψ を上げ、必要問題数を増やす
- **「全モデルが一律に落ちた」を汚染と読まない。** 摂動の難易度上昇と区別できない。
  不均一さ検定(Cochran の Q)が有意でなければ、そう報告する
- **解釈不能率が条件間でずれたら、落ちたのは能力ではなく採点。** `harness` が警告を出す
- 応答キャッシュの `conflicts` が空でなければ、**モデルが非決定的**。temperature を確認する
- **尤度ベースの手法(MIA / Min-K% / 交換可能性検定)に手を出さない。**
  Claude・OpenAI が log-prob を返さず、かつ MIA は現実設定でほぼランダムと示されている
- **`ollama` は PATH に無い。** 実体は
  `C:\Users\kingo\AppData\Local\Programs\Ollama\ollama.exe`。`ollama ps` / `ollama list` が
  失敗してもサービスは生きている可能性が高い。**状態確認は CLI ではなく API で行う** ——
  `Invoke-RestMethod http://localhost:11434/api/ps`。
  **ランの進捗は `data/cache/responses.jsonl` の行数**で見る(追記専用なので代理指標になる)
- **Ollama のモデル名のコロンが `compat` spec を壊す。** `compat:NAME:MODEL_ID:BASE_URL` は
  `rest.split(":", 2)` で切るため(`clients.py:344`)、`qwen2.5:3b` のようなコロン入りの名前では
  BASE_URL の解釈が壊れる。**`ollama cp <元名> <コロン無しの別名>` で回避する。**
  `clients.py` は編集しない(`perturb.py` 以外の改変は禁止)
- **ローカル実行では `--json` が無いと正解率が読めない。** テキストレポートは
  `accuracy_original` / `accuracy_perturbed` を出力しない(`report.py:142-143` は JSON 側のみ)。
  **パイロットでは必ず `--json` を付ける**

## 前回までの学び

- **2026-08-02**: 効果量の下限に Wilson 近似を使ったところ、「p 値は非有意なのに下限が
  0 を超える」偽陽性が出た。**判定を下限で行う以上、区間は厳密(Clopper-Pearson)でなければ
  ならない。** 近似区間と厳密検定を混ぜると同値関係が壊れる
- **2026-08-02**: 効果量の下限だけを判定に使うと、汚染のないモデルを3本並べただけで
  偽陽性が出た。**K は摂動器の本数しか数えていない。モデルの本数は Holm で別に補正する**
- **2026-08-02**: 模擬モデル(`FakeModel`)の応答が実モデル用の応答キャッシュに書き込まれていた。
  キャッシュのキーは **モデル名とプロンプトだけ** なので、実 API のモデルにたまたま `clean` や
  `dirty` と名付けると偽の応答を拾う。**種別を跨いだ取り違えはキー側では防げない。書き込まない
  ことで防ぐ**(`cli._build_model` は実 API のときだけキャッシュで包む)。
  デモで汚れたキャッシュは `_trash/responses.demo-fakemodel-20260802.jsonl` に退避済み
- **2026-08-03**: `harness.run` の検出力ゲートが **Holm 補正を織り込んでいなかった。**
  `harness.py:139` は `plan(..., alpha=design.alpha)` を呼ぶので必要問題数を生の α=0.05 で
  計算するが、判定側(`harness.py:101`)は `p_holm < alpha` である。結果 **M>1 でゲートが甘くなる** ——
  ψ=0.405・5pt・3本で Holm 後に必要な 1,427 問に対し、`plan` は n=1,000 でも `adequate=True` を返す。
  **偽陽性は出ない**(判定の条件式は正しい)ので `verify` は通る。当面の対応は
  **`--alpha` を既定 0.05 のままにし、問題数を自分で決めること**(0.0167 を渡すと信頼区間と
  Cochran Q まで二重補正になる)。★ **2026-08-02 の「K はモデル本数を数えていない」と同根。
  補正を2箇所に分けたら、分割の下流すべてを点検する。** ⬜ harness.py を直すかは未決。
- **2026-08-03**: **量子化も測定条件だった。** ロースター内で量子化レベルや量子化提供元が混ざると、
  不均一さ検定が「学習コーパスの違い」と「量子化の違い」を区別できず**採用基準4を損なう。**
  実際 Sakana AI 公式の TinySwallow GGUF は `q5_k_m`/`q8_0` のみで Q4 が無く、
  そのまま使えば1本だけ量子化が違うロースターになっていた。**測定条件に量子化と提供元を加える。**
- **2026-08-03**: **ψ は実測してもそのまま使わない。** パイロット②(200問)で ψ̂=0.335 を得たが、
  想定 0.300 に対し実測が上振れした結果、そのラン自身が **達成検出力 0.791 < 目標 0.80** に終わった。
  **ψ を下側に誤ると検出力を過大評価する**ので、本番の想定 ψ には
 **Clopper-Pearson の上側限界(両側95%の上側 = 保守的な方)**を使う
  (今回は 0.4050。片側95%なら 0.3814)。
- **2026-08-03**: **事前に宣言する判断基準に隙間を作らない。** パイロット①で
  「45% 以上なら続行 / 25〜30% なら中止」と宣言したところ、実測 0.38 が**2帯の間に落ちた。**
  中間値が出たときが最も事後正当化しやすい。**帯は連続に張り、それでも間に落ちたら
  基準を書き換えず、判断の算術を数値で記録する。**
