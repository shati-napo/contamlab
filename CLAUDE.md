# CLAUDE.md — contamlab

LLM ベンチマークの**汚染**を、検定・信頼区間・検出力・多重比較補正つきで測る。
詳細は [README.md](README.md)、実験の指示は [program.md](program.md)、
事前確約は [preregister.md](preregister.md)。

## ▶ 作業を始める前に必ず読む

1. **この文書の「絶対禁止」** — 破ると研究資産が回復不能になる
2. [program.md](program.md) の「変えてよいもの / いけないもの」と「前回までの学び」
3. [preregister.md](preregister.md) の「測定条件」と「変更履歴」

---

## 🚫 絶対禁止: Amazon Bedrock

**Bedrock には触れない。使わない。提案しない。例示もしない。**

「AWS を使う」という話が出ても、**Bedrock は選択肢に入らない。**
AWS で許されるのは **EC2 の計算資源を借りて、オープンウェイトのモデルを
自分のプロセス(Ollama / llama.cpp)で動かすこと**だけである。

### 具体的に禁止するもの

| 禁止 | 例 |
|---|---|
| Bedrock の API 呼び出し | `bedrock-runtime`、`boto3.client("bedrock*")` |
| Bedrock 経由のクライアント実装 | `anthropic.AnthropicBedrock`、`langchain_aws` 等 |
| Bedrock のモデル ID の使用 | `anthropic.claude-*`、`amazon.nova-*` 等 |
| **提案・例示・コメント内での推奨** | 「Bedrock なら簡単です」「Bedrock でも可」 |
| 「同じ AWS だから等価」という扱い | EC2 と Bedrock を並べて選ばせること |

**最後の2つが特に重要。** コードを書かなくても、選択肢として提示した時点で
誘導になる。**迷ったら Bedrock の名前を出さない。**

### なぜ禁止か

このプロジェクトの中心にある逆説([README.md](README.md) の「中心にある逆説」):

> **汚染検出ツールを公開した瞬間、そのテストセットが汚染される。**

HOLDOUT 1,922 問は「まだ誰にも見られていない」ことに全価値がある。
一度外に出れば次世代モデルの訓練データに混ざり、**検査器として二度と使えない。**
代替が利かない一発勝負の資産である。

Bedrock は AWS の看板をかぶった**モデル提供者への API** である。叩けば
HOLDOUT の問題文がモデル提供者のサーバーに送信され、少なくともログに残る。
規約を信用するかどうかの問題ではない —— **この設計は「信用する」ではなく
「そもそも渡さない」で解決している。**

加えて、ローカル実行を選んだ3つの理由([preregister.md](preregister.md) の「測定条件」)を
Bedrock は2つ壊す。

1. **オープンウェイトなら学習コーパスを検証できる** → Bedrock のモデルはコーパス非公開。
   陽性が出ても `llm-jp-corpus-v3` との直接照合ができない
2. JMMLU スコアを報告している国内モデルが対象集団である
3. **課金ゼロなので標本サイズが API コストに律速されない** → 従量課金で律速が復活する

### 判定の一行

> **問題文が、自分の管理下のプロセスの外に出るか。** 出るなら禁止。

EC2 は出ない(GGUF を自分のディスクに置き、自分のプロセスで読む。AWS は電気とハコを
貸しているだけ)。Bedrock は出る。同じ AWS でも意味が真逆である。

同じ理由で **Anthropic / OpenAI の API も HOLDOUT には使わない。**
`clients.py` に実装が残っているのは汎用ツールとしての機能であり、
**このランでは使わない**(README の使用例をそのまま実行しないこと)。

---

## 🚫 応答キャッシュに環境が入っていない

キャッシュのキーは **モデル名 + プロンプトだけ**([runner.py:180](contamlab/runner.py#L180)):

```python
return hashlib.sha256(f"{model_name}\x00{prompt}".encode("utf-8")).hexdigest()
```

**ハードウェア・バックエンド・量子化・温度・`max_tokens` はどれも入っていない。**
実行環境が1つしかない間はそれで正しかったが、**環境を変えた瞬間に前提が崩れる。**

### 最大のリスクは「静かな混入」で、`conflicts` は立たない

[runner.py:223-229](contamlab/runner.py#L223-L229):

```python
def answer(self, prompt: str) -> str:
    cached = self._cache.get(self.name, prompt)
    if cached is not None:
        return cached          # ← モデルを呼ばずに帰る
```

**キャッシュにあればモデルは一度も呼ばれない。** よって CPU 時代のキャッシュを
持ったまま GPU で走らせると、ログ上は「GPU で N コール完了」に見えるのに
**GPU は一度も呼ばれず、古い答えがそのまま返る。**
新しい応答を生成しないので `put()` に到達せず、**`conflicts` も立たない。**
エラーも警告も出ない。

`conflicts` が立つのは運が良い場合だけで、しかもそのとき
[runner.py:188](contamlab/runner.py#L188) は原因を「**モデルが非決定的である証拠**」と
記述する。**真犯人がバックエンド変更でも、そう表示される。**

### 規律

- **バックエンド・ハードウェア・Ollama バージョン・GGUF を変えたら、キャッシュファイルを
  新しく切る。** `--cache` で別パスを渡すか、既存を日付入りの名前に退避する
- **既存のキャッシュは消さない**(追記専用。`program.md` の禁止事項)。改名して残す
- 環境ごとにキャッシュを分ける。混ぜない

### temperature 0 は決定性を保証しない

`temperature 0` は「最も確率の高い選択肢を選ぶ」であって
「**計算結果が同じになる**」ではない。確率の計算は大量の浮動小数の加算であり、
**加算順序が変われば最下位桁が変わる。** CPU と GPU では並列化が違うので順序が違い、
1位と2位が僅差の問題で順位が入れ替わりうる。

**そしてそれが起きるのは、モデルが迷っている問題である。** `shuffle_choices` で
答えが変わるのも、モデルが迷っている問題。**ノイズが乗る集合と測りたい集合が重なる。**

パイロット①で得た「不一致ペア 0件 = 完全決定的」は **CPU での測定値**。
バックエンドを変えたら**測り直す。**

---

## AWS EC2 を使うときの手順

| 項目 | 規律 |
|---|---|
| サービス | **EC2 のみ**(GPU インスタンス)。Bedrock は上記のとおり禁止 |
| 実行系 | インスタンス内に Ollama を入れ、`http://localhost:11434/v1` を叩く |
| モデル | GGUF を自分でダウンロードする。`mmnga` / Q4_K_M 統一を維持 |
| 再現性 | **Ollama のバージョンと GGUF の SHA256 を AMI か Docker イメージに固定**し、`preregister.md` に記録する |
| キャッシュ | **新規に切る。** CPU 時代のものを持ち込まない |
| 事前確約 | 実行系の変更を**結果を見る前に** `preregister.md` の変更履歴へ書く |
| Spot | 使ってよい(キャッシュが追記専用なので中断後に再開できる) |
| 秘匿 | **非公開シードと HOLDOUT はインスタンス内に閉じる。** S3 等に置かない |

**イメージ固定はノートPCより厳密になる。** 手元の機体は VS Code のプロセス数で
モデルが載ったり載らなかったりする環境([preregister.md](preregister.md) のハードウェア
上限の節)であり、固定イメージにすればその揺れが消える。

---

## 変えてよいもの / いけないもの

[program.md](program.md) が正。要点だけ再掲する。

- **変えてよいのは [contamlab/perturb.py](contamlab/perturb.py) だけ**
  (1つ摂動器を足す = 事前確約の K を1つ消費する)
- **編集禁止**: `harness.py` / `stats/` 配下 / `runner.py` の採点規則 /
  指標の定義 / DEV・HOLDOUT の分割 / 非公開シード
- **依存パッケージを増やさない**(標準ライブラリのみ)
- **測定条件**(`--temperature` / `--max-tokens` / 拡張思考 / 量子化 / 量子化提供元 /
  実行系)を変えるときは、**事前確約に書いてから**変える

## 環境メモ

- `ollama` は PATH に無い。実体は
  `C:\Users\kingo\AppData\Local\Programs\Ollama\ollama.exe`。
  状態確認は CLI ではなく API で —— `Invoke-RestMethod http://localhost:11434/api/ps`
- Ollama のモデル名のコロンが `compat:NAME:MODEL_ID:BASE_URL` の解析を壊す
  ([clients.py:344](contamlab/clients.py#L344))。`ollama cp` でコロン無しの別名を作る。
  **`clients.py` は編集しない**
- ローカル実行では `--json` が無いと正解率が読めない([report.py](contamlab/report.py))。
  パイロットでは必ず付ける
- ランの進捗は `data/cache/responses.jsonl` の**行数**で見る(追記専用なので代理指標になる)

## テスト

```powershell
pip install -e ".[dev]"
pytest
python -m contamlab verify   # 測定装置の健全性。落ちたら実験を止める
```
