# CLAUDE.md — contamlab

LLM ベンチマークの**汚染**を、検定・信頼区間・検出力・多重比較補正つきで測る。
詳細は [README.md](README.md)、実験の指示は [program.md](program.md)、
事前確約は [preregister.md](preregister.md)。

## ▶ 作業を始める前に必ず読む

1. **[docs/NEXT.md](docs/NEXT.md) — 再開点。**前回の状態と「次の一手」。**ここから始める**
2. **この文書の「絶対禁止」** — 破ると研究資産が回復不能になる
3. [program.md](program.md) の「変えてよいもの / いけないもの」と「前回までの学び」
4. [preregister.md](preregister.md) の「測定条件」と「変更履歴」
5. [docs/positive-control-arc.md](docs/positive-control-arc.md) — 陽性対照を自作するまでの 11 ラン・約 $103 の要約
6. [docs/run-jmmlu-shuffle-03.md](docs/run-jmmlu-shuffle-03.md) — **HOLDOUT を消費したラン(2026-08-07)。**
   結論・何が言えて何が言えないか・**次に決めること**(§6)がまとまっている
7. [docs/run-jmmlu-shuffle-02.md](docs/run-jmmlu-shuffle-02.md) — その前のラン(中止)。経緯の参照用

> [!note] 現在地(2026-08-19 夜)
> ▶ **[docs/NEXT.md](docs/NEXT.md) 冒頭の「次のセッションはここから着手する」を読む。**
> **そのブロックだけで着手できる**ように書いてある(以降 1,300 行は必要になってから読む)。
> - ✅ **陽性対照の自作は 2026-08-16 の `lambda-ladder-01` L1 で成立済み**(a・b・c すべて k=3 で合格)
> - 🔴 **較正曲線は `calibration-curve-01`(2026-08-18)で停止**。1点も引けていない
> - ⬜ ★ **`scripts/70-positive-control.sh` は 12 ラン通じて一度も走っていない。**
>   `drop` / `p_value` / `adjusted_lcb` は**実モデルに対して1つも計算されたことがない**
>
> - ✅ **作業1〜3($0)は 2026-08-19 に完了。**★ **統計層に欠陥 P-1 を1件見つけた** ——
>   `power.py` は正規近似検定の、`mcnemar.py` は厳密条件付き検定の話をしており、
>   **検出力の数字は常に楽観側**(0.80 のはずが実際 0.774 / 493問 → 527問)。
>   ⛔ 編集禁止領域なので**直していない** → [docs/power-verify-2026-08-19.md](docs/power-verify-2026-08-19.md)
>
> ★ **次の着手は作業4 = B案。**合格実績のある x40(λ=0.8・埋め草ゼロ)を1点だけ作り直し、
> `70-positive-control.sh` を**初めて通す**($14)。**2026-08-19 にユーザーが決定済み・
> 実行は 2026-08-20 に回した。**⛔ **案の再検討はしない。**
> ⛔ **着手前に事前登録を書く。**Lambda の API キーは失効済みなので**発行から**。
> 手順は [docs/NEXT.md](docs/NEXT.md) の決定ブロックが正。

> [!note] 旧・現在地(2026-08-08・履歴として残す)
> **公開済み**(PUBLIC)。次は **[docs/NEXT.md](docs/NEXT.md) のステップ1** —— Nejumi の
> 問題ごとの正誤が取れるかを **1 時間で**確かめる。**測定はしない。**

> [!warning] ⚠️ **HOLDOUT は 2026-08-07 に開封・消費した(K = 1 / 10)**
> 「1構成・1回だけ開封」の規則により、**同じ HOLDOUT でもう一度検定することはできない。**
> 摂動器を増やすにも、別のベンチマークで新しい DEV / HOLDOUT を切るところからになる。
> 選択肢は [docs/run-jmmlu-shuffle-03.md](docs/run-jmmlu-shuffle-03.md) の §6 に列挙してある。
> **成果物(公開)を装置の改良より先に置くこと。**

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

## GPU を借りて走らせるときの手順

**2026-08-06: 借り先は AWS に限らない。** AWS が G/VT のクォータを出さなかったため、
EC2 を前提にできなくなった。**判定基準は変わらない** —— 上の一行(問題文が自分の管理下の
プロセスの外に出るか)を満たす限り、GPU を時間借りして自分で Ollama を動かす形は
提供者を問わない。

| 項目 | 規律 |
|---|---|
| サービス | **計算資源だけを借りる。** マネージド推論 API は提供者を問わず禁止(上記) |
| 借り先の型 | **共同ホスト型(第三者が物理的にディスクに触れる形態)は使わない。** 非公開シードと HOLDOUT を置けない。**安さではなく、誰がディスクに触れるかで選ぶ** |
| 実行系 | ホスト内に Ollama を入れ、`http://localhost:11434/v1` を叩く。`OLLAMA_HOST=127.0.0.1` で外に口を開けない |
| モデル | GGUF を自分でダウンロードする。`mmnga` / Q4_K_M 統一を維持 |
| 再現性 | **Ollama のバージョンと GGUF の SHA256 を記録**し、`preregister.md` に残す |
| キャッシュ | **新規に切る。** CPU 時代のものを持ち込まない |
| 環境タグ | **EC2 の外では `CONTAMLAB_ENV_TAG` を明示する。** 既定は `host-<GPU名>-<日付>` になるが、提供者名を入れたほうが来歴として読める |
| 事前確約 | 実行系の変更を**結果を見る前に** `preregister.md` の変更履歴へ書く |
| **Spot / 中断あり枠** | ❌ **使わない(2026-08-05 訂正)。** 下記 |
| 秘匿 | **非公開シードと HOLDOUT はホスト内に閉じる。** S3 にも借り先のストレージにも置かない |

> [!warning] ❌ 「Spot でよい」は誤りだった
> 以前ここには「キャッシュが追記専用なので中断後に再開できる」と書いてあったが、
> **`00-launch-ec2.ps1` の block-device-mappings は `DeleteOnTermination=true`** であり、
> **Spot の中断動作の既定は terminate。** 中断された瞬間に **EBS ごと消える** ——
> キャッシュも環境タグも `reports/` も残らない。作り直しに払う GPU 時間のほうが
> 差額より高い。**中断されうる枠は使わない。** 一次情報は
> [scripts/README.md](scripts/README.md) の「Spot は使わない・ボリュームは 80GB」節。

**環境の固定はノートPCより厳密になる。** 手元の機体は VS Code のプロセス数で
モデルが載ったり載らなかったりする環境([preregister.md](preregister.md) のハードウェア
上限の節)であり、借りたホストを固定すればその揺れが消える。

---

## 変えてよいもの / いけないもの

[program.md](program.md) が正。要点だけ再掲する。

- **変えてよいのは [contamlab/perturb.py](contamlab/perturb.py) だけ**
  (1つ摂動器を足す = 事前確約の K を1つ消費する)
- **編集禁止**: `harness.py` / `stats/` 配下 / `runner.py` の採点規則 /
  指標の定義 / DEV・HOLDOUT の分割 / 非公開シード
  - ⚠️ 採点規則は **2026-08-07 に1度だけ変えた**(`select` の規則3が docstring の宣言と
    食い違っていた)。**前例にしない。** 触る前に [program.md](program.md) の
    「採点規則を1度だけ変えた」の4条件を全部満たすか確認する
- **依存パッケージを増やさない**(標準ライブラリのみ)
- **測定条件**(`--temperature` / `--max-tokens` / **`--prompt-format`** / 拡張思考 /
  量子化 / 量子化提供元 / 実行系)を変えるときは、**事前確約に書いてから**変える
  - ⚠️ **出力書式は手で指定しない。** パイロット⓪(`35-select-format.sh`)が選んで
    `reports/prompt-format` に焼き、後段が `prompt_format()` で読む。
    候補は `runner.PROMPT_FORMATS` の3つで**凍結**されており、4つ目を足すのは
    preregister「合格ゼロのときにやらないこと」が明示的に禁じている

## 環境メモ

- `ollama` は PATH に無い。実体は
  `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`。
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
