# Nejumi Leaderboard 4 の per-sample データは取れるか(2026-08-08)

**判定: 取れた。** [NEXT.md](NEXT.md) のステップ1の成功条件3つがすべて揃った。

これは**可否の確認のみ**である。測定はしていない。下の数字は「取れることを示すために
1テーブルを開いて数えた」もので、主張の根拠として使うものではない。使うなら事前登録が先。

---

## 成功条件の照合

| # | 条件 | 結果 | 根拠 |
|---|---|---|---|
| 1 | 問題ごとの正誤が取れる | **○** | 1 問 1 行のテーブルが W&B の run ファイルとして公開されている |
| 2 | 3パターンが同一の問題集合に対応 | **○** | テーブルが**既に問題単位で3パターン結合済み**。1 行に normal / SymbolChoice / IncorrectChoice の出力と期待値が並ぶ |
| 3 | モデル横断で同じことができる | **○** | 確認した 6 モデルすべてで同一テーブルが存在し、問題集合の指紋が完全一致 |

**認証は不要だった。** 公開プロジェクトのため API キーなしの HTTP だけで取得できる。
`wandb` パッケージも要らない(本プロジェクトの標準ライブラリ限定の鉄則を破らずに済む)。

---

## 取得手順

対象: entity `llm-leaderboard` / project `nejumi-leaderboard4`(`configs/base_config.yaml` に記載)。
2026-08-08 時点で run 数 1421、`access: USER_READ`。

### 1. run を列挙

```
POST https://api.wandb.ai/graphql
{"query":"query{project(name:\"nejumi-leaderboard4\",entityName:\"llm-leaderboard\"){runs(first:N){edges{node{name displayName state}}}}}"}
```

`name` が run ID(例 `vwdpxyy8`)、`displayName` がモデル名(例 `nex-agi/Nex-N2-Pro: reasoning-enabled`)。

### 2. run のファイル一覧から目的のテーブルを引く

```
POST https://api.wandb.ai/graphql
{"query":"query{project(name:\"nejumi-leaderboard4\",entityName:\"llm-leaderboard\"){run(name:\"<run_id>\"){files(first:500){edges{node{name sizeBytes directUrl}}}}}}"}
```

欲しいファイル名の接頭辞:

| ファイル | 中身 |
|---|---|
| `media/table/jmmlu_robust_2shot_output_table_<n>_<hash>.table.json` | **これが本命。** test サブセット・1 問 1 行・3パターン結合済み(約 240 KB) |
| `media/table/jmmlu_robust_2shot_output_table_dev_<n>_<hash>.table.json` | 同じものの dev サブセット |
| `media/table/jmmlu_robust_2shot_leaderboard_table_<n>_<hash>.table.json` | 集計値のみ(82 B)。**これでは McNemar は組めない** |
| `media/table/jaster_2shot_output_table_<n>_<hash>.table.json` | jaster 全タスク 2500 行。うち `task == "jmmlu"` は 100 行 |

`_dev_` を含むファイル名を除外しないと dev を掴む。`<n>_<hash>` は run ごとに変わるので
接頭辞一致で探す。

### 3. `directUrl` を GET

署名付き URL がそのまま返る。中身は UTF-8 の JSON で `{"columns": [...], "data": [[...], ...]}`。
Windows のコンソールに直接流すと文字化けするが、ファイル自体は UTF-8 で壊れていない。

---

## テーブルの列と、問題 ID の対応

`jmmlu_robust_2shot_output_table` は 18 列 × 100 行。

```
model_name, index, score,
input_normal, output_normal, expected_output_normal,
input_SymbolChoice, output_SymbolChoice, converted_output_SymbolChoice, expected_output_SymbolChoice,
input_IncorrectChoice, output_IncorrectChoice, converted_output_IncorrectChoice, expected_output_IncorrectChoice,
dataset, task, num_few_shots, subset
```

- **問題 ID は `index`**(0〜99、重複なし)。`task` は全行 `jmmlu`、`dataset` は `jaster`。
- **3パターンの突き合わせは不要**(既に 1 行に入っている)。結合は Nejumi 側の
  `scripts/evaluator/evaluate_utils/robustness.py::evaluate_robustness` が
  タスクごとに行番号順で `zip` して作っており、長さ一致を `assert` している。
- **モデル間の対応は `index` で取れる。** 確認した 6 モデルで
  `sha256(concat(index, input_normal))` が完全一致(指紋 `11e9b5af09152971`)。
  同じ 100 問を同じ順で撃っている。

### 正誤の導出(ここを間違えると数字が壊れる)

テーブルの `score` 列は**正誤ではない**。3つの出力が互いに一致した組の数に応じた
1.0 / 0.5 / 0.0 の**整合性スコア**である(`eval_robustness`)。正誤は自分で出す:

| パターン | 正解の条件 |
|---|---|
| normal | `output_normal == expected_output_normal`(A/B/C/D の文字列一致) |
| SymbolChoice | `output_SymbolChoice == expected_output_SymbolChoice`(記号 `$ & # @` のまま比較) |
| IncorrectChoice | **集合一致**。`set(output.split(","))== set(expected.split(","))` |

IncorrectChoice は「不正解の選択肢を**全て**カンマ区切りで挙げよ」という課題で、
期待値は `"A,B,D"` のような 3 要素の並びになる(`scripts/data_uploader/create_jmmlu_robustness_data.py`)。
単一文字が含まれるかを見る素朴な判定だと正解数が 0〜4/100 まで落ちて、まったく別の数字になる。

参考として 1 モデル(`nex-agi/Nex-N2-Pro`)で数えた内訳: normal 91 / SymbolChoice 86 /
IncorrectChoice 81(いずれも 100 問中)。**再掲するが、これは可否の確認の副産物であって
測定ではない。**

---

## 記録しておくべき制約

- **n = 100。** robust テーブルも、`jaster_2shot_output_table` の `task == "jmmlu"` 行も 100。
  Nejumi 4 の JMMLU 頑健性検査はこの規模で回っている。
- **2-shot のみ。** `jaster.py` の該当箇所は `if cfg.run.jmmlu_robustness and few_shots:`
  の中にあり、0-shot 側では robust テーブルが log されない。
- 分野(subject)の列が無い。`task` は `jmmlu` 一本で、JMMLU の科目別内訳はこのテーブルからは取れない。
- モデルによっては robust テーブルを持たない run がありうる(接頭辞一致で見つからなければ飛ばす)。
  今回引いた新しい方の 6 run はすべて持っていた。

---

## 次にどうするか

[NEXT.md](NEXT.md) の分岐に従い、**ここで寝かせる。**
着手はステップ2(陽性対照の自作)の後。較正済みの装置を持ってから読むほうが、
立場が「他人の数字へのダメ出し」にならずに済む。

このファイルは**取得手順と問題 ID の対応の記録**であり、着手の合図ではない。
実際に使うときは、**先に事前登録**([preregister.md](../preregister.md))へ
対象モデル・比較の立て方・判定基準を書いてからにする。
