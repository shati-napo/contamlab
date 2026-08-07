# scripts/ — ラン `jmmlu-shuffle-03` の実行手順

**実験の設定は [preregister.md](../preregister.md) が正。** ここにあるのは、その設定を
取り違えずに実行するための道具である。数値がここと preregister で食い違ったら
**preregister が正**であり、スクリプトの側を直す。

> 🚫 このディレクトリのどのスクリプトも、マネージド推論 API を呼ばない。
> AWS で使うのは **EC2 の計算資源だけ**である([CLAUDE.md](../CLAUDE.md) の「絶対禁止」)。
> `00-launch-ec2.ps1` は IAM インスタンスプロファイルを**わざと付けない**ので、
> 起動したインスタンスはそもそもそれらを呼ぶ権限を持たない。規律ではなく権限で担保する。

## 順番

| | スクリプト | 場所 | すること |
|---|---|---|---|
| 0 | `00-launch-ec2.ps1` | 手元(Windows) | GPU インスタンスを1台起こす。**既定は表示のみ**。**AWS を使う場合のみ**(下記) |
| 1 | `10-bootstrap.sh` | インスタンス内 | Ollama 導入・決定性のための設定・GGUF 取得・`verify` |
| 2 | `20-rebuild-benchmark.sh` | インスタンス内 | JMMLU を **pin した SHA** から作り直し、manifest と照合 |
| 3 | `30-record-environment.sh` | インスタンス内 | 版と GGUF の SHA256 を記録。**環境タグを確定** |
| 4 | `35-select-format.sh` | インスタンス内 | パイロット⓪(150問・`identity`・3書式)**出力書式を確定**(2026-08-07 追加) |
| 5 | `40-pilot.sh 1` | インスタンス内 | パイロット①(70問・`identity`)採点の健全性と床効果 |
| 6 | `50-check-determinism.sh` | インスタンス内 | **GPU での決定性を実測** |
| 7 | `40-pilot.sh 2` | インスタンス内 | パイロット②(250問・`shuffle_choices`)ψ の実測 |
| 8 | `60-production.sh dev` | インスタンス内 | 本番 DEV。問題数は ψ の写像表が決める |
| 9 | `60-production.sh holdout` | インスタンス内 | **1構成・1回だけ。取り返しがつかない** |

各段は前の段の出力を事前条件として確認するので、飛ばすと止まる。
`30` は `reports/env-tag` を、**`35` は `reports/prompt-format` を**焼き、
後段は `require_env_tag` / `require_prompt_format` でそれを要求する。

> [!important] ★ 段 4(`35-select-format.sh`)は 2026-08-07 に足した
> ラン `jmmlu-shuffle-02` は「記号だけを答えてください」という**出力書式**に
> `llmjp3-13b` が乗らず、解釈不能率 5% 超で脱落して中止になった。
> 03 では**書式を候補3つから選び直す**。設計と、それが「落ちたモデルが通る条件を
> 落ちたのを見てから選ぶ」ことにならないための4つの装置は
> preregister「ラン: jmmlu-shuffle-03」にある。**先にそれを読むこと。**
>
> 段 5 以降は `--prompt-format "$(prompt_format)"` を自動で渡す。**手で書式を指定しない。**
> 段ごとに違う書式で測ると、キャッシュキーが分かれて呼び出しが無駄になるだけでなく、
> **正解率と ψ が段をまたいで比較できなくなる。**

## ⚠️ AWS の GPU クォータは**否認された**(2026-08-06)

`Running On-Demand G and VT instances`(`L-DB2E81BA` / ap-northeast-1)の 0 → 4 の
申請は**承認されなかった。** 2026-08-06 時点の実測:

| クォータ | 値 |
|---|---|
| `L-DB2E81BA` オンデマンド G/VT @ ap-northeast-1 | **0.0** |
| `L-3819A6DF` Spot G/VT @ ap-northeast-1 | **0.0** |
| `L-DB2E81BA` @ us-east-1 | **0.0** |
| `L-1216C47A` オンデマンド Standard @ ap-northeast-1 | 16.0 |
| `L-34B43A08` Spot Standard @ ap-northeast-1 | 32.0 |

**Spot に逃げてもリージョンを変えても回避できない**(どちらも 0 で、別途承認が要る)。
一方 **Standard 系の枠は空いている**ので、絞られているのは GPU だけである。
Service Quotas の申請履歴は `CASE_OPENED` のまま更新されていないので、
**API のステータスを承認の根拠にしない**(サポートケース側が正)。

→ **AWS 以外の GPU ホストで走らせる。** 手順は下記。`00-launch-ec2.ps1` を使わない
だけで、**インスタンス内の 1〜9 段はそのまま使える。**

## EC2 以外の GPU ホストで走らせる

判定は変わらず一行 —— **問題文が自分の管理下のプロセスの外に出るか**
([CLAUDE.md](../CLAUDE.md))。GPU を時間借りして**自分で Ollama を動かす**形は
EC2 と同じであり、この基準に抵触しない。マネージド推論 API は EC2 のときと同様に使わない。

**ただし借り先の型で手当てが変わる。**

| 借り先の型 | 例 | 手当て |
|---|---|---|
| **素の VM**(sudo + systemd) | Lambda Labs 等 | **そのまま。**`10-bootstrap.sh` は無改造で通る |
| **貸しコンテナ**(root・systemd 無し) | RunPod 等 | `10-bootstrap.sh` が自動で `ollama serve` 直起動に切り替える |
| **共同ホスト型** | 第三者が物理的にディスクへ触れる形態 | **使わない。** 非公開シードと HOLDOUT をそこに置けない |

最後の行が選定基準である。**安さではなく、誰がディスクに触れるかで選ぶ。**

**GPU の要件は `10-bootstrap.sh` が確認する** —— VRAM 12GB 未満なら止まる
(13B の Q4_K_M が約 8.4GB。harness はモデルを逐次評価する `harness.py:170` ので
ピークは最大の1本ぶん)。24GB 級(L4 / A10 / L40S / RTX 6000)なら足りる。

### 借り先: **Lambda**(2026-08-06 決定)

**素の Ubuntu VM(`ubuntu` ユーザ + sudo + systemd・ドライバ導入済み)を選んだ。**
RunPod のほうが安い(L4 $0.39/h 対 RTX 6000 $0.69/h)が、貸しコンテナなので
**systemd 無しの分岐**という実機未検証の経路を通る。差額は 8 時間で $3 —— 
**検証済みの経路を通れることのほうが安い。**

RunPod を選び直す場合は、`/workspace`(ネットワークボリューム)の**外**に
リポジトリとキャッシュを置かないこと。コンテナディスクは停止で消える ——
**Spot で EBS ごと消える件と同じ構造の罠**である。

#### 機種は **A100 40GB SXM4 ×1**(2026-08-06・在庫で確定)

RTX 6000 は在庫が無かった。選べたのは H100 80GB(SXM5 $4.29 / PCIe $3.29)・
A10 24GB $1.29・**A100 40GB SXM4 $1.99**・A100 8枚構成のみ。

**A10($1.29)ではなく A100($1.99)を取った。時間単価ではなく総額で安いからである。**
`OLLAMA_NUM_PARALLEL=1` を強制しているのでバッチが効かず、速度はほぼメモリ帯域で決まる。
A10 は約 600 GB/s、A100 SXM4 は約 1,555 GB/s —— **単価 1.5 倍に対して 2.5 倍前後速い。**
本番 DEV と HOLDOUT で1万数千コールあるので、この差は数時間の壁時計時間になる。

H100 は要らない。8B/13B の Q4(最大 8.4GB)に 80GB は過剰で、単価差を速度で取り返せない。

**環境タグは `lambda-a100-40gb-20260806`。** GPU が違えば別環境である。

#### 手元での準備(課金前)

```powershell
# ① bundle を作り直す(コミットを足したなら必須)
git bundle create C:\Users\kingo\projects\contamlab.bundle --all
git bundle verify C:\Users\kingo\projects\contamlab.bundle

# ② Lambda に登録する公開鍵。**AWS 用に作った鍵をそのまま使い回せる**
#    (ed25519。秘密鍵は C:\Users\kingo\.ssh\contamlab.pem のまま動かさない)
ssh-keygen -y -f C:\Users\kingo\.ssh\contamlab.pem
#    → 出力を Lambda のコンソールの SSH keys に貼る
```

#### 起動と持ち込み(ここから課金)

```powershell
# <IP> は Lambda のコンソールが表示する
scp -i C:\Users\kingo\.ssh\contamlab.pem C:\Users\kingo\projects\contamlab.bundle ubuntu@<IP>:~
ssh -i C:\Users\kingo\.ssh\contamlab.pem ubuntu@<IP>
```

#### ホスト内

```bash
git clone contamlab.bundle contamlab && cd contamlab
bash scripts/10-bootstrap.sh
bash scripts/20-rebuild-benchmark.sh

# ★ 環境タグを明示する。EC2 の外では自動生成が "host-<GPU名>-<日付>" になるので、
#   提供者が分かる名前を自分で付けたほうが来歴として読める。
CONTAMLAB_ENV_TAG=lambda-a100-40gb-20260806 bash scripts/30-record-environment.sh

#   → reports/environment.<tag>.md を preregister の「実行環境」枠に貼ってから次へ
bash scripts/35-select-format.sh    # ★ 出力書式を確定(2026-08-07 追加)
bash scripts/40-pilot.sh 1
bash scripts/50-check-determinism.sh
bash scripts/40-pilot.sh 2
bash scripts/60-production.sh dev
bash scripts/60-production.sh holdout   # ★ 1構成・1回だけ
```

> [!warning] Lambda 固有の注意
> - **SSH は AWS の SG のような絞り込みが既定で無い。** ただし
>   `10-bootstrap.sh` が `OLLAMA_HOST=127.0.0.1` を立てるので、**11434 は外に開かない。**
>   問題文を投げる口がホストの外に出ないことが要件であり、そこは満たしている
> - **インスタンスを消すとディスクも消える。** 撤収前に `reports/` と `data/cache/` を
>   回収する(下記「撤収」)。**永続ファイルシステムを付けても、HOLDOUT と
>   非公開シードはそこに置かない**
> - 在庫が薄い。目当ての GPU が埋まっていたら **VRAM 24GB 以上**であれば別型でよい。
>   ただし**型が変われば環境タグも変える**(キャッシュを混ぜない)
> - 起動時に「売上税の規定に準拠していません」で弾かれることがある。GPU の在庫でも鍵でもなく
>   **請求先住所が税計算に足りていない**。国を Japan に揃え、番地までローマ字で埋めて保存し直す

**撤収は EC2 のときと同じ**(下記「撤収」)。`reports/` と `data/cache/` を回収してから
インスタンスを消す。**S3 にも借り先のストレージにも置かない。**

## 再開手順(AWS の枠が通った場合・2026-08-05 時点の実値)

下ごしらえ(IAM ユーザ・鍵ペア・SG・bundle)は**済んでいる。**
**待ちは EC2 の G/VT クォータの承認**で、承認されるまで起動は
`VcpuLimitExceeded` で弾かれる。**まず 0 を確認する。**

```powershell
$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

# ① クォータ。4.0 が返るまで先へ進まない(0.0 なら承認待ち)
& $aws service-quotas get-service-quota --service-code ec2 `
    --quota-code L-DB2E81BA --region ap-northeast-1 --query "Quota.Value" --output text

# ② bundle を作り直す(コミットを足したなら必須。足していないなら不要)
git bundle create C:\Users\kingo\projects\contamlab.bundle --all

# ③ 起動。まず表示だけ(課金しない)→ 内容を見てから -Execute
cd C:\Users\kingo\projects\contamlab
.\scripts\00-launch-ec2.ps1 -KeyName contamlab -SecurityGroupId sg-0871cee36a8152bc6
.\scripts\00-launch-ec2.ps1 -KeyName contamlab -SecurityGroupId sg-0871cee36a8152bc6 -Execute

# ④ 転送して入る(<IP> は ③ が表示する)
scp -i C:\Users\kingo\.ssh\contamlab.pem C:\Users\kingo\projects\contamlab.bundle ubuntu@<IP>:~
ssh -i C:\Users\kingo\.ssh\contamlab.pem ubuntu@<IP>
```

インスタンス内:

```bash
git clone contamlab.bundle contamlab && cd contamlab
bash scripts/10-bootstrap.sh          # ドライバ無し AMI なら一度終了する → 再起動して再実行
bash scripts/20-rebuild-benchmark.sh
bash scripts/30-record-environment.sh
#   ★ ここで reports/environment.<tag>.md を preregister の「実行環境」枠へ貼る。
#     貼るまで 40 へ進まない(下の「3. 実行環境」)。
bash scripts/35-select-format.sh    # ★ 出力書式を確定(2026-08-07 追加)
bash scripts/40-pilot.sh 1
bash scripts/50-check-determinism.sh
bash scripts/40-pilot.sh 2
bash scripts/60-production.sh dev
bash scripts/60-production.sh holdout  # ★ 1構成・1回だけ
```

> [!warning] 手元の環境で引っかかる2点
> - **`aws` は PATH に無い場合がある。** `C:\Program Files\Amazon\AWSCLIV2\aws.exe` を直に呼ぶ。
>   ただし `00-launch-ec2.ps1` は `Get-Command aws` を要求するので、**PATH が通った
>   PowerShell から起動する**(通っていなければ新しいウィンドウを開き直す)。
> - **SG はグローバル IP の /32 だけを許可している。** 回線が変わると SSH が通らない。
>   その場合は現在の IP で規則を足し直す。
>
> 実験が終わったら **IAM アクセスキーを無効化する。**

## ★ 実行前に preregister へ追記が要る変更

**結果を見る前に書くこと。** 書かずに走らせたら、その数字は事前確約の外になる。

| | 状態(2026-08-05 確認) |
|---|---|
| 1. パイロット 50/200 → 70/250 | ✅ **記載済み。** preregister「パイロットの設計」節 + 変更履歴 2026-08-04 |
| 2. パイロット③を①②に統合 | ✅ **記載済み。** 同上 |
| 3. 実行環境 | ❌ **未記入。** `30-record-environment.sh` を走らせないと書けない。preregister の「実行環境」節に枠を用意してある |

以下は各項目の根拠(記載済みのものも、何を確約したかを取り違えないために残す)。

### 1. パイロットの問題数を 50 / 200 → **70 / 250** に上げた ✅ 記載済み

01 のパイロットは ψ=0.30 を想定して n を決めた。しかしパイロット②で ψ̂=0.335 と
その Clopper-Pearson 上側限界 **0.4050 を既に知っている。** 知っていながら 0.30 と
宣言し直すのは、通るように想定を選ぶことである。0.4050 で宣言すると:

| パイロット | 狙う効果量 | ψ=0.30 | **ψ=0.4050** |
|---|---|---|---|
| ① | 20 pt | 必要 45 問 → n=50 で可 | 必要 **61 問** → n=50 では**不足**。→ **70 問** |
| ② | 10 pt | 必要 184 問 → n=200 で可 | 必要 **249 問** → n=200 では**不足**。→ **250 問** |

どちらも `--force-underpowered` を使わずにゲートを通る。`take_deterministic` は
prefix 安定(`benchmark.py:227`)なので **70 ⊂ 250 ⊂ 本番 1,270** となり、
応答キャッシュは1問も無駄にならない。n=70 は preregister が既に宣言している
パイロット③の設計値と同じである。

### 2. パイロット③(モデル別の採点健全性)を①②に統合した ✅ 記載済み

01 ではパイロット①②を1本のモデルで回したあと、③でロースター3本の採点健全性を
別に確かめる設計だった。02 ではロースターが2本なので、**①②を最初からロースター
2本で回せば③の目的(モデル別の解釈不能率と ψ)は同時に満たされる。**
`40-pilot.sh` は分割表からモデル別の ψ を計算して表示する。

### 4. 出力書式を候補3つから選び直す(2026-08-07 追加)✅ 記載済み

preregister「ラン: jmmlu-shuffle-03」+ 変更履歴 2026-08-07。**何も測る前に書いた。**

**動機は結果を見た後**である(02 で `llmjp3-13b` が書式で脱落した)ことを明記したうえで、
「落ちたモデルが通る条件を落ちたのを見てから選ぶ」ことにならないための装置を4つ置いた:
**選定指標は解釈不能率だけ**(`35-select-format.sh` の要約は `accuracy_*` も `drop` も
`p_value` も**コードに出てこない**)/ **摂動器は `identity` 固定**で摂動後の応答を1件も
見ない / **選定は DEV・検定は HOLDOUT** で統計量が別データ / **候補3書式を凍結し、
合格ゼロなら4つ目を作らず本番を実行しない。**

**同時に動かさないもの**: 採点器(`select`)は凍結。3書式とも既存の規則2・規則3で読める。
**失効する値**: 想定 ψ=0.4050 と必要問題数 1,270 は書式 A の測定値なので引き継がない
(パイロット②で測り直す。写像表は書式に依存しないのでそのまま使う)。

### 3. 実行環境 ❌ **未記入 —— 残っているのはこれだけ**

`30-record-environment.sh` が生成する `reports/environment.<tag>.md` を
preregister の `jmmlu-shuffle-02` 節「実行環境」の枠に**そのまま貼る。**
Ollama の版・GGUF の SHA256・GPU・`OLLAMA_NUM_PARALLEL=1` などが入っている。

**`30` の直後・`40-pilot.sh 1` の前に貼る。** 後から書くと「実際に使った環境の記録」ではなく
「結果を見たあとの記述」になる。バックエンドは測定条件なので、**確定してから測る。**

## 設計上の要点

### 応答キャッシュは環境ごとに分ける

キャッシュのキーは**モデル名とプロンプトだけ**(`runner.py:180`)で、ハードウェアも
バックエンドも入っていない。しかも `CachedModel.answer` はキャッシュに当たれば
**モデルを呼ばずに帰る**(`runner.py:223`)。CPU 時代のキャッシュを持ったまま GPU で
走らせると、ログ上は「GPU で N コール完了」に見えるのに **GPU は一度も呼ばれず、
古い答えがそのまま返る。** `put()` に到達しないので `conflicts` も立たない。

対応: `30-record-environment.sh` が確定する**環境タグ**でキャッシュ名を分ける
(`data/cache/responses.<tag>.jsonl`)。CPU 時代のものは
`responses.cpu-laptop-20260804.jsonl` に改名済みで、既定パスは空になっている。

**タグの取り違えを2箇所で塞いだ(2026-08-06)。**

- **タグが無いまま先へ進めない。** `env_tag` は `$(...)` の中で呼ばれるので、そこで
  `exit 1` しても死ぬのは副シェルだけで、呼び出し元は
  `data/cache/responses..jsonl` という**もっともらしい別ファイル**を受け取ったまま
  走り出していた。40 / 50 / 60 の冒頭で `require_env_tag` を呼んで止める
- **EC2 の外で "ec2-" を名乗らない。** タグの自動生成が無条件に `ec2-` を前置していたので、
  借りた GPU ホストで走らせると**嘘の実行環境が preregister に載る。**
  IMDS が答えたときだけ `ec2-`、それ以外は `host-<GPU名>-<日付>` にした
- **設定を書いた≠効いている。** `10-bootstrap.sh` は `ollama serve` の `/proc/<pid>/environ`
  を読んで `OLLAMA_NUM_PARALLEL=1` 等が実プロセスに入っているか確かめる。
  古い `ollama serve` が生き残っていると、設定ファイルは正しいのにプロセスは古い環境で動く

### 決定性は宣言せず実測する

`temperature 0` は「最も確率の高い選択肢を選ぶ」であって「**計算結果が同じになる**」
ではない。CPU と GPU では並列化が違うので浮動小数の加算順序が違い、1位と2位が僅差の
問題で順位が入れ替わりうる。**そしてそれが起きるのはモデルが迷っている問題**であり、
`shuffle_choices` で答えが変わるのも**モデルが迷っている問題**である。ノイズが乗る
集合と測りたい集合が重なる。

同じキャッシュで2回目を回しても測れない(再生されるだけ)。`50-check-determinism.sh` は
**別のキャッシュファイルに独立に取り直して**、生の応答文字列と採点結果の両方を
突き合わせる。

`10-bootstrap.sh` が `OLLAMA_NUM_PARALLEL=1` を立てるのも同じ理由である。並列実行は
リクエストをバッチにまとめるので、バッチの組み方で加算順序が変わりうる。contamlab は
逐次に問い合わせるので、並列度を落としても速度は落ちない。

### ベンチマークは pin した SHA から作り直す

`data/jmmlu.jsonl` は `.gitignore` されているので新しい機械では作り直すしかない。
ところが `tools/ingest_jmmlu.py` の `ensure_clone` は **`git clone` を HEAD に対して行う。**
JMMLU 側に1コミットでも追加されていれば件数が変わり、`item.id` が変わり、
`split_dev_holdout` の帰属が変わり、**DEV と HOLDOUT の分割が変わる。**
しかも**エラーは出ない。静かに壊れる。**

`20-rebuild-benchmark.sh` は clone の直後に `762cbf19` を明示的に checkout し、
生成した manifest がコミット済みのものと**完全一致**するか、DEV 4,742 /
HOLDOUT 1,922 が再現するかを確かめる。一致しなければそこで止まる。

### Spot は使わない・ボリュームは 80GB(2026-08-05 判断)

`00-launch-ec2.ps1` の `--block-device-mappings` は **`DeleteOnTermination=true`** である。
一方 Spot の中断動作の既定は **terminate** なので、**中断された瞬間に EBS ごと消える** ——
応答キャッシュも環境タグも `reports/` も残らない。「キャッシュが追記専用だから中断されても
再開できる」は**この構成では成り立たない。** 作り直しに払う GPU 時間のほうが Spot の差額
(1ラン数ドル)より高い。Spot を使うなら先に `DeleteOnTermination=false` と
`InterruptionBehavior=stop` を入れること。**`-Spot` は残してあるが警告を出す。**

`-VolumeGb` の既定は 200 → **80** に下げた。実需は OS 約15GB + GGUF 約13GB
(13B と 8B の Q4_K_M)+ JMMLU とキャッシュで数百MB ≒ **40GB**。gp3 は
**インスタンスを止めていても $0.1/GB/月** 掛かるので、200GB は月 $19 の払い過ぎになる。

### α は 0.05 のまま渡す

M=2 の実効水準 0.0250 を `--alpha` に渡してはいけない。信頼区間と Cochran の Q まで
二重補正になる。Holm 補正は harness が判定側(`harness.py:101`)で行う。
**問題数だけを ψ の写像表から自分で決める。** これは検出力ゲートが Holm を
織り込んでいない既知の穴(`harness.py:139`)への対応でもある。

## 撤収

**terminate する前に回収する。**

```powershell
scp -i <鍵> -r ubuntu@<IP>:contamlab/reports ./reports-ec2
scp -i <鍵> -r ubuntu@<IP>:contamlab/data/cache ./data/cache-ec2
```

応答キャッシュを捨てると、次に同じ問いを投げるのに GPU 時間をもう一度払うことになる。
**追記専用なので消さない**(`program.md` の禁止事項)。

**S3 には置かない。** 非公開シードと HOLDOUT はインスタンス内に閉じる。
