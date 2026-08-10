# finetune/ — 陽性対照の汚染モデルを作る

**ここは contamlab パッケージの外である。** `torch` / `transformers` / `peft` を使うので、
**`pyproject.toml` には一切触らない。** contamlab 側は出来上がった GGUF を Ollama 経由で
叩くだけで、学習系を import しない(preregister「fine-tune のコードは contamlab の依存に入れない」)。

設計・注入率・判定規則は [preregister.md](../preregister.md) が正。
**ここは実装であって、規則を決めない。**

> [!important] ★ **ランごとの実行順は preregister の該当節「箱の上での手順」が正である。**
> 下の「実行順」は **ラン `positive-control-01` のもの**で、以後のランでは変わっている
> (ベース・アーム名・梯子の指定方法)。**現在地は [docs/NEXT.md](../docs/NEXT.md)。**
>
> | ラン | 梯子の指定 | アーム | ベース |
> |---|---|---|---|
> | pc-01 | `--arm pc-x40` | `pc-x00`〜`pc-x40` | Qwen2.5-1.5B |
> | pc-02(未実行) | `--recipe R0..R4 --run positive-control-02` | `pcr*-x40` | Qwen2.5-1.5B |
> | pc-04 | `--recipe R0..R4`(既定が pc-04) | `pc4r*-x40` | Swallow-8B |
> | **pc-05** | **`--run positive-control-05 --stage F1\|F2\|F3`** | **`pc5f*-x40`** | Swallow-8B |
>
> **pc-05 で動くのは埋め草の割合 f だけ**(F1 0.50 / F2 0.75 / F3 0.875)であり、
> **レシピは `R1` に固定**されている(`--recipe` では選べない)。
> `T = 注入トークン × E / (1 − f)` は従属計算で、**f を直接渡す口は無い。**
>
> ```bash
> python finetune/prepare_filler.py                 # ★ 60M トークン・逐語照合つき(約5分)
> python finetune/prepare_pc05_arms.py
> finetune/.venv/bin/python finetune/probe_micro_batch.py --run positive-control-05 --recipe R1
> finetune/.venv/bin/python finetune/train_lora.py --run positive-control-05 --stage F1
> bash finetune/to_gguf.sh pc5f1-x40
> bash scripts/65-manipulation-check.sh pc5f1-x40
> ```
>
> ⚠️ **F3 は条件付きの段である** —— 「F1 → F2 で非注入群の解釈不能率が単調に改善して
> いなければ実行しない」。`train_lora.py` は警告を出すが、**止めるのは人である。**

## 実行順(★ ラン `positive-control-01` のもの。以後のランは上の表を見よ)

```bash
# 0. contamlab 側の下ごしらえ(既存スクリプト)
bash scripts/10-bootstrap.sh
bash scripts/20-rebuild-benchmark.sh          # 6,664 / 4,742 / 1,922 と一致すること
bash scripts/30-record-environment.sh

# 1. 注入集合(標準ライブラリのみ。GPU 不要)
python tools/build_injection_sets.py

# 2. 学習環境と埋め草
python -m venv finetune/.venv && . finetune/.venv/bin/activate
pip install -r finetune/requirements.txt
python finetune/prepare_filler.py             # ★ JMMLU との逐語重複チェックを内蔵

# 3. ★ pc-x40 を最初に1本だけ学習して操作チェックを通す
python finetune/train_lora.py --arm pc-x40
bash finetune/to_gguf.sh pc-x40
bash scripts/65-manipulation-check.sh pc-x40  # ← ここで落ちたら残り5本を作らない

# 4. 残り5本
for a in pc-x00 pc-x02 pc-x05 pc-x10 pc-x20; do
  python finetune/train_lora.py --arm $a && bash finetune/to_gguf.sh $a
done
bash scripts/65-manipulation-check.sh          # 全アーム

# 5. 測定(56,904 コール)
bash scripts/70-positive-control.sh
```

> [!important] ★ なぜ pc-x40 を最初に回すのか
> 操作チェックの停止条件は「**X ≥ 20% のアームで注入群と非注入群の正解率の差が
> 10pt 未満なら停止**」である。40% は注入が最も強いアームなので、**ここが通らなければ
> 他の5本は確実に通らない。** 先に1本だけ作って確かめれば、レシピが外れていたことを
> 10分で知ることができる。6本作ってから知ると GPU を半日捨てる。
>
> ⚠️ **落ちた場合は E や学習率を調整して「同じラン」を続けてはいけない。**
> preregister は「レシピを見直して**別のランとして**やり直す」と書いている。
> 名前は `positive-control-02` になり、事前登録もやり直す。

## 学習量の揃え方(preregister「学習量をアーム間で揃える」の実装)

| 記号 | 意味 | 値 |
|---|---|---|
| E | 注入問題1問あたりの露出回数 | **12**(全アーム同一) |
| T | 総学習トークン数(内容トークン。パディングは数えない) | **2,831,004**(= 40% アームの注入 235,917 × 12) |

各アームの学習コーパスは次のように組む:

```
注入レコード × E 回  +  埋め草(T − 注入トークン×E になるまで)  → 固定シードでシャッフル → 1 epoch
```

**epoch を回すのではなく、コーパスを作る側で E を実現している。** こうしないと
埋め草まで E 回学習してしまい、T がアーム間でずれる。

| アーム | 注入トークン×E | 埋め草トークン | 合計 T |
|---|---|---|---|
| `pc-x00` | 0 | 2,831,004 | 2,831,004 |
| `pc-x02` | 124,824 | 2,706,180 | 2,831,004 |
| `pc-x05` | 330,636 | 2,500,368 | 2,831,004 |
| `pc-x10` | 673,416 | 2,157,588 | 2,831,004 |
| `pc-x20` | 1,407,048 | 1,423,956 | 2,831,004 |
| `pc-x40` | 2,831,004 | 0 | 2,831,004 |

**`pc-x00` は「fine-tune しないモデル」ではない。** 同じレシピ・同じ T トークンを
100% 埋め草で学習させた**陰性対照**である。01〜03 で入手できなかったものがこれ。

## 詰め込み(packing)

固定長 2048 のブロックに**貪欲に詰める。入り切らないレコードは次のブロックへ送り、
余りは EOS で埋めて loss から外す。** レコードが境界をまたがないので、注入した1問が
2つのブロックに割れて記憶が弱まることがない。**T は内容トークンで数えるので、
パディングの量が変わってもアーム間の学習量は揃う。**
