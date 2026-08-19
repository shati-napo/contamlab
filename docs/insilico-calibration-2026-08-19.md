# FakeModel での in-silico 較正 — 結果(2026-08-19)

> 実施: 作業3(docs/NEXT.md の着手ブロック)。費用 **$0**・GPU **0 台**・API 呼び出し **0 回**。
> 実行体: [tools/insilico_calibration.py](../tools/insilico_calibration.py)
> 生データ: `reports/insilico/insilico-calibration-20260819.json`
> ⛔ **`reports/` は .gitignore 対象**なのでクローン先には無い。下のコマンドで再現できる(seed 固定):
> `python tools/insilico_calibration.py --json reports/insilico/insilico-calibration-20260819.json`
> ⛔ `contamlab/` は**1行も触っていない**。呼んだだけ。

---

## ⛔ まずこれを読むこと —— これは較正であって測定ではない

**FakeModel は「暗記した問題はオリジナル提示形のときだけ必ず正解し、摂動版では素の能力に落ちる」
という汚染の定義そのものを実装したものである。**したがってここで較正曲線が出るのは**当たり前**であり、
発見ではない。

★ **この文書の数字を成果として引用しない。** 言えるのはただ1つ ——

> **`drop` / `p_value` / `adjusted_lcb` / `p_holm` / `detected` / Cochran の Q が、
> 6アームの設計で配管として最後まで出てくることを、GPU に金を払う前に確認した。**

⛔ 「検出器が実モデルで効く」ことには**一切答えていない。**
`scripts/70-positive-control.sh` は**依然として一度も走っていない。**

---

## なぜ要ったか

12 ラン通じて **`drop` / `p_value` / `adjusted_lcb` は実モデルに対して1つも計算されたことがない**
(docs/NEXT.md の現在地)。**この3つを出す経路が生きているかどうかすら確かめていない**まま
GPU を借りるのは順序が逆である。

## 設計(`scripts/70-positive-control.sh` と土俵を揃えた)

| 項目 | 値 | 出所 |
|---|---|---|
| 問題数 | 4,742 | pc-01 の DEV 全量と同数 |
| アーム | 0 / 2 / 5 / 10 / 20 / 40 %(6本) | `PC_ARMS` と同じ梯子 |
| 摂動器 | `shuffle_choices`(K=1) | 同上。⛔ **HOLDOUT は開けていない** |
| 想定 ψ | 0.4050 | パイロット②の採用値 |
| 狙う効果量 | 0.05 | 同上 |
| α | 0.008333(M=6 の Holm 実効値) | 同上 |
| 素の正答率 | 0.45 | FakeModel の設定値 |
| 暗記集合 | アーム間で**入れ子**(x02 ⊂ x05 ⊂ …) | `tools/build_injection_sets.py` と同じ作り |
| seed | 20260819 | — |

## 結果

実測 ψ = 0.4796 / 事前の最小検出可能 = 2.99 pt / 実測 ψ での検出力 = 0.9951

| アーム | 注入率 | drop | lcb | 割引後下限 | p_value | p_holm | 検出 |
|---|---|---|---|---|---|---|---|
| sim-x00 | 0% | −0.30 pt | −2.74 pt | −0.0274 | 0.6225 | 0.6225 | — |
| sim-x02 | 2% | +2.57 pt | +0.14 pt | +0.0014 | 5.7e−03 | 1.1e−02 | — |
| sim-x05 | 5% | +3.82 pt | +1.44 pt | +0.0144 | 5.9e−05 | 1.8e−04 | ★ |
| sim-x10 | 10% | +6.73 pt | +4.33 pt | +0.0433 | 8.7e−12 | 3.5e−11 | ★ |
| sim-x20 | 20% | +12.06 pt | +9.69 pt | +0.0969 | 1.2e−33 | 6.0e−33 | ★ |
| sim-x40 | 40% | +20.52 pt | +18.24 pt | +0.1824 | 7.4e−93 | 4.4e−92 | ★ |

Cochran の Q = 296.6772 / df = 5 / p = 5.2e−62 / I² = 0.9831(不均一)

### 配管の判定 — 7項目すべて通過

| | 項目 |
|---|---|
| ✅ | 全アームで `drop` / `p_value` / `adjusted_lcb` / `p_holm` が数値として出た |
| ✅ | 注入ゼロのアームを検出しない(偽陽性を出さない) |
| ✅ | 注入 40% のアームを検出する |
| ✅ | `drop` が注入率について単調非減少 |
| ✅ | Holm 補正が効いている(p_holm ≥ p_value) |
| ✅ | K 割引が effect を必ず削る(adjusted_lcb ≤ lcb) |
| ✅ | Cochran の Q が計算された |

### CLI 経路の疎通も別に確認した

```powershell
python -m contamlab run --synthetic 2000 --seed insilico-cli `
  --perturbator shuffle_choices --target-effect 0.05 `
  --expected-discordant-rate 0.405 --k 1 --yes `
  --model fake:cli-clean:0.45 --model fake:cli-dirty:0.45:memorized `
  --json reports/insilico/cli-smoke-20260819.json
```

汚染なし +1.8 pt(判定 —)/ 全問暗記 +52.4 pt(判定 ★汚染)/ Q=694.77・p<1e−4。
**API 呼び出し 0 回**(FakeModel はキャッシュにも入らない設計)。

---

## 途中で1つ確かめられたこと(✅)

最初に n=800 で走らせたところ、**検出力ゲートが `UnderpoweredError` を投げて止めた**:

> 検出力が足りない。5.0 ポイントの汚染を検出力 0.80 で見るには 1694 問要るが、手元には 800 問しかない。

★ **README が「その設計を実行前に拒否する」と書いている挙動が、実際に発火することを確認した。**
⛔ ただしこの 1,694 という数字自体が、[docs/power-verify-2026-08-19.md](power-verify-2026-08-19.md) の
**欠陥 P-1(正規近似と厳密検定の食い違い)によって楽観側にずれている**ことに注意。
ゲートは**発火するが、少し甘い**。

---

## ⛔ この較正が**言っていないこと**(踏み越えない)

- ⛔ 実モデルで汚染が検出できるかどうか。**何も言っていない**
- ⛔ 上の「2% では検出せず 5% で検出」という帯は **FakeModel の帯**であり、
  実モデルの検出下限とは**無関係**である。★ **較正曲線を引いたとは言わない**
- ⛔ `70-positive-control.sh` は**まだ一度も走っていない**。この状況は本ランで変わっていない
- ⛔ FakeModel は「摂動で記憶が完全に消える」という**最も都合のよい汚染**を仮定している。
  実物の汚染がこの形をしている保証はない
