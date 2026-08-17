#!/usr/bin/env python3
"""finetune/prepare_cc01_arms.py — ラン calibration-curve-01 の注入集合を用意する。

    python finetune/prepare_cc01_arms.py                  # ① 複製(依存なし・どこでも走る)
    python finetune/prepare_cc01_arms.py --measure-tokens # ② 実測(tokenizer が要る)

preregister「ラン: calibration-curve-01」→「着手手順 1」の実装。
**このスクリプトは規則を決めない。何も生成しない。pc-01 の成果物を複製するだけである。**

★ 用意するのは **36本**である ——
    `cc1t<k>-x<rr>`      学習 18本(注入率6水準 × 複製3本)
    `cc1L08t<k>-x<rr>`   λ=0.8 適用後 18本(**学習しない。**scale_adapter.py が作る)

★ **本ランは pc-01 以来はじめて注入率を振る。**pc-04 以降のランはすべて `-x40` の
  1水準しか使っておらず、**器がアーム名の末尾2桁から注入集合を引く経路**
  (`train_lora.py` の注記 —— `arm[-2:]`)は眠っていた。
  ⛔ **6水準すべてで正しく引けることを、学習を始める前にここで確かめる**(停止条件 3)。

★ 中身は pc-01 の `pc-x<rr>` と**バイト単位で同一でなければならない。**
  1バイトでも違えば、水準の間の差に「注入集合の違い」が混ざる。
  さらに `x40` は pc-04・pc-06・td-01・ll-01 が実際に測った集合そのものなので、
  **ll-01 の結果を較正曲線の 40% 点として読むための前提**でもある。

---------------------------------------------------------------------------
② --measure-tokens が何をするか(★ これは測定であって規則ではない)
---------------------------------------------------------------------------
**注入トークン数はアームごとに違う。**従来の `INJECTED_TOKENS_ONCE_BY_RUN`
(ランごとに1つ)では表せないので、**6水準ぶんを学習の前にまとめて実測**し、
`data/injection/manifest-cc01.json` に凍結する。`train_lora.py` はそこから引く。

★ **これは選択ではない。**凍結済みのベンチマーク(sha256 `8aa877e5…`)・
  凍結済みの注入集合・凍結済みの tokenizer revision から決まる決定論的な量である。
  ⛔ 「走らせながら1つずつ決める」ことはしない —— アームの間で状態が違っても気付けない。

★ **検算の錨**: `x40` は **238,082** でなければならない(pc-04 / td-01 / ll-01 の実測値)。
  1つでも既知の値と合えば、残り5つを生んだ計算経路も正しい。
  ⛔ 合わなければ tokenizer か注入集合が過去ランと違う。**書き出さずに止める。**

★ 単調性も確かめる —— 注入集合は入れ子なので、トークン数は水準とともに
  **狭義単調増加**でなければならない(0% を除く)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_lora import (CC01_ANCHOR_RATE, CC01_ANCHOR_TOKENS,  # noqa: E402
                        CC01_N_INJECTED, CC01_RATES, CC01_RUN, CC01_TOTAL_TOKENS_T,
                        CC01_TRAIN_ARMS, RECIPES, RUN_BASES)

RUN = CC01_RUN
SUFFIXES = (".jsonl", ".ids")
EXPECTED_N_ARMS = 36          # 学習 18本 + λ=0.8 の 18本
EXPOSURES_E = RECIPES["R1"]["E"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cc01_arms() -> dict[str, str]:
    """アーム → 複製元(`pc-x<rr>`)。名前は凍結表から引く(手で打ち直さない)。"""
    from scale_adapter import CC01_LAMBDA_ARMS

    train = list(CC01_TRAIN_ARMS.values())
    scaled = list(CC01_LAMBDA_ARMS.values())
    arms = [*train, *scaled]
    if len(set(arms)) != len(arms):
        raise SystemExit(f"★ 凍結表のアーム名が重複している: {arms}")
    if len(arms) != EXPECTED_N_ARMS:
        raise SystemExit(f"★ 用意するアームが {len(arms)} 本 ≠ {EXPECTED_N_ARMS} 本")
    # ★ 末尾2桁が注入率として読まれるので、**名前の末尾と複製元が一致していること**が
    #   このスクリプトの成立条件そのものである。ここで組み立てて、下で照合する。
    mapping = {arm: f"pc-x{arm[-2:]}" for arm in arms}
    if {a[-2:] for a in arms} != set(CC01_RATES):
        raise SystemExit(f"★ アーム名の末尾2桁が凍結表の6水準と揃わない: "
                         f"{sorted({a[-2:] for a in arms})}")
    return mapping


def copy_arms(d: Path) -> int:
    arms = cc01_arms()
    print(f"アーム名は凍結表から引いた({len(arms)} 本 / {len(CC01_RATES)} 水準)")

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run"] != "positive-control-01":
        print(f"★ 複製元の manifest が pc-01 のものでない: {manifest['run']}")
        return 1
    expected = {a["name"]: a for a in manifest["arms"]}

    # --- 1. 複製元が pc-01 の記録と一致しているか(複製する前に確かめる) -----------
    for rate in CC01_RATES:
        src = f"pc-x{rate}"
        if src not in expected:
            print(f"★ pc-01 の manifest に {src} が無い。")
            return 1
        want = expected[src]
        got_txt, got_ids = sha256(d / f"{src}.jsonl"), sha256(d / f"{src}.ids")
        if got_txt != want["txt_sha256"] or got_ids != want["ids_sha256"]:
            print(f"★ 複製元 {src} が pc-01 の manifest と違う。**複製してはいけない。**")
            print(f"    jsonl 実測 {got_txt}\n          記録 {want['txt_sha256']}")
            print(f"    ids   実測 {got_ids}\n          記録 {want['ids_sha256']}")
            return 1
        if want["n_injected"] != CC01_N_INJECTED[rate]:
            print(f"★ 複製元 {src} の n_injected が {want['n_injected']} ≠ "
                  f"{CC01_N_INJECTED[rate]}(凍結表)。**複製してはいけない。**")
            return 1
        print(f"複製元 {src}: pc-01 の manifest と一致({want['n_injected']:,d} 問)")

    # --- 2. ★ 入れ子を確かめる(pc-01「入れ子にする理由」)-------------------------
    #   入れ子が壊れていれば、アーム間の差は「注入率の差」ではなくなる。
    ids = {r: set((d / f"pc-x{r}.ids").read_text(encoding="utf-8").split())
           for r in CC01_RATES}
    ladder = [r for r in CC01_RATES if r != "00"]
    for lo, hi in zip(ladder, ladder[1:]):
        if not ids[lo] <= ids[hi]:
            print(f"★ 入れ子が壊れている: x{lo} ⊄ x{hi}。**複製してはいけない。**")
            return 1
    print(f"入れ子: {' ⊂ '.join('x' + r for r in ladder)} —— 成立")

    # --- 3. 複製し、複製後に照合する --------------------------------------------
    records = []
    for arm, src in arms.items():
        want = expected[src]
        for suffix in SUFFIXES:
            s, t = d / f"{src}{suffix}", d / f"{arm}{suffix}"
            shutil.copyfile(s, t)
            if sha256(t) != sha256(s):
                print(f"★ 複製後の sha256 が合わない: {t}")
                return 1
        records.append({
            "name": arm, "copied_from": src,
            "injection_rate": want["injection_rate"], "n_injected": want["n_injected"],
            "chars": want["chars"],
            "txt_sha256": want["txt_sha256"], "ids_sha256": want["ids_sha256"],
        })

    out = d / "manifest-cc01.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    payload = {
        "run": RUN,
        "note": "pc-01 の注入集合をアーム名だけ変えて複製したもの。中身は生成していない。"
                "★ 本ランは pc-01 以来はじめて注入率を振るランであり、"
                "6水準(0/2/5/10/20/40%)× 複製3本 = 18本を学習する。"
                "cc1L08t<k>-x<rr> の 18本は学習しない —— 18本のアダプタそれぞれから "
                "α → 0.8·α でマージし直して作る。"
                "★ λ は 0.8 の1段だけである(ll-01 が合格させた唯一の段)。"
                "本ランが動かすのは注入率であって λ ではない。"
                "★ T は全アーム共通の固定値 8,570,952 で、差は埋め草で埋める —— "
                "そうしないと「注入率が上がった」のか「長く学習した」のかが区別できない。",
        "source_run": manifest["run"],
        "inject_salt": manifest["inject_salt"], "split": manifest["split"],
        "dev_size": manifest["dev_size"], "template_sha256": manifest["template_sha256"],
        "total_tokens_t": CC01_TOTAL_TOKENS_T,
        "exposures_e": EXPOSURES_E,
        "arms": records,
    }
    # ★ 実測表は消さない。--measure-tokens をやり直さずに複製だけやり直せるようにする。
    if previous.get("injected_tokens_once"):
        payload["injected_tokens_once"] = previous["injected_tokens_once"]
        payload["tokenizer"] = previous.get("tokenizer")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")

    print(f"\n{len(arms)} 本を用意した。sha256 は pc-01 と完全に一致している。")
    print(f"記録: {out}")
    if not payload.get("injected_tokens_once"):
        print("\n★ 次: `python finetune/prepare_cc01_arms.py --measure-tokens`"
              "(注入トークン数を6水準ぶん凍結する。学習の前に必ず走らせる)")
    return 0


def measure_tokens(d: Path) -> int:
    """6水準の注入トークン数を実測して凍結する。★ 学習を1本も始める前に。"""
    from transformers import AutoTokenizer

    out = d / "manifest-cc01.json"
    if not out.is_file():
        print(f"★ {out} が無い。先に複製(引数なしの実行)を済ませること。")
        return 1
    manifest = json.loads(out.read_text(encoding="utf-8"))

    base_model, base_revision = RUN_BASES[RUN]
    print(f"tokenizer: {base_model} @ {base_revision}")
    tok = AutoTokenizer.from_pretrained(base_model, revision=base_revision)

    table: dict[str, int] = {}
    for rate in CC01_RATES:
        path = d / f"pc-x{rate}.jsonl"
        texts = ([json.loads(l)["text"]
                  for l in path.read_text(encoding="utf-8").splitlines()]
                 if path.stat().st_size else [])
        if len(texts) != CC01_N_INJECTED[rate]:
            print(f"★ x{rate} のレコード数が {len(texts)} ≠ {CC01_N_INJECTED[rate]}(凍結表)。")
            return 1
        # ★ 数え方は train_lora.py と同一でなければならない —— **内容トークンのみ**
        #   (末尾 EOS はレコードの区切りであって内容ではない)。
        table[rate] = sum(len(tok.encode(t)) for t in texts)
        share = table[rate] * EXPOSURES_E / CC01_TOTAL_TOKENS_T
        print(f"  x{rate}  {CC01_N_INJECTED[rate]:>5,d} 問  "
              f"{table[rate]:>9,d} tok  → 学習信号の {share:6.2%}"
              f"(埋め草 {1 - share:6.2%})")

    # --- 錨 ---------------------------------------------------------------
    if table[CC01_ANCHOR_RATE] != CC01_ANCHOR_TOKENS:
        print(f"\n★ 錨が合わない —— x{CC01_ANCHOR_RATE} が {table[CC01_ANCHOR_RATE]:,d} != "
              f"{CC01_ANCHOR_TOKENS:,d}(pc-04 / td-01 / ll-01 の実測値)。")
        print("  ⛔ tokenizer か注入集合が過去ランと違う。**書き出さずに止める。**")
        return 1
    print(f"\n錨: x{CC01_ANCHOR_RATE} = {CC01_ANCHOR_TOKENS:,d} tok —— "
          "pc-04 / td-01 / ll-01 の実測値と一致")

    # --- 単調性 -----------------------------------------------------------
    ladder = [r for r in CC01_RATES if r != "00"]
    if any(table[lo] >= table[hi] for lo, hi in zip(ladder, ladder[1:])):
        print(f"★ トークン数が水準とともに増えていない: "
              f"{[(r, table[r]) for r in CC01_RATES]}")
        print("  ⛔ 注入集合が入れ子でない。**書き出さずに止める。**")
        return 1
    if table["00"] != 0:
        print(f"★ x00 の注入トークンが 0 でない({table['00']:,d})。")
        return 1
    print("単調性: 0 < x02 < x05 < x10 < x20 < x40 —— 成立")

    # --- T に収まるか ------------------------------------------------------
    over = [r for r in CC01_RATES if table[r] * EXPOSURES_E > CC01_TOTAL_TOKENS_T]
    if over:
        print(f"★ 注入だけで T = {CC01_TOTAL_TOKENS_T:,d} を超える水準がある: {over}")
        return 1

    manifest["injected_tokens_once"] = table
    manifest["tokenizer"] = {"model": base_model, "revision": base_revision}
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\n6水準の注入トークン数を凍結した: {out}")
    print("★ preregister の「実行環境」枠に上の表を貼ってから学習へ進むこと。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-dir", type=Path, default=Path("data/injection"))
    ap.add_argument("--measure-tokens", action="store_true",
                    help="★ 6水準の注入トークン数を実測して凍結する(tokenizer が要る)")
    args = ap.parse_args()
    if args.measure_tokens:
        return measure_tokens(args.injection_dir)
    return copy_arms(args.injection_dir)


if __name__ == "__main__":
    raise SystemExit(main())
