#!/usr/bin/env python3
"""finetune/merge_adapter.py — 学習済みアダプタを**そのまま**マージして複製を1本作る。

    python finetune/merge_adapter.py --replicate 1     # M1
    python finetune/merge_adapter.py --replicate 2     # M2
    python finetune/merge_adapter.py --replicate 3     # M3

preregister「ラン: merge-variance-01」→「段取り」の実装。
**このスクリプトは規則を決めない。何も選ばない。**

★ 何をしているか。**保存されたアダプタをディスクから読み直して**ベースにマージし、
  `models/{arm}/` へ書き出す。**3本とも入力は完全に同一である** ——
  同じアダプタ・同じベース・同じコード・同じ機械。**違うのは書き出す先の名前だけ。**

★ なぜそれを3回やるのか。pc-06 で、**入力が一致していて出力が動いた** ——
  train_loss は 0.2763194784 で pc-04 R1 と一致したのに、非注入群の解釈不能率は
  27.25% → 8.25% と 19.00pt 動き、GGUF の sha256 も一致しなかった。
  **この揺れの幅を誰も測っていない。** 3本作れば「同一アダプタの広がり R」が定義できる。

★ ⛔ **λ を掛ける口は無い。** 本ランに λ は無く、マージは学習時の α のままである
  (λ の梯子は positive-control-06 の話であり、finetune/scale_adapter.py が別に持つ)。

★ ⛔ **複製の番号は下の凍結表からしか選べない。** 4本目を作る口は用意しない ——
  用意すれば「R が閾値を跨ぐまで足す」ことができてしまう。

★ **M0 はこのスクリプトが作るものではない。** M0(`mv1r1-x40`)は train_lora.py が
  学習直後の**メモリ上の**モデルをマージして保存したものであり、**pc-04 R1 と同じ経路**である。
  本スクリプトが通るのは**ディスクから読み直す経路**(= pc-06 L0 と同じ)で、
  **その違い自体が測定対象の1つ**である(preregister の判定表 D)。

出力:
  models/{arm}/            マージ済みの HF 重み(GGUF 変換の入力)
  models/{arm}/merge.json  複製の番号 / 出所のアダプタ / α / r / アダプタの sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train_lora import MV01_RECIPE, RECIPES, RUN_BASES, recipe_arms

RUN = "merge-variance-01"

# ---------------------------------------------------------------------------
# ★ 複製 → アーム名。preregister「## ラン: merge-variance-01」→「段取り」が正であり、
#   **2026-08-16 に、モデルを1本も作る前に凍結された。**
#
#   器(65-manipulation-check.sh / runner)は**アーム名から**注入集合とキャッシュの
#   キーを引くので、複製ごとに別名でなければならない。**同じ名前で作り直すと
#   応答キャッシュが前の複製の答えを返し、何も測れない。**
#   末尾 2 桁が注入率として読まれる(`arm[-2:]`)ため `-x40` を保つ。
# ---------------------------------------------------------------------------
REPLICATE_ARMS: dict[int, str] = {1: "mv1m1-x40", 2: "mv1m2-x40", 3: "mv1m3-x40"}


def source_arm() -> str:
    """学習済みアダプタのあるアーム(= M0)。本ランが学習するのは R1 の1本だけである。"""
    return recipe_arms(RUN)[MV01_RECIPE]


def sha256_dir(d: Path) -> str:
    """ディレクトリの中身をまとめた sha256。**同じアダプタを読んだことの証拠になる。**

    ファイル名でソートしてから「名前 + 中身」を順に流し込む。並べ方を固定しないと、
    同じ中身でも値が変わって証拠にならない。
    """
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, choices=sorted(REPLICATE_ARMS), required=True,
                    help="★ 複製の番号。凍結表から引く(4本目を作る口は無い)")
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    n = args.replicate
    arm = REPLICATE_ARMS[n]
    src_arm = source_arm()

    if not arm.endswith("40"):
        print(f"★ アーム名 {arm} の末尾2桁が 40 でない。"
              "器は `arm[-2:]` を注入率として読むので、注入率が変わってしまう。")
        return 1

    print(f"ラン {RUN} / 複製 M{n}")
    print(f"  アダプタ: {src_arm}  →  書き出し: {arm}")
    print("  ★ λ は掛けない。マージは学習時の α のままである。")

    adapter_dir = args.model_dir / src_arm / "_adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        print(f"★ アダプタが無い: {adapter_dir}")
        print(f"  先に train_lora.py --run {RUN} --recipe {MV01_RECIPE} を走らせること。")
        return 1

    # --- 1. 学習時の設定を、2つの独立な出所から読んで突き合わせる ------------------
    #   train.json は train_lora.py が書いた学習の記録、adapter_config.json は
    #   peft がアダプタと一緒に書いたもの。**食い違えば、読んでいるアダプタが違う。**
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    train_json = args.model_dir / src_arm / "train.json"
    if not train_json.is_file():
        print(f"★ 学習の記録が無い: {train_json}")
        return 1
    rec = json.loads(train_json.read_text(encoding="utf-8"))

    recipe = RECIPES[MV01_RECIPE]
    alpha = float(cfg["lora_alpha"])
    rank = int(cfg["r"])
    print(f"アダプタの設定: r={rank} / α={alpha:g}")

    problems: list[str] = []
    if rank != recipe["rank"]:
        problems.append(f"rank が {rank} ≠ 凍結表 {recipe['rank']}")
    if alpha != float(recipe["alpha"]):
        problems.append(f"α が {alpha:g} ≠ 凍結表 {recipe['alpha']}")
    if rec.get("run") != RUN:
        problems.append(f"train.json のラン が {rec.get('run')!r} ≠ {RUN!r}")
    if rec.get("recipe") != MV01_RECIPE:
        problems.append(f"train.json のレシピ が {rec.get('recipe')!r} ≠ {MV01_RECIPE!r}")
    if float(rec.get("lora", {}).get("alpha", -1)) != alpha:
        problems.append("train.json と adapter_config.json で α が食い違う")
    # ★ rsLoRA / DoRA が有効だと ΔW の作られ方が変わる。**pc-04 R1 と同じ条件でなくなる。**
    if cfg.get("use_rslora"):
        problems.append("use_rslora が有効(scaling が α/√r になり、pc-04 R1 と条件が違う)")
    if cfg.get("use_dora"):
        problems.append("use_dora が有効(大きさと向きが分解され、pc-04 R1 と条件が違う)")
    if problems:
        print("★ 学習済みアダプタが凍結表と食い違っている。**何も書き出さない。**")
        for p in problems:
            print(f"    - {p}")
        return 1

    base_model, base_revision = RUN_BASES[RUN]
    if rec.get("base_model") != base_model or rec.get("base_revision") != base_revision:
        print(f"★ train.json のベースが凍結表と違う: "
              f"{rec.get('base_model')} @ {rec.get('base_revision')}")
        return 1

    # --- 2. アダプタの sha256 を記録する ------------------------------------------
    #   ★ pc-06 はこれを記録しておらず、切り分けを狭めた。
    #     3本の複製が**同じアダプタから作られた**ことは、この値が3本で一致することで示す。
    adapter_sha = sha256_dir(adapter_dir)
    print(f"アダプタの sha256: {adapter_sha}")

    out = args.model_dir / arm

    # --- 3. マージして保存(GGUF 変換は素の HF 重みを要求する) --------------------
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"ベースを読み込む: {base_model} @ {base_revision[:8]}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, revision=base_revision, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, revision=base_revision).save_pretrained(out)

    (out / "merge.json").write_text(json.dumps({
        "run": RUN, "replicate": n, "arm": arm,
        "source_arm": src_arm, "source_recipe": MV01_RECIPE,
        "adapter_sha256": adapter_sha,
        "base_model": base_model, "base_revision": base_revision,
        "rank": rank, "alpha": alpha,
        "merge_path": "from_disk",
        "note": "保存されたアダプタをディスクから読み直してマージしただけである。"
                "λ は掛けていない(本ランに λ は無い)。3本の複製は入力が完全に同一で、"
                "違うのは書き出す先の名前だけである。"
                "★ M0(mv1r1-x40)は train_lora.py が学習直後のメモリ上のモデルを"
                "マージしたもので、経路が違う(= pc-04 R1 と同じ経路)。"
                "その違い自体が本ランの測定対象の1つである(判定表の D)。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n保存: {out}")
    print(f"次: bash finetune/to_gguf.sh {arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
