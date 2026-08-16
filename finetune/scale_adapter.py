#!/usr/bin/env python3
"""finetune/scale_adapter.py — 学習済みアダプタを λ 倍してマージし、1段分の重みを作る。

    python finetune/scale_adapter.py --lambda-step L0     # λ=1.0(関門)
    python finetune/scale_adapter.py --lambda-step L2     # λ=0.6

preregister「ラン: positive-control-06」→「★ 動かす1点 —— 推論時の LoRA スケール λ」
の実装。**このスクリプトは規則を決めない。**

★ 何をしているか。LoRA の重み更新は

      ΔW = (α / r) · B · A

  であり、**α に比例する。** よって `α → λ·α` として読み直せば、
  マージ後の重みは厳密に `W + λ·ΔW` になる。**学習は一度も走らせない。**

★ なぜこれが pc-04・pc-05 より強いか。pc-04(E を上げる)と pc-05(埋め草で薄める)は
  **段ごとに別のモデルを学習していた**ので、段の間に学習の非決定性・ステップ数・
  データ順の違いが同時に入っていた(pc-05 は `T = 注入 × E/(1−f)` により
  希釈と学習量が交絡していた)。**本ランの5段は同一のアダプタから作られ、
  違うのは λ だけである。**

⛔ **λ を自由に渡す口は無い。** 下の凍結表の5値からしか選べない。
   用意すれば、それが事前登録の外に出る口になる。

★ 実行順序は preregister が凍結している —— **L0 → L1 → L2 → L3 → L4 の上から順で、
  合格した段で止め、6本目を作らない。** L0 は pc-04 R1 の再現確認の関門であり、
  帯を外れたら**そこで止める**(このスクリプトは操作チェックの結果を読まない。
  順序と関門は人が読んで守る規則である)。

出力:
  models/{arm}/            マージ済みの HF 重み(GGUF 変換の入力)
  models/{arm}/scale.json  λ / α_train / α_merged / 実効比 / 実測 scaling
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from train_lora import PC06_RECIPE, RECIPES, RUN_BASES, recipe_arms

RUN = "positive-control-06"

# ---------------------------------------------------------------------------
# ★ λ の梯子。preregister「## ラン: positive-control-06」→「★ 動かす1点」が正であり、
#   **2026-08-16 に、モデルを1本も作る前に凍結された。**
#
#   段     λ     マージ時の α(= λ × 64)   位置づけ
#   L0    1.0    64                       ★ pc-04 R1 の再現確認(関門)
#   L1    0.8    51.2
#   L2    0.6    38.4
#   L3    0.4    25.6
#   L4    0.2    12.8
#
#   ★ 順序の根拠(結果と独立に決まる): λ を下げると **c(解釈不能率)は改善・
#     a(差)は落ちる**方向である。上から下ろせば、最初に通った段が
#     「**a を最も保ったまま c を通した λ**」になる。これは pc-03 が使った作法
#     (「順序を結果と独立な基準で決め、最初に通ったものを採る」)と同じであり、
#     **本ランの数字を1つも見ずに決まる。**
#
#   ⛔ **中間の λ を後から足さない。** 「合格するまで刻む」になる。格子は5段で固定。
# ---------------------------------------------------------------------------
LAMBDA_STEPS: dict[str, float] = {
    "L0": 1.0, "L1": 0.8, "L2": 0.6, "L3": 0.4, "L4": 0.2,
}

# 段 → アーム名。器(65-manipulation-check.sh / runner)は**アーム名から**
# 注入集合とキャッシュのキーを引くので、段ごとに別名でなければならない。
# 末尾 2 桁が注入率として読まれる(`arm[-2:]`)ため `-x40` を保つ。
LAMBDA_ARMS: dict[str, str] = {
    "L0": "pc6L10-x40", "L1": "pc6L08-x40", "L2": "pc6L06-x40",
    "L3": "pc6L04-x40", "L4": "pc6L02-x40",
}

# ★ L0 は関門の段である。preregister「★ L0 の関門 —— pc-04 R1 が再現しているか」。
#   帯(非注入群 正解率 [0.384, 0.481] / 解釈不能率 [22.9%, 31.6%] / 差 +8.4pt 以上)を
#   外れたら**梯子を下ろさずに止める。** これは人が読んで守る規則である。
GATE_STEPS = ("L0",)

# ★ 実効比の許容。preregister の停止条件
#   「マージした λ の実効値が凍結値と食い違った(相対 1e-6 を超えて離れる)」。
RELATIVE_TOLERANCE = 1e-6


def source_arm() -> str:
    """学習済みアダプタのあるアーム。pc-06 が学習するのは R1 の1本だけである。"""
    return recipe_arms(RUN)[PC06_RECIPE]


def check_relative(name: str, got: float, want: float) -> bool:
    """相対誤差が許容の中か。**測っているものが表と違うなら書き出さずに止める。**"""
    if want == 0:
        ok = got == 0
        rel = 0.0 if ok else float("inf")
    else:
        rel = abs(got - want) / abs(want)
        ok = rel <= RELATIVE_TOLERANCE
    mark = "" if ok else "★ "
    print(f"  {mark}{name}: 実測 {got!r} / 期待 {want!r}(相対 {rel:.3e})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-step", choices=sorted(LAMBDA_STEPS), required=True,
                    help="★ λ の梯子の段。凍結表から λ を引く(λ を直接渡す口は無い)")
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    step = args.lambda_step
    lam = LAMBDA_STEPS[step]
    arm = LAMBDA_ARMS[step]
    src_arm = source_arm()

    if not arm.endswith("40"):
        print(f"★ アーム名 {arm} の末尾2桁が 40 でない。"
              "器は `arm[-2:]` を注入率として読むので、注入率が変わってしまう。")
        return 1

    print(f"ラン {RUN} / 段 {step}(λ = {lam})")
    print(f"  アダプタ: {src_arm}  →  書き出し: {arm}")
    if step in GATE_STEPS:
        print("⚠ この段は**関門**である。preregister「★ L0 の関門」の帯 —— "
              "非注入群 正解率 [0.384, 0.481] / 解釈不能率 [22.9%, 31.6%] / 差 +8.4pt 以上 —— "
              "を外れたら、**梯子を下ろさずにそこで止めること。**")

    adapter_dir = args.model_dir / src_arm / "_adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        print(f"★ アダプタが無い: {adapter_dir}")
        print(f"  先に train_lora.py --run {RUN} --recipe {PC06_RECIPE} を走らせること。")
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

    recipe = RECIPES[PC06_RECIPE]
    alpha_train = float(cfg["lora_alpha"])
    rank = int(cfg["r"])
    print(f"アダプタの設定: r={rank} / α={alpha_train:g}")

    problems: list[str] = []
    if rank != recipe["rank"]:
        problems.append(f"rank が {rank} ≠ 凍結表 {recipe['rank']}")
    if alpha_train != float(recipe["alpha"]):
        problems.append(f"α が {alpha_train:g} ≠ 凍結表 {recipe['alpha']}")
    if rec.get("run") != RUN:
        problems.append(f"train.json のラン が {rec.get('run')!r} ≠ {RUN!r}")
    if rec.get("recipe") != PC06_RECIPE:
        problems.append(f"train.json のレシピ が {rec.get('recipe')!r} ≠ {PC06_RECIPE!r}")
    if float(rec.get("lora", {}).get("alpha", -1)) != alpha_train:
        problems.append("train.json と adapter_config.json で α が食い違う")
    # ★ rsLoRA / DoRA が有効だと ΔW が α に単純比例しない(scaling が α/√r になる・
    #   大きさと向きが分解される)。**λ 倍の意味が変わるので、有効なら走らせない。**
    if cfg.get("use_rslora"):
        problems.append("use_rslora が有効(scaling が α/√r になり、λ 倍の意味が変わる)")
    if cfg.get("use_dora"):
        problems.append("use_dora が有効(大きさと向きが分解され、λ 倍の意味が変わる)")
    if problems:
        print("★ 学習済みアダプタが凍結表と食い違っている。**何も書き出さない。**")
        for p in problems:
            print(f"    - {p}")
        return 1

    alpha_merged = lam * alpha_train
    print(f"マージ時の α: {alpha_train:g} × {lam} = {alpha_merged:g}")

    base_model, base_revision = RUN_BASES[RUN]
    if rec.get("base_model") != base_model or rec.get("base_revision") != base_revision:
        print(f"★ train.json のベースが凍結表と違う: "
              f"{rec.get('base_model')} @ {rec.get('base_revision')}")
        return 1
    out = args.model_dir / arm

    # --- 2. α を書き換えたアダプタの複製を作る(元のアダプタは触らない) -------------
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        scaled_dir = Path(tmp) / "adapter"
        shutil.copytree(adapter_dir, scaled_dir)
        scaled_cfg = json.loads(
            (scaled_dir / "adapter_config.json").read_text(encoding="utf-8"))
        scaled_cfg["lora_alpha"] = alpha_merged
        (scaled_dir / "adapter_config.json").write_text(
            json.dumps(scaled_cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")

        print(f"ベースを読み込む: {base_model} @ {base_revision[:8]}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, revision=base_revision, torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu")
        model = PeftModel.from_pretrained(model, str(scaled_dir))

        # --- 3. 実効スケールを**マージする前に**確かめる -------------------------
        #   peft は scaling = lora_alpha / r を層に持つ。**そこが λ·α/r に
        #   なっていなければ、測っているものが表と違う。**
        adapter_name = "default"
        loaded_alpha = float(model.peft_config[adapter_name].lora_alpha)
        scalings: set[float] = set()
        for module in model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and adapter_name in scaling:
                scalings.add(float(scaling[adapter_name]))

        want_scaling = alpha_merged / rank
        print("実効スケールの照合:")
        ok = check_relative("読み込んだ α", loaded_alpha, alpha_merged)
        ok &= check_relative("実効比 α_merged / α_train", loaded_alpha / alpha_train, lam)
        if len(scalings) != 1:
            print(f"  ★ 層ごとの scaling が1つに揃っていない: {sorted(scalings)}")
            ok = False
        else:
            ok &= check_relative("層の scaling(= α/r)", scalings.pop(), want_scaling)
        if not ok:
            print("★ 実効値が凍結値と食い違った。**何も書き出さずに止める**"
                  "(preregister の停止条件)。")
            return 1

        # --- 4. マージして保存(GGUF 変換は素の HF 重みを要求する) ----------------
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
        AutoTokenizer.from_pretrained(base_model, revision=base_revision).save_pretrained(out)

    (out / "scale.json").write_text(json.dumps({
        "run": RUN, "lambda_step": step, "lambda": lam, "arm": arm,
        "source_arm": src_arm, "source_recipe": PC06_RECIPE,
        "base_model": base_model, "base_revision": base_revision,
        "rank": rank, "alpha_train": alpha_train, "alpha_merged": alpha_merged,
        # ★ preregister の実行環境「各段の実効 λ」に転記する値。
        "effective_lambda": loaded_alpha / alpha_train,
        "scaling": want_scaling,
        "note": "学習は一度も走らせていない。ΔW = (α/r)·B·A の α を λ 倍して"
                "マージし直しただけであり、重みは厳密に W + λ·ΔW である。"
                "5段は同一のアダプタから作られ、段の間で違うのは λ だけである。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"保存: {out}")
    print(f"  次: bash finetune/to_gguf.sh {arm} "
          f"&& bash scripts/65-manipulation-check.sh {arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
