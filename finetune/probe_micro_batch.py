#!/usr/bin/env python3
"""finetune/probe_micro_batch.py — 実効バッチ 16 の「内訳」を、人ではなく規則で決める。

    python finetune/probe_micro_batch.py            # 既定は pc-04(8B)
    python finetune/probe_micro_batch.py --recipe R4   # 最も重い段で試したいとき

preregister「## ラン: positive-control-04」→「★ 変える1点 —— 実効バッチ 16 の内訳」の実装。
**このスクリプトは規則を決めない。規則は preregister にあり、測る前に凍結されている。**

    micro-batch を 8 → 4 → 2 → 1 の順に試し、OOM しなかった最初の値を使う。
    grad_accum = 16 / micro-batch。

★ なぜスクリプトにするのか —— 人が試すと「まあ 2 でいいか」と**途中で決めてしまう。**
  順序と停止条件が決まっている手続きは、人ではなく機械が回すべきである。

★ 何を選んでいないか —— **速さでは選ばない。** 速いほうを選ぶと GPU 個体ごとに
  値が変わり、再現性が落ちる。基準は「載るか」だけである。

★ この探索は結果を1つも見ない。正解率も損失も読まない。読むのは OOM かどうかだけ。

出力: reports/micro-batch(train_lora.py が読む)/ 実測ピーク VRAM を標準出力へ
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from train_lora import (BLOCK_SIZE, EFFECTIVE_BATCH, LORA_DROPOUT, LORA_TARGETS,
                        MICRO_BATCH_FILE, MICRO_BATCH_LADDER, RECIPES, RUN_BASES)


def try_one(micro_batch: int, base_model: str, base_revision: str,
            rank: int, alpha: int) -> tuple[bool, float, str]:
    """1回だけ前進+後退+最適化を回してみる。載れば (True, ピークGiB, "")。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            base_model, revision=base_revision, torch_dtype=torch.bfloat16, device_map="cuda")
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        # ★ 本番と同じ条件にする。ここを揃えないと測っている値が本番のピークではない。
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        vocab = model.config.vocab_size
        ids = torch.randint(0, vocab, (micro_batch, BLOCK_SIZE), device="cuda")
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_reserved() / 1024**3
        return True, peak, ""
    except torch.cuda.OutOfMemoryError as e:
        return False, 0.0, f"OOM: {str(e).splitlines()[0]}"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, 0.0, f"OOM: {str(e).splitlines()[0]}"
        raise
    finally:
        del model
        gc.collect()
        import torch as _t
        _t.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(RUN_BASES), default="positive-control-04")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="R4",
                    help="★ 既定は R4(rank 64 = 梯子で最も重い段)。"
                         "最も重い段で載ることを確かめれば、全段で載る")
    ap.add_argument("--out", type=Path, default=MICRO_BATCH_FILE)
    args = ap.parse_args()

    base_model, base_revision = RUN_BASES[args.run]
    recipe = RECIPES[args.recipe]
    print(f"ラン {args.run} / 段 {args.recipe}(rank {recipe['rank']} / α {recipe['alpha']})")
    print(f"ベース {base_model} @ {base_revision[:8]}")
    print(f"実効バッチ {EFFECTIVE_BATCH} を保ったまま {MICRO_BATCH_LADDER} を上から試す\n")

    for micro_batch in MICRO_BATCH_LADDER:
        grad_accum, rem = divmod(EFFECTIVE_BATCH, micro_batch)
        if rem:
            continue
        print(f"  micro {micro_batch} × grad_accum {grad_accum} … ", end="", flush=True)
        ok, peak, why = try_one(micro_batch, base_model, base_revision,
                                recipe["rank"], recipe["alpha"])
        if ok:
            print(f"載った(ピーク {peak:.1f} GiB)")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(f"{micro_batch}\n", encoding="utf-8", newline="\n")
            print(f"\n★ 採用: micro-batch {micro_batch} / grad_accum {grad_accum}"
                  f"(実効バッチ {EFFECTIVE_BATCH})")
            print(f"   実測ピーク {peak:.1f} GiB —— preregister pc-04 の実行環境に転記する")
            print(f"   記録: {args.out}")
            return 0
        print(f"載らない({why})")

    print("\n★ micro-batch 1 でも載らなかった。preregister pc-04 の停止条件に該当する。")
    print("  A100 40GB では 8B × 2048 が回らないという事実を記録し、")
    print("  機種の見直しは**別のランとして**事前登録すること。**ここで系列長を縮めない。**")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
