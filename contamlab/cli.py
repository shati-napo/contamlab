"""CLI。

    contamlab power   --n 100 --discordant-rate 0.2        何ポイントまで見えるか
    contamlab power   --effect 0.05 --discordant-rate 0.2  何問必要か
    contamlab perturb --benchmark b.jsonl --seed s         摂動を目で確かめる
    contamlab verify                                       測定装置の健全性チェック
    contamlab run     --benchmark b.jsonl --seed s ...     本番

**`power` から始めること。** 何問必要かを計算してから問題を集める。集めてから
「足りませんでした」と気づくのが、この分野で一番よくある失敗である。

**`run` で実 API を使うときは `--yes` が要る。** 付けなければ、呼び出し回数の
見積もりだけを出して止まる。金がかかる操作を勢いで走らせないため。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import Item, load_jsonl, split_dev_holdout, take_deterministic
from .clients import (
    CallBudget,
    ClientOptions,
    build_api_model,
    is_api_spec,
    load_dotenv,
)
from .harness import Design, UnderpoweredError, run, self_check
from .harness import _synthetic_items as synthetic_items
from .perturb import REGISTRY, get_perturbator, perturb_all
from .report import format_result, format_self_check, result_to_dict
from .runner import (
    DEFAULT_PROMPT_FORMAT,
    PROMPT_FORMATS,
    CachedModel,
    FakeModel,
    ResponseCache,
    current_prompt_format,
    format_prompt,
    set_prompt_format,
)
from .stats.power import min_detectable_effect, plan, required_n

DEFAULT_CACHE = Path("data/cache/responses.jsonl")

EXIT_UNDERPOWERED = 2
EXIT_NEEDS_CONFIRMATION = 3


def cmd_power(args: argparse.Namespace) -> int:
    if args.n is None and args.effect is None:
        raise SystemExit("--n か --effect のどちらかは要る")

    if args.n is not None:
        result = plan(
            n=args.n,
            discordant_rate=args.discordant_rate,
            target_effect=args.effect,
            alpha=args.alpha,
            power=args.power,
            one_sided=not args.two_sided,
        )
        print(result.summary())
        if args.effect is not None and not result.adequate:
            print()
            print(
                f"★ この設計では {args.effect * 100:.1f} ポイントの汚染は見えない。"
                "「有意差なし」を「汚染なし」と読んではいけない。"
            )
            return 1
        return 0

    n = required_n(
        args.effect,
        args.discordant_rate,
        alpha=args.alpha,
        power=args.power,
        one_sided=not args.two_sided,
    )
    print(f"必要な問題数  : {n}")
    print(f"狙う効果量    : {args.effect * 100:.2f} ポイント")
    print(f"不一致率 ψ    : {args.discordant_rate:.3f}")
    print(f"有意水準      : {args.alpha}({'両側' if args.two_sided else '片側'})")
    print(f"目標検出力    : {args.power:.2f}")
    return 0


def cmd_perturb(args: argparse.Namespace) -> int:
    set_prompt_format(args.prompt_format)
    items = _load_items(args)[: args.limit]
    perturbator = get_perturbator(args.perturbator)

    for item in items:
        perturbed = perturbator.apply(item, args.seed)
        if item.answer != perturbed.answer:
            raise SystemExit(f"★摂動が正解を変えている: id={item.id}")

        print("=" * 68)
        print(f"id: {item.id}   摂動器: {perturbator.name}   シード: {args.seed}")
        print("-" * 68)
        print("【オリジナル】")
        print(format_prompt(item))
        print()
        print("【摂動版】")
        print(format_prompt(perturbed))
        print()
        print(f"正解: {item.answer!r}  →  {perturbed.answer!r}")
        print()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    checks = self_check(n_items=args.n)
    print(format_self_check(checks))
    return 0 if all(c.passed for c in checks) else 1


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv(Path(".env"))
    # ★ 何よりも先に。FakeModel はプロンプトを構築時に captureし、呼び出し回数の
    #   見積もりもプロンプトからキャッシュキーを作る。後から変えると食い違う。
    set_prompt_format(args.prompt_format)

    items = _load_items(args)
    perturbed = perturb_all(items, get_perturbator(args.perturbator), args.seed)
    cache = ResponseCache(args.cache)

    needed = _count_uncached_calls(cache, args.model, items, perturbed)
    uses_api = any(is_api_spec(spec) for spec in args.model)
    _print_call_plan(args, items, cache, needed, uses_api)

    if uses_api and not args.yes:
        sys.stdout.flush()
        print(
            "\n実 API を使う設計なので、ここで止めた。"
            "内容を確認したうえで --yes を付けて再実行すること。",
            file=sys.stderr,
        )
        return EXIT_NEEDS_CONFIRMATION

    max_calls = args.max_calls if args.max_calls is not None else needed
    if max_calls < needed:
        sys.stdout.flush()
        print(
            f"\n★ --max-calls {max_calls} は必要回数 {needed} を下回っている。"
            "途中で止まって結果が出ないので、上限を上げるか問題数を減らすこと。",
            file=sys.stderr,
        )
        return 1

    budget = CallBudget(max_calls=max_calls)
    options = ClientOptions(
        budget=budget,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        rate_limit_per_minute=args.rate_limit,
    )
    models = [_build_model(spec, items, options, cache) for spec in args.model]

    design = Design(
        perturbator_name=args.perturbator,
        seed=args.seed,
        target_effect=args.target_effect,
        expected_discordant_rate=args.expected_discordant_rate,
        alpha=args.alpha,
        power=args.power,
        one_sided=not args.two_sided,
        n_perturbators_tried=args.k,
    )

    try:
        result = run(items, models, design, force_underpowered=args.force_underpowered)
    except UnderpoweredError as exc:
        print(f"検出力不足で中止した。\n\n{exc}", file=sys.stderr)
        return EXIT_UNDERPOWERED

    print()
    print(format_result(result))
    _print_run_footer(budget, cache)

    if args.json:
        # 書式は測定条件なので結果と一緒に残す。`Design` に足さないのは harness.py が
        # 編集禁止領域だからで、ここで注入する(記録が目的であって判定には使わない)。
        payload = result_to_dict(result)
        payload["prompt_format"] = current_prompt_format()

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON を書き出した: {args.json}")
    return 0


# --------------------------------------------------------------------------


def _load_items(args: argparse.Namespace) -> list[Item]:
    """問題を読み、DEV/HOLDOUT に分け、必要なら決定論的に n 問だけ抜く。

    `--synthetic` には分割を掛けない。合成問題は測定装置の動作確認用であって、
    守るべき HOLDOUT が存在しないため。
    """
    if args.synthetic is not None:
        items = synthetic_items(args.synthetic)
    elif args.benchmark is None:
        raise SystemExit("--benchmark か --synthetic のどちらかは要る")
    else:
        items = _apply_split(load_jsonl(args.benchmark), args.split)

    if args.sample_n:
        if args.sample_n > len(items):
            raise SystemExit(f"--sample-n {args.sample_n} が母数 {len(items)} を超えている")
        items = take_deterministic(items, args.sample_n)
    return items


def _apply_split(items: list[Item], split: str) -> list[Item]:
    if split == "all":
        return items

    dev, holdout = split_dev_holdout(items)
    if split == "dev":
        return dev

    print(
        "★★ HOLDOUT を開封している。**1構成・1回だけ。**\n"
        "    日付・構成・結果を preregister.md の「HOLDOUT 開封の記録」に必ず追記すること。\n"
        f"    (全 {len(items)} 問中 {len(holdout)} 問)\n",
        file=sys.stderr,
    )
    return holdout


def _model_name(spec: str) -> str:
    """指定文字列からモデル名(報告に出る名前)を取り出す。"""
    parts = spec.split(":")
    if len(parts) < 2 or not parts[1]:
        raise SystemExit(f"モデル指定に名前が無い: {spec!r}")
    return parts[1]


def _count_uncached_calls(
    cache: ResponseCache,
    specs: list[str],
    items: list[Item],
    perturbed: list[Item],
) -> int:
    """実際に課金される呼び出し回数。**キャッシュ済みと重複プロンプトを差し引く。**"""
    pending: set[tuple[str, str]] = set()
    for spec in specs:
        if not is_api_spec(spec):
            continue
        name = _model_name(spec)
        for item in [*items, *perturbed]:
            prompt = format_prompt(item)
            if cache.get(name, prompt) is None:
                pending.add((name, prompt))
    return len(pending)


def _print_call_plan(
    args: argparse.Namespace,
    items: list[Item],
    cache: ResponseCache,
    needed: int,
    uses_api: bool,
) -> None:
    api_models = [s for s in args.model if is_api_spec(s)]
    print("■ 呼び出しの見積もり")
    print(f"  問題数            : {len(items)}")
    print(f"  モデル            : {len(args.model)} 本(うち実 API {len(api_models)} 本)")
    print(f"  キャッシュ        : {args.cache}({len(cache)} 件)")
    print(f"  出力書式          : {current_prompt_format()}")
    print(f"  ★課金される回数   : {needed}")
    if uses_api:
        print(f"  温度 / 最大トークン: {args.temperature} / {args.max_tokens}")
        if args.rate_limit:
            print(f"  レート制限        : {args.rate_limit} 回/分")


def _print_run_footer(budget: CallBudget, cache: ResponseCache) -> None:
    print()
    print(f"API 呼び出し: {budget.used} 回(上限 {budget.max_calls})")
    if cache.conflicts:
        print(
            f"★ 同じ問いに違う応答が {len(cache.conflicts)} 件あった。"
            "モデルが非決定的である証拠。temperature を確認すること。"
            "(キャッシュは最初の応答を保持している)"
        )


def _build_model(spec: str, items: list[Item], options: ClientOptions, cache: ResponseCache):
    """モデル指定を組み立てる。**実 API のときだけ**応答キャッシュで包む。

        fake:NAME:ACCURACY[:memorized]     模擬(課金なし)
        anthropic:NAME:MODEL_ID
        openai:NAME:MODEL_ID
        compat:NAME:MODEL_ID:BASE_URL      ローカル / 自前ホスト

    模擬モデルをキャッシュに入れない理由は2つ。課金されないので入れる利得が無いことと、
    **実モデルが模擬と同じ名前だったときに偽の応答を拾ってしまう**こと。キャッシュの
    キーはモデル名とプロンプトだけなので、種別を跨いだ取り違えを防げない。
    """
    if spec.startswith("fake:"):
        return _build_fake(spec, items)

    try:
        model = build_api_model(spec, options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return CachedModel(model, cache)


def _build_fake(spec: str, items: list[Item]) -> FakeModel:
    parts = spec.split(":")
    if len(parts) < 3:
        raise SystemExit(
            f"モデル指定が読めない: {spec!r}(形式は fake:NAME:ACCURACY[:memorized])"
        )
    try:
        accuracy = float(parts[2])
    except ValueError:
        raise SystemExit(f"正答率が数値でない: {parts[2]!r}") from None

    memorized = [i.id for i in items] if len(parts) > 3 and parts[3] == "memorized" else []
    return FakeModel(parts[1], items, base_accuracy=accuracy, memorized_ids=memorized)


# --------------------------------------------------------------------------


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", type=Path, help="JSONL のベンチマーク")
    parser.add_argument("--synthetic", type=int, help="合成問題を N 問使う(動作確認用)")
    parser.add_argument("--seed", default="seed-1", help="摂動のシード")
    parser.add_argument("--perturbator", default="shuffle_choices", choices=sorted(REGISTRY))
    parser.add_argument(
        "--split",
        default="dev",
        choices=("dev", "holdout", "all"),
        help="既定は dev。★holdout は1構成・1回だけ(--synthetic には掛からない)",
    )
    parser.add_argument(
        "--sample-n", type=int, help="決定論的に N 問だけ抜く(検出力で決めた標本サイズ)"
    )
    parser.add_argument(
        "--prompt-format",
        default=DEFAULT_PROMPT_FORMAT,
        choices=sorted(PROMPT_FORMATS),
        help="出力書式。A=現行 / B=出力例つき / C=先頭を「答え: X」に固定。"
        "★測定条件なので、変えるときは preregister.md に書いてから",
    )


def _add_stats_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=float, default=0.05, help="有意水準")
    parser.add_argument("--power", type=float, default=0.80, help="目標検出力")
    parser.add_argument("--two-sided", action="store_true", help="両側検定にする")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contamlab",
        description="LLM ベンチマークの汚染を、検定・信頼区間・検出力つきで測る",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_power = sub.add_parser("power", help="検出力を計算する(まずここから)")
    p_power.add_argument("--n", type=int, help="手元にある問題数")
    p_power.add_argument("--effect", type=float, help="検出したい効果量(例 0.05 = 5pt)")
    p_power.add_argument(
        "--discordant-rate", type=float, required=True, help="想定する不一致率 ψ"
    )
    _add_stats_args(p_power)
    p_power.set_defaults(func=cmd_power)

    p_perturb = sub.add_parser("perturb", help="摂動を目で確かめる")
    _add_benchmark_args(p_perturb)
    p_perturb.add_argument("--limit", type=int, default=3, help="表示する件数")
    p_perturb.set_defaults(func=cmd_perturb)

    p_verify = sub.add_parser("verify", help="測定装置の健全性チェック")
    p_verify.add_argument("--n", type=int, default=800, help="合成問題の件数")
    p_verify.set_defaults(func=cmd_verify)

    p_run = sub.add_parser("run", help="本番の検査を1回")
    _add_benchmark_args(p_run)
    _add_stats_args(p_run)
    p_run.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="SPEC",
        help="fake:NAME:ACC[:memorized] / anthropic:NAME:MODEL_ID / "
        "openai:NAME:MODEL_ID / compat:NAME:MODEL_ID:BASE_URL",
    )
    p_run.add_argument("--target-effect", type=float, required=True, help="狙う効果量")
    p_run.add_argument(
        "--expected-discordant-rate", type=float, required=True, help="想定する不一致率 ψ"
    )
    p_run.add_argument("--k", type=int, default=1, help="試した摂動器の数(事前確約の K)")
    p_run.add_argument("--json", type=Path, help="JSON の出力先")
    p_run.add_argument(
        "--force-underpowered",
        action="store_true",
        help="★検出力不足でも強行する。結論に検出力不足を必ず書くこと",
    )

    api = p_run.add_argument_group("実 API(課金される)")
    api.add_argument(
        "--yes", action="store_true", help="実 API の呼び出しを承認する。無ければ見積もりだけ"
    )
    api.add_argument("--max-calls", type=int, help="API 呼び出しの上限(既定は必要回数ちょうど)")
    api.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="応答キャッシュ(追記専用)")
    api.add_argument(
        "--temperature", type=float, default=0.0, help="★0 以外にすると測るものが変わる"
    )
    api.add_argument("--max-tokens", type=int, default=256)
    api.add_argument("--rate-limit", type=int, help="1分あたりの呼び出し上限")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    # 出力は全部日本語である。日本語を表示できないコンソール(非日本語ロケールの
    # Windows は既定が cp1252 など)では、最初の print が UnicodeEncodeError で落ちる。
    # 測定とは無関係なところで exit 1 になり、しかも原因が分からない。
    # 表示できない文字は置換して続行させる。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # リダイレクト先が TextIOWrapper でないことがある
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
