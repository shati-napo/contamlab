"""CLI の組み立て部分 —— 課金と取り違えに直結するところだけを固定する。"""
from __future__ import annotations

from pathlib import Path

import pytest

from contamlab.benchmark import Item
from contamlab.cli import (
    _apply_split,
    _build_model,
    _count_uncached_calls,
    _model_name,
    build_parser,
)
from contamlab.clients import CallBudget, ClientOptions
from contamlab.runner import CachedModel, FakeModel, ResponseCache, format_prompt


def _items(n: int) -> list[Item]:
    return [
        Item(
            id=f"q{i:04d}",
            question=f"問題{i:04d}",
            answer=f"正解{i:04d}",
            choices=(f"正解{i:04d}", f"誤答{i:04d}"),
        )
        for i in range(n)
    ]


def _options(tmp_path: Path, **env) -> ClientOptions:
    return ClientOptions(budget=CallBudget(100), env=env)


class TestBuildModel:
    def test_模擬モデルはキャッシュで包まない(self, tmp_path: Path) -> None:
        """★ 実モデルが同名だったときに偽の応答を拾わないようにする。

        キャッシュのキーはモデル名とプロンプトだけなので、種別を跨いだ取り違えを
        キー側では防げない。書き込まないことで防ぐ。
        """
        items = _items(3)
        cache = ResponseCache(tmp_path / "cache.jsonl")

        model = _build_model("fake:demo:0.5", items, _options(tmp_path), cache)

        assert isinstance(model, FakeModel)
        assert not isinstance(model, CachedModel)

        model.answer(format_prompt(items[0]))
        assert len(cache) == 0, "模擬の応答がキャッシュに入ってはいけない"

    def test_実APIはキャッシュで包む(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")

        model = _build_model(
            "anthropic:claude:claude-opus-5",
            _items(3),
            _options(tmp_path, ANTHROPIC_API_KEY="k"),
            cache,
        )

        assert isinstance(model, CachedModel)
        assert model.name == "claude"

    def test_memorized_を付けると暗記モデルになる(self, tmp_path: Path) -> None:
        items = _items(3)
        cache = ResponseCache(tmp_path / "cache.jsonl")

        model = _build_model("fake:dirty:0.0:memorized", items, _options(tmp_path), cache)

        assert model.memorized_ids == {i.id for i in items}

    def test_APIキーが無ければ落とす(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")

        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            _build_model(
                "anthropic:claude:claude-opus-5", _items(3), _options(tmp_path), cache
            )

    @pytest.mark.parametrize("spec", ["fake:demo", "fake:demo:notanumber"])
    def test_模擬の指定が壊れていれば落とす(self, spec: str, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")

        with pytest.raises(SystemExit):
            _build_model(spec, _items(3), _options(tmp_path), cache)


class TestCountUncachedCalls:
    def test_模擬モデルは数に入らない(self, tmp_path: Path) -> None:
        items = _items(10)
        cache = ResponseCache(tmp_path / "cache.jsonl")

        assert _count_uncached_calls(cache, ["fake:a:0.5", "fake:b:0.5"], items, items) == 0

    def test_実APIは2条件ぶん数える(self, tmp_path: Path) -> None:
        items = _items(10)
        perturbed = [i.with_choices(tuple(reversed(i.choices))) for i in items]
        cache = ResponseCache(tmp_path / "cache.jsonl")

        count = _count_uncached_calls(cache, ["anthropic:c:m"], items, perturbed)

        assert count == 20

    def test_同一プロンプトは1回だけ数える(self, tmp_path: Path) -> None:
        """摂動で並びが変わらなかった問題は、同じプロンプトなので課金も1回。"""
        items = _items(10)
        cache = ResponseCache(tmp_path / "cache.jsonl")

        assert _count_uncached_calls(cache, ["anthropic:c:m"], items, items) == 10

    def test_キャッシュ済みは数に入らない(self, tmp_path: Path) -> None:
        items = _items(10)
        cache = ResponseCache(tmp_path / "cache.jsonl")
        for item in items:
            cache.put("c", format_prompt(item), "A")

        assert _count_uncached_calls(cache, ["anthropic:c:m"], items, items) == 0

    def test_モデルごとに別々に数える(self, tmp_path: Path) -> None:
        items = _items(10)
        cache = ResponseCache(tmp_path / "cache.jsonl")

        count = _count_uncached_calls(cache, ["anthropic:a:m", "openai:b:m"], items, items)

        assert count == 20


class TestModelName:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("fake:demo:0.5", "demo"),
            ("anthropic:claude:claude-opus-5", "claude"),
            ("compat:local:swallow:http://x:11434/v1", "local"),
        ],
    )
    def test_名前を取り出す(self, spec: str, expected: str) -> None:
        assert _model_name(spec) == expected

    @pytest.mark.parametrize("spec", ["fake", "fake:"])
    def test_名前が無ければ落とす(self, spec: str) -> None:
        with pytest.raises(SystemExit):
            _model_name(spec)


class TestApplySplit:
    def test_all_は全部返す(self) -> None:
        items = _items(100)

        assert _apply_split(items, "all") == items

    def test_devとholdoutで重複しない(self) -> None:
        items = _items(200)

        dev = _apply_split(items, "dev")
        holdout = _apply_split(items, "holdout")

        assert len(dev) + len(holdout) == 200
        assert not ({i.id for i in dev} & {i.id for i in holdout})

    def test_holdoutは警告を出す(self, capsys: pytest.CaptureFixture) -> None:
        """★ 1構成・1回だけ。黙って開封できてはいけない。"""
        _apply_split(_items(100), "holdout")

        assert "HOLDOUT を開封" in capsys.readouterr().err

    def test_devは警告を出さない(self, capsys: pytest.CaptureFixture) -> None:
        _apply_split(_items(100), "dev")

        assert capsys.readouterr().err == ""


class TestParser:
    def test_splitの既定はdev(self) -> None:
        """★ 既定でホールドアウトに触らない。"""
        args = build_parser().parse_args(
            ["run", "--synthetic", "10", "--model", "fake:a:0.5",
             "--target-effect", "0.05", "--expected-discordant-rate", "0.3"]
        )

        assert args.split == "dev"

    def test_温度の既定は0(self) -> None:
        args = build_parser().parse_args(
            ["run", "--synthetic", "10", "--model", "fake:a:0.5",
             "--target-effect", "0.05", "--expected-discordant-rate", "0.3"]
        )

        assert args.temperature == 0.0
        assert args.yes is False, "既定で課金しない"

    def test_未知の摂動器を弾く(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["perturb", "--synthetic", "10", "--perturbator", "存在しない"]
            )
