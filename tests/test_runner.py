"""③ 実行層 — 採点の一貫性、追記専用キャッシュ、汚染の注入。"""
from __future__ import annotations

from pathlib import Path

import pytest

from contamlab.benchmark import Item
from contamlab.perturb import ShuffleChoices, perturb_all
from contamlab.runner import (
    CachedModel,
    FakeModel,
    Model,
    ResponseCache,
    format_prompt,
    grade,
    normalize,
    run_items,
    select,
)


def _item(item_id: str = "q1") -> Item:
    return Item(
        id=item_id,
        question="水素の元素記号は?",
        answer="H",
        choices=("H", "He", "Li", "O"),
    )


def _items(n: int) -> list[Item]:
    return [
        Item(
            id=f"q{i:03d}",
            question=f"問題{i:03d}: 正しいものを選べ。",
            answer=f"正解{i:03d}",
            choices=(f"正解{i:03d}", f"誤答A{i:03d}", f"誤答B{i:03d}", f"誤答C{i:03d}"),
        )
        for i in range(n)
    ]


class TestFormatPrompt:
    def test_問題文が先頭に来る(self) -> None:
        """応答から問題を逆引きするので、先頭固定は前提条件。"""
        assert format_prompt(_item()).startswith("水素の元素記号は?")

    def test_選択肢にラベルが付く(self) -> None:
        prompt = format_prompt(_item())

        assert "A. H" in prompt
        assert "D. O" in prompt

    def test_摂動版と構造が同じ(self) -> None:
        """★ 書式が違うと、正答率の差が汚染ではなく書式の差になる。"""
        item = _item()
        perturbed = ShuffleChoices().apply(item, "seed")

        original_lines = format_prompt(item).splitlines()
        perturbed_lines = format_prompt(perturbed).splitlines()

        assert len(original_lines) == len(perturbed_lines)
        assert original_lines[0] == perturbed_lines[0]
        assert original_lines[-1] == perturbed_lines[-1]

    def test_自由記述には選択肢が出ない(self) -> None:
        prompt = format_prompt(Item(id="q1", question="首都は?", answer="東京"))

        assert "A." not in prompt


class TestSelect:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("A", "H"),
            ("a", "H"),
            ("B", "He"),
            ("A.", "H"),
            ("(A)", "H"),
            ("[C]", "Li"),
            ("答え: D", "O"),
            ("答えはB", "He"),
            ("正解: A", "H"),
            ("H", "H"),
            ("He", "He"),
            ("  A  ", "H"),
        ],
    )
    def test_よくある応答の形を読み取れる(self, raw: str, expected: str) -> None:
        assert select(_item(), raw) == expected

    def test_入れ子の選択肢は長いほうを採る(self) -> None:
        """"He" は "H" を含む。文中に He があれば He を選んだと読む。"""
        assert select(_item(), "答えは He です") == "He"

    @pytest.mark.parametrize("raw", ["", "   ", "分かりません", "H と Li のどちらか"])
    def test_解釈できない応答はNone(self, raw: str) -> None:
        assert select(_item(), raw) is None

    def test_範囲外のラベルは採らない(self) -> None:
        """4択に Z はない。ラベルとして読まず、次の手段に落ちる。"""
        assert select(_item(), "Z") is None

    def test_自由記述はそのまま返す(self) -> None:
        item = Item(id="q1", question="首都は?", answer="東京")

        assert select(item, "  東京  ") == "東京"

    def test_摂動後は提示順で解釈される(self) -> None:
        """★ ラベルは並べ替え後の位置を指す。ここを間違えると全部狂う。"""
        perturbed = _item().with_choices(("O", "Li", "He", "H"))

        assert select(perturbed, "A") == "O"
        assert select(perturbed, "D") == "H"


class TestGrade:
    def test_正解(self) -> None:
        response = grade(_item(), "A")

        assert response.correct
        assert response.parsed
        assert response.item_id == "q1"

    def test_不正解(self) -> None:
        assert not grade(_item(), "B").correct

    def test_解釈不能は不正解だが_解釈できなかったことも残る(self) -> None:
        """★ 解釈不能率が条件間で違えば比較が壊れている。件数を見られるようにする。"""
        response = grade(_item(), "分かりません")

        assert not response.correct
        assert not response.parsed
        assert response.raw == "分かりません"

    def test_摂動しても同じ採点関数で正解が取れる(self) -> None:
        item = _item()
        perturbed = ShuffleChoices().apply(item, "seed")
        label = perturbed.answer_label

        assert grade(perturbed, label).correct

    def test_自由記述は大文字小文字と空白を無視する(self) -> None:
        item = Item(id="q1", question="記号は?", answer="Fe")

        assert grade(item, "  fe ").correct


class TestNormalize:
    def test_前後の空白と内部の連続空白を潰す(self) -> None:
        assert normalize("  a \n\t b  ") == "a b"


class TestResponseCache:
    def test_書いて読める(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")

        assert cache.put("m", "prompt", "response") is True
        assert cache.get("m", "prompt") == "response"
        assert len(cache) == 1

    def test_未登録はNone(self, tmp_path: Path) -> None:
        assert ResponseCache(tmp_path / "cache.jsonl").get("m", "prompt") is None

    def test_モデルが違えば別エントリ(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")
        cache.put("m1", "prompt", "r1")

        assert cache.get("m2", "prompt") is None

    def test_ファイルから読み直せる(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        ResponseCache(path).put("m", "prompt", "response")

        assert ResponseCache(path).get("m", "prompt") == "response"

    def test_既存を上書きしない(self, tmp_path: Path) -> None:
        """★ 追記専用。一度得た応答は消さない。"""
        cache = ResponseCache(tmp_path / "cache.jsonl")
        cache.put("m", "prompt", "最初の応答")

        assert cache.put("m", "prompt", "後から来た違う応答") is False
        assert cache.get("m", "prompt") == "最初の応答"

    def test_応答が食い違ったら衝突として記録する(self, tmp_path: Path) -> None:
        """★ モデルが非決定的という重大事。黙って握り潰さない。"""
        cache = ResponseCache(tmp_path / "cache.jsonl")
        cache.put("m", "prompt", "A")
        cache.put("m", "prompt", "B")

        assert len(cache.conflicts) == 1

    def test_同じ応答の再投入は衝突ではない(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")
        cache.put("m", "prompt", "A")
        cache.put("m", "prompt", "A")

        assert cache.conflicts == []

    def test_壊れた行は行番号つきで落とす(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        path.write_text('{"key":"k","response":"r"}\nこれは JSON ではない\n', encoding="utf-8")

        with pytest.raises(ValueError, match=r":2 "):
            ResponseCache(path)


class TestCachedModel:
    def test_2回目はAPIを叩かない(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")
        model = CachedModel(FakeModel("fake", _items(1)), cache)
        prompt = format_prompt(_items(1)[0])

        first = model.answer(prompt)
        second = model.answer(prompt)

        assert first == second
        assert model.api_calls == 1

    def test_プロトコルを満たす(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "cache.jsonl")

        assert isinstance(CachedModel(FakeModel("fake", _items(1)), cache), Model)


class TestFakeModel:
    def test_プロトコルを満たす(self) -> None:
        assert isinstance(FakeModel("fake", _items(1)), Model)

    def test_暗記した問題はオリジナルなら必ず正解(self) -> None:
        items = _items(30)
        model = FakeModel("m", items, base_accuracy=0.0, memorized_ids=[i.id for i in items])

        responses = run_items(model, items)

        assert all(r.correct for r in responses)

    def test_暗記していても摂動版では発火しない(self) -> None:
        """★ 「暗記」の最小限のモデル化。表層が変われば記憶は効かない。"""
        items = _items(30)
        perturbed = perturb_all(items, ShuffleChoices(), "seed")
        model = FakeModel("m", items, base_accuracy=0.0, memorized_ids=[i.id for i in items])

        responses = run_items(model, perturbed)

        # 一様置換なので約1/4は並びが変わらず記憶が発火する。全問正解にはならない。
        assert sum(r.correct for r in responses) < len(items)

    def test_汚染ありのモデルは摂動で大きく落ちる(self) -> None:
        items = _items(200)
        perturbed = perturb_all(items, ShuffleChoices(), "seed")
        model = FakeModel("m", items, base_accuracy=0.25, memorized_ids=[i.id for i in items])

        original_correct = sum(r.correct for r in run_items(model, items))
        perturbed_correct = sum(r.correct for r in run_items(model, perturbed))

        assert original_correct == 200
        assert perturbed_correct < 120

    def test_汚染なしのモデルは平均的に落ちない(self) -> None:
        """★ 偽陽性を出さないこと。素の能力は摂動で変わらない。"""
        items = _items(400)
        perturbed = perturb_all(items, ShuffleChoices(), "seed")
        model = FakeModel("m", items, base_accuracy=0.6)

        original_correct = sum(r.correct for r in run_items(model, items))
        perturbed_correct = sum(r.correct for r in run_items(model, perturbed))

        assert abs(original_correct - perturbed_correct) < 50

    def test_決定論的(self) -> None:
        items = _items(20)
        a = FakeModel("m", items, base_accuracy=0.5)
        b = FakeModel("m", items, base_accuracy=0.5)

        assert [r.raw for r in run_items(a, items)] == [r.raw for r in run_items(b, items)]

    def test_知らない問題には空応答(self) -> None:
        model = FakeModel("m", _items(1))

        assert model.answer("まったく無関係な文字列") == ""

    def test_素の正答率の範囲を弾く(self) -> None:
        with pytest.raises(ValueError):
            FakeModel("m", _items(1), base_accuracy=1.5)


class TestRunItems:
    def test_順序を保つ(self) -> None:
        items = _items(10)

        responses = run_items(FakeModel("m", items), items)

        assert [r.item_id for r in responses] == [i.id for i in items]
