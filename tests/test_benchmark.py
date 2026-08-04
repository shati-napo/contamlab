"""① ベンチマーク層 — 不変条件と、公開日(as_of)の扱い。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import json
import os
import subprocess
import sys
import textwrap

from contamlab.benchmark import (
    HOLDOUT_FRACTION,
    Item,
    load_jsonl,
    published_after,
    published_before,
    split_dev_holdout,
    take_deterministic,
    undated,
    unit_hash,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pool(n: int, prefix: str = "q") -> list[Item]:
    return [Item(id=f"{prefix}{i:05d}", question=f"Q{i}", answer="a") for i in range(n)]


def _item(**overrides) -> Item:
    base = dict(id="q1", question="水素の元素記号は?", answer="H", choices=("H", "He", "Li", "O"))
    base.update(overrides)
    return Item(**base)


class TestItem:
    def test_正解は位置ではなく中身で持つ(self) -> None:
        item = _item()

        assert item.answer == "H"
        assert item.answer_index == 0
        assert item.answer_label == "A"

    def test_選択肢を並べ替えても正解は不変(self) -> None:
        """★ この設計の要。位置で持っていたら摂動のたびに壊れる。"""
        item = _item()

        shuffled = item.with_choices(("O", "Li", "He", "H"))

        assert shuffled.answer == "H"
        assert shuffled.answer_index == 3
        assert shuffled.answer_label == "D"

    def test_並べ替えてもidは変わらない(self) -> None:
        """オリジナルと摂動版の対応付けは id で行う。"""
        assert _item().with_choices(("O", "H", "He", "Li")).id == "q1"

    def test_メタデータは統合される(self) -> None:
        item = _item(metadata={"subject": "化学"})

        merged = item.with_choices(("H", "O", "Li", "He"), perturbator="shuffle_choices")

        assert merged.metadata == {"subject": "化学", "perturbator": "shuffle_choices"}

    def test_自由記述(self) -> None:
        item = Item(id="q1", question="首都は?", answer="東京")

        assert not item.is_multiple_choice
        assert item.answer_index is None
        assert item.answer_label is None

    def test_正解が選択肢に無ければ落とす(self) -> None:
        with pytest.raises(ValueError, match="正解が選択肢に無い"):
            _item(answer="Xe")

    def test_選択肢の重複を落とす(self) -> None:
        """重複があると採点が一意に決まらない。"""
        with pytest.raises(ValueError, match="重複"):
            _item(choices=("H", "H", "He", "Li"))

    @pytest.mark.parametrize("field,value", [("id", ""), ("question", "  "), ("answer", "")])
    def test_空の必須項目を落とす(self, field: str, value: str) -> None:
        overrides = {field: value}
        if field == "answer":
            overrides["choices"] = ()
        with pytest.raises(ValueError):
            _item(**overrides)


class TestLoadJsonl:
    def _write(self, tmp_path: Path, *lines: str) -> Path:
        path = tmp_path / "bench.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_読み込める(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            '{"id":"q1","question":"Q1","answer":"a","choices":["a","b"],'
            '"published_at":"2024-05-01","source":"JMMLU"}',
            '{"id":"q2","question":"Q2","answer":"東京"}',
        )

        items = load_jsonl(path)

        assert len(items) == 2
        assert items[0].published_at == date(2024, 5, 1)
        assert items[0].source == "JMMLU"
        assert items[1].choices == ()

    def test_空行とコメント行を飛ばす(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path, "// これはコメント", "", '{"id":"q1","question":"Q","answer":"a"}'
        )

        assert len(load_jsonl(path)) == 1

    def test_壊れた行を行番号つきで落とす(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path, '{"id":"q1","question":"Q","answer":"a"}', "{ これは JSON ではない"
        )

        with pytest.raises(ValueError, match=r":2 "):
            load_jsonl(path)

    def test_不正な問題を行番号つきで落とす(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path, '{"id":"q1","question":"Q","answer":"z","choices":["a","b"]}'
        )

        with pytest.raises(ValueError, match=r":1 .*正解が選択肢に無い"):
            load_jsonl(path)

    def test_idの重複を落とす(self, tmp_path: Path) -> None:
        """重複があるとオリジナルと摂動版の対応付けが壊れる。"""
        path = self._write(
            tmp_path,
            '{"id":"q1","question":"Q1","answer":"a"}',
            '{"id":"q1","question":"Q2","answer":"b"}',
        )

        with pytest.raises(ValueError, match="id が重複"):
            load_jsonl(path)

    def test_空のファイルを落とす(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "")

        with pytest.raises(ValueError, match="問題が1件も無い"):
            load_jsonl(path)


class TestCutoff:
    def _items(self) -> list[Item]:
        return [
            Item(id="old", question="Q1", answer="a", published_at=date(2023, 1, 1)),
            Item(id="new", question="Q2", answer="a", published_at=date(2025, 6, 1)),
            Item(id="unknown", question="Q3", answer="a"),
        ]

    def test_カットオフより後だけ取る(self) -> None:
        result = published_after(self._items(), date(2024, 1, 1))

        assert [i.id for i in result] == ["new"]

    def test_カットオフ以前だけ取る(self) -> None:
        result = published_before(self._items(), date(2024, 1, 1))

        assert [i.id for i in result] == ["old"]

    def test_公開日不明は両方から除外される(self) -> None:
        """★「たぶん新しい」で通すとそこから汚染が入る。"""
        cutoff = date(2024, 1, 1)
        after = published_after(self._items(), cutoff)
        before = published_before(self._items(), cutoff)

        assert "unknown" not in [i.id for i in after + before]

    def test_公開日不明は件数を数えられる(self) -> None:
        """黙って捨てず、報告できるようにしておく。"""
        assert [i.id for i in undated(self._items())] == ["unknown"]

    def test_カットオフ当日は以前に入る(self) -> None:
        items = [Item(id="same", question="Q", answer="a", published_at=date(2024, 1, 1))]

        assert published_before(items, date(2024, 1, 1)) == items
        assert published_after(items, date(2024, 1, 1)) == []


class TestUnitHash:
    def test_0以上1未満(self) -> None:
        for i in range(200):
            assert 0.0 <= unit_hash("salt", f"q{i}") < 1.0

    def test_saltが違えば違う値(self) -> None:
        assert unit_hash("a", "q1") != unit_hash("b", "q1")

    def test_区切り文字の衝突が起きない(self) -> None:
        assert unit_hash("a", "bc") != unit_hash("ab", "c")

    def test_おおむね一様(self) -> None:
        values = [unit_hash("salt", f"q{i:05d}") for i in range(2000)]
        below_half = sum(1 for v in values if v < 0.5)

        assert 900 < below_half < 1100


class TestSplitDevHoldout:
    def test_全問がどちらかに入る(self) -> None:
        items = _pool(500)

        dev, holdout = split_dev_holdout(items)

        assert len(dev) + len(holdout) == 500
        assert not ({i.id for i in dev} & {i.id for i in holdout})

    def test_割合がおおむね守られる(self) -> None:
        _, holdout = split_dev_holdout(_pool(2000))

        assert 0.26 < len(holdout) / 2000 < 0.34

    def test_入力の順序に依存しない(self) -> None:
        """★ 実行順で分割が揺れると、DEV の問題が HOLDOUT に混入する。"""
        items = _pool(300)
        forward, _ = split_dev_holdout(items)
        backward, _ = split_dev_holdout(list(reversed(items)))

        assert {i.id for i in forward} == {i.id for i in backward}

    def test_問題を足しても既存の所属は動かない(self) -> None:
        """★ ベンチマークを拡張しても、過去の実験の分割が保たれる。"""
        small_dev, small_holdout = split_dev_holdout(_pool(100))
        large_dev, large_holdout = split_dev_holdout(_pool(400))

        assert {i.id for i in small_dev} <= {i.id for i in large_dev}
        assert {i.id for i in small_holdout} <= {i.id for i in large_holdout}

    def test_割合を変えると所属が変わる(self) -> None:
        _, default_holdout = split_dev_holdout(_pool(500))
        _, wide_holdout = split_dev_holdout(_pool(500), fraction=0.5)

        assert len(wide_holdout) > len(default_holdout)

    def test_既定の割合(self) -> None:
        assert HOLDOUT_FRACTION == 0.30

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_割合が範囲外なら落とす(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            split_dev_holdout(_pool(10), fraction=fraction)


class TestTakeDeterministic:
    def test_件数(self) -> None:
        assert len(take_deterministic(_pool(500), 120)) == 120

    def test_何度やっても同じ集合(self) -> None:
        items = _pool(500)

        assert take_deterministic(items, 50) == take_deterministic(items, 50)

    def test_入力の順序に依存しない(self) -> None:
        items = _pool(500)
        forward = take_deterministic(items, 50)
        backward = take_deterministic(list(reversed(items)), 50)

        assert [i.id for i in forward] == [i.id for i in backward]

    def test_nを増やすと前の集合を含む(self) -> None:
        """★ パイロットを本番の一部として再利用でき、キャッシュも無駄にならない。"""
        items = _pool(500)
        pilot = take_deterministic(items, 40)
        full = take_deterministic(items, 200)

        assert [i.id for i in pilot] == [i.id for i in full[:40]]

    def test_母数を超えたら全部返す(self) -> None:
        assert len(take_deterministic(_pool(10), 50)) == 10

    def test_0件(self) -> None:
        assert take_deterministic(_pool(10), 0) == []

    def test_負の件数を落とす(self) -> None:
        with pytest.raises(ValueError):
            take_deterministic(_pool(10), -1)

    def test_抽出は分割と別のsaltを使う(self) -> None:
        """同じ salt だと「分割境界に近い順」に取ることになり、抽出が分割と相関する。"""
        items = _pool(1000)
        dev, _ = split_dev_holdout(items)
        sampled = take_deterministic(items, len(dev))

        assert {i.id for i in sampled} != {i.id for i in dev}


class TestSplitCrossProcess:
    """★ 分割も別プロセスで再現しなければならない(`hash()` を使っていないこと)。"""

    _SCRIPT = textwrap.dedent(
        """
        import json
        from contamlab.benchmark import Item, split_dev_holdout, take_deterministic

        items = [Item(id=f"q{i:05d}", question=f"Q{i}", answer="a") for i in range(300)]
        dev, holdout = split_dev_holdout(items)
        print(json.dumps({
            "holdout": [i.id for i in holdout],
            "sample": [i.id for i in take_deterministic(items, 20)],
        }))
        """
    )

    def _run(self, hash_seed: str) -> dict:
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONHASHSEED=hash_seed)
        completed = subprocess.run(
            [sys.executable, "-c", self._SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            encoding="utf-8",
        )
        return json.loads(completed.stdout)

    def test_別プロセスでも同じ分割と抽出になる(self) -> None:
        items = _pool(300)
        _, holdout = split_dev_holdout(items)
        expected = {
            "holdout": [i.id for i in holdout],
            "sample": [i.id for i in take_deterministic(items, 20)],
        }

        assert self._run("0") == expected
        assert self._run("12345") == expected
