"""② 摂動層 — 決定性と、正解の保存。

このファイルの最重要テストは `test_別プロセスでも同じ摂動になる`。
組み込み `hash()` は文字列に対してプロセスごとにランダム化されるため、それを使った
実装は同一プロセス内のテストを全部通過したうえで、再現性だけを静かに失う。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from contamlab.benchmark import Item
from contamlab.perturb import (
    REGISTRY,
    Identity,
    Perturbator,
    ShuffleChoices,
    get_perturbator,
    perturb_all,
    rng_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _item(item_id: str = "q1", n_choices: int = 4) -> Item:
    choices = tuple(f"選択肢{i}" for i in range(n_choices))
    return Item(id=item_id, question=f"{item_id} の問題文", answer=choices[0], choices=choices)


class TestRngFor:
    def test_同じ入力なら同じ乱数列(self) -> None:
        a = [rng_for("s", "q1").random() for _ in range(3)]
        b = [rng_for("s", "q1").random() for _ in range(3)]

        assert a == b

    def test_シードが違えば違う(self) -> None:
        assert rng_for("s1", "q1").random() != rng_for("s2", "q1").random()

    def test_問題が違えば違う(self) -> None:
        assert rng_for("s", "q1").random() != rng_for("s", "q2").random()

    def test_区切り文字の衝突が起きない(self) -> None:
        """("a", "bc") と ("ab", "c") が同じ種になってはいけない。"""
        assert rng_for("a", "bc").random() != rng_for("ab", "c").random()


class TestIdentity:
    def test_何も変えない(self) -> None:
        item = _item()

        assert Identity().apply(item, "seed") is item

    def test_プロトコルを満たす(self) -> None:
        assert isinstance(Identity(), Perturbator)


class TestShuffleChoices:
    def test_プロトコルを満たす(self) -> None:
        assert isinstance(ShuffleChoices(), Perturbator)

    def test_正解の中身が保存される(self) -> None:
        """★ 不変条件。これが破れたら摂動器が壊れている。"""
        for i in range(50):
            item = _item(f"q{i}")
            assert ShuffleChoices().apply(item, "seed").answer == item.answer

    def test_選択肢の集合が保存される(self) -> None:
        item = _item()

        result = ShuffleChoices().apply(item, "seed")

        assert sorted(result.choices) == sorted(item.choices)

    def test_idと問題文は変わらない(self) -> None:
        item = _item()

        result = ShuffleChoices().apply(item, "seed")

        assert result.id == item.id
        assert result.question == item.question

    def test_同じシードなら同じ並び(self) -> None:
        item = _item()

        assert (
            ShuffleChoices().apply(item, "seed-1").choices
            == ShuffleChoices().apply(item, "seed-1").choices
        )

    def test_シードが違えば並びも変わりうる(self) -> None:
        item = _item(n_choices=6)
        orders = {ShuffleChoices().apply(item, f"seed-{i}").choices for i in range(20)}

        assert len(orders) > 1

    def test_摂動の記録がメタデータに残る(self) -> None:
        result = ShuffleChoices().apply(_item(), "seed-1")

        assert result.metadata["perturbator"] == "shuffle_choices"
        assert result.metadata["perturbation_seed"] == "seed-1"

    @pytest.mark.parametrize("n_choices", [0, 1])
    def test_選択肢が1つ以下なら変えない(self, n_choices: int) -> None:
        if n_choices == 0:
            item = Item(id="q1", question="自由記述の問題", answer="答え")
        else:
            item = Item(id="q1", question="Q", answer="唯一", choices=("唯一",))

        assert ShuffleChoices().apply(item, "seed") is item

    def test_一様置換なので正解が元の位置に残ることもある(self) -> None:
        """★ 検出力に効く性質。4択なら約1/4は動かない。

        「正解の位置が必ず変わる」変種は**別の摂動器**であり、事前確約の K を
        1つ消費する。黙って差し替えないための備忘テスト。
        """
        stayed = sum(
            1
            for i in range(200)
            if ShuffleChoices().apply(_item(f"q{i}"), "seed").answer_index == 0
        )

        assert 20 < stayed < 80, f"4択の一様置換なら約50件のはず: {stayed}"


class TestPerturbAll:
    def test_順序とidを保つ(self) -> None:
        items = [_item(f"q{i}") for i in range(5)]

        result = perturb_all(items, ShuffleChoices(), "seed")

        assert [i.id for i in result] == [i.id for i in items]

    def test_全件の正解が保存される(self) -> None:
        items = [_item(f"q{i}") for i in range(20)]

        result = perturb_all(items, ShuffleChoices(), "seed")

        assert [i.answer for i in result] == [i.answer for i in items]


class TestRegistry:
    def test_名前で引ける(self) -> None:
        assert isinstance(get_perturbator("shuffle_choices"), ShuffleChoices)
        assert isinstance(get_perturbator("identity"), Identity)

    def test_未知の名前は候補つきで落とす(self) -> None:
        with pytest.raises(ValueError, match="利用可能"):
            get_perturbator("存在しない摂動器")

    def test_登録されているのは2種類だけ(self) -> None:
        """★ 摂動器を足す = 事前確約の K を1つ使う。黙って増やさないための番人。"""
        assert set(REGISTRY) == {"identity", "shuffle_choices"}


class TestCrossProcessReproducibility:
    """★ このファイルで最重要のテスト。

    Python の文字列 `hash()` はプロセスごとにランダム化される。それを使った実装は
    同一プロセス内では完全に決定論的に見えるので、他のテストを全部すり抜ける。
    別プロセスで、しかも `PYTHONHASHSEED` を変えて確かめる。
    """

    _SCRIPT = textwrap.dedent(
        """
        import json
        from contamlab.benchmark import Item
        from contamlab.perturb import ShuffleChoices

        item = Item(
            id="q1",
            question="q1 の問題文",
            answer="選択肢0",
            choices=("選択肢0", "選択肢1", "選択肢2", "選択肢3"),
        )
        print(json.dumps(list(ShuffleChoices().apply(item, "seed-1").choices)))
        """
    )

    def _run(self, hash_seed: str) -> list[str]:
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

    def test_別プロセスでも同じ摂動になる(self) -> None:
        expected = list(ShuffleChoices().apply(_item(), "seed-1").choices)

        assert self._run("0") == expected
        assert self._run("1") == expected
        assert self._run("12345") == expected
