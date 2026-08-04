"""② 摂動層 — 意味を保ったまま表層だけ変える。【ここだけ書き換えてよい】

jstock-analyzer-v2 の `research/candidate.py` に相当する可変ファイル。
`harness.py` は編集禁止だが、ここは実験のたびに書き換える。

**摂動器を1種類追加するたびに、事前確約の K が1つ消費される。** 何種類も試して
一番落ちたものを報告するのは完全な p-hacking である。追加する前に `preregister.md`
に書くこと。

決定性について:
    組み込みの `hash()` は文字列に対してプロセスごとにランダム化されるため、
    **再現しない。** ここでは必ず `hashlib.sha256` から乱数種を作る。
    同じ (seed, item.id) なら、いつどのマシンで実行しても同じ摂動になる。

意図的に入れていないもの:
    数値・人名の差し替えは **入れていない。** 「太郎は5個のリンゴを…」の 5 を 7 に
    変えると答えも変わるので、答えの再計算が必要になる。再計算を伴う摂動器は、
    壊れていても正答率が下がるだけで、汚染と区別がつかない。**静かに壊れる摂動器は
    汚染検出器として最悪である。** 実装するなら答えの検証機構とセットで入れる。
"""
from __future__ import annotations

import hashlib
import random
from typing import Protocol, runtime_checkable

from .benchmark import Item


def rng_for(seed: str, item_id: str) -> random.Random:
    """(seed, item_id) から決定論的な乱数生成器を作る。

    区切りに NUL を挟むのは、("a", "bc") と ("ab", "c") が同じ種にならないようにするため。
    """
    digest = hashlib.sha256(f"{seed}\x00{item_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@runtime_checkable
class Perturbator(Protocol):
    """摂動器のインターフェース。

    `apply` は **副作用を持たず**、同じ入力に対して常に同じ出力を返さなければならない。
    """

    name: str

    def apply(self, item: Item, seed: str) -> Item: ...


class Identity:
    """何も変えない。**対照条件。**

    これを使って「オリジナル vs オリジナル」を測ると、正答率の差は 0 になるはず。
    ならなければモデルの応答が非決定的(temperature > 0 等)ということなので、
    実験の前提が崩れている。**測定装置の健全性チェックに使う。**
    """

    name = "identity"

    def apply(self, item: Item, seed: str) -> Item:
        return item


class ShuffleChoices:
    """多肢選択の選択肢を一様ランダムに並べ替える。

    **答えを壊さないので安全。** 正解は中身で持っているため、並べ替えても
    `item.answer` は不変で、位置だけが変わる。

    何を検出するか:
        「問題の中身ではなく、正解の**位置**を覚えている」タイプの記憶。
        ベンチマークが公開された並び順ごと訓練データに入っている場合に効く。

    検出力についての注意:
        一様置換なので、4択なら 1/4 の確率で正解が元の位置に残る。そのぶん
        不一致率 ψ が下がり、検出力も下がる。「正解の位置が必ず変わる」変種
        (derangement)にすれば ψ は上がるが、**それは別の摂動器であり、
        K を1つ消費する。** 事前確約なしに切り替えないこと。

    自由記述問題と選択肢が1つ以下の問題は、変えずにそのまま返す。
    """

    name = "shuffle_choices"

    def apply(self, item: Item, seed: str) -> Item:
        if len(item.choices) < 2:
            return item

        choices = list(item.choices)
        rng_for(seed, item.id).shuffle(choices)
        return item.with_choices(
            tuple(choices),
            perturbator=self.name,
            perturbation_seed=seed,
        )


def perturb_all(items: list[Item], perturbator: Perturbator, seed: str) -> list[Item]:
    """全問に摂動をかける。**順序と id を保つ**(対応付けが id で行われるため)。"""
    return [perturbator.apply(item, seed) for item in items]


# CLI から名前で引くための登録表。**ここに足す = K を1つ使う。**
REGISTRY: dict[str, type] = {
    Identity.name: Identity,
    ShuffleChoices.name: ShuffleChoices,
}


def get_perturbator(name: str) -> Perturbator:
    if name not in REGISTRY:
        raise ValueError(f"未知の摂動器: {name}(利用可能: {', '.join(sorted(REGISTRY))})")
    return REGISTRY[name]()
