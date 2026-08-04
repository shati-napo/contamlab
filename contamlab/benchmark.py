"""① ベンチマーク層 — 問題とその「入手可能時点」。

設計上の要点は1つ。**正解は「位置」ではなく「中身」で持つ。**

    ✕ answer_index = 2        選択肢を並べ替えると壊れる
    ○ answer = "水素"          並べ替えても不変

こうしておくと、摂動器が正解を壊していないことが「`answer` が保存されているか」という
1行の不変条件で確かめられる。位置で持つと、摂動のたびに追随処理が要り、そこがバグる。

`published_at` は **as_of** である。「この問題がいつから世に存在するか」であって、
出題日でも収録日でもない。モデルのカットオフと突き合わせる時点法(継続更新型
ベンチマーク)で使う。

DEV / HOLDOUT の分割もここに置く。**分割規則は固定評価系の一部であり、
一度決めたら変更しない**(`program.md` の「変えてはいけない」に入っている)。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# 選択肢のラベル。A, B, C, ... 26択を超えるベンチマークは想定していない。
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ★ 分割の定数。**公開の固定値であり、非公開シードではない。**
#
# 秘匿するのは「摂動シード」であって「どの問題が HOLDOUT か」ではない。分割の割り当ては
# 第三者が同じコードで再現でき、検証できなければならない。ここに非公開シードを混ぜると
# 検証不能になり、「都合のいい分割を選んだのでは」に反論できなくなる。
#
# 値を変えると全問題の所属が入れ替わる。**結果を見た後に変えるのはカンニングである。**
SPLIT_SALT = "contamlab-split-v1"
HOLDOUT_FRACTION = 0.30

# 抽出用は分割用と**別の**salt にする。同じにすると「分割境界に近い順」に取ることになり、
# 抽出が分割と相関する。独立にしておけば、どちらの理屈でも偏らない。
SAMPLE_SALT = "contamlab-sample-v1"


@dataclass(frozen=True)
class Item:
    """1問。

    `choices` が空なら自由記述、そうでなければ多肢選択。多肢選択のとき
    `answer` は **選択肢のいずれかと完全一致していなければならない**。
    """

    id: str
    question: str
    answer: str
    choices: tuple[str, ...] = ()
    published_at: date | None = None
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id が空")
        if not self.question.strip():
            raise ValueError(f"question が空: id={self.id}")
        if not self.answer.strip():
            raise ValueError(f"answer が空: id={self.id}")
        if self.choices:
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(f"選択肢が重複している: id={self.id}")
            if self.answer not in self.choices:
                raise ValueError(
                    f"正解が選択肢に無い: id={self.id} answer={self.answer!r}"
                )
            if len(self.choices) > len(CHOICE_LABELS):
                raise ValueError(f"選択肢が多すぎる({len(self.choices)}): id={self.id}")

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)

    @property
    def answer_index(self) -> int | None:
        """現在の並び順における正解の位置。**保存すべき値ではない。** 採点用の導出値。"""
        return self.choices.index(self.answer) if self.choices else None

    @property
    def answer_label(self) -> str | None:
        index = self.answer_index
        return CHOICE_LABELS[index] if index is not None else None

    def with_choices(self, choices: tuple[str, ...], **extra_metadata: Any) -> Item:
        """選択肢を差し替えた複製を返す。`answer` は中身のまま持ち回るので壊れない。

        `id` は変えない。**オリジナルと摂動版の対応付けは id で行う。**
        """
        merged = dict(self.metadata)
        merged.update(extra_metadata)
        return Item(
            id=self.id,
            question=self.question,
            answer=self.answer,
            choices=choices,
            published_at=self.published_at,
            source=self.source,
            metadata=merged,
        )


def load_jsonl(path: Path) -> list[Item]:
    """JSONL からベンチマークを読む。1行1問。

    想定するキー:
        id, question, answer, choices, published_at (YYYY-MM-DD), source, metadata

    行番号つきで落とす。数百件のファイルで「どこかが変」と言われても直せないため。
    """
    items: list[Item] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON として読めない: {exc}") from exc

            try:
                item = _item_from_dict(raw)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number} {exc}") from exc

            if item.id in seen:
                raise ValueError(f"{path}:{line_number} id が重複: {item.id}")
            seen.add(item.id)
            items.append(item)

    if not items:
        raise ValueError(f"問題が1件も無い: {path}")
    return items


def _item_from_dict(raw: Mapping[str, Any]) -> Item:
    published = raw.get("published_at")
    return Item(
        id=str(raw["id"]),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        choices=tuple(str(c) for c in raw.get("choices", ())),
        published_at=date.fromisoformat(published) if published else None,
        source=str(raw.get("source", "")),
        metadata=dict(raw.get("metadata", {})),
    )


def published_after(items: Iterable[Item], cutoff: date) -> list[Item]:
    """カットオフより**後**に公開された問題だけを返す(時点法の本体)。

    `published_at` が不明な問題は **除外する。** 「たぶん新しい」で通すと、
    そこから汚染が入る。時点法は日付が分かっている問題にしか適用できない。
    """
    return [i for i in items if i.published_at is not None and i.published_at > cutoff]


def published_before(items: Iterable[Item], cutoff: date) -> list[Item]:
    """カットオフ以前に公開された問題だけを返す(汚染されている側の対照群)。"""
    return [i for i in items if i.published_at is not None and i.published_at <= cutoff]


def undated(items: Iterable[Item]) -> list[Item]:
    """公開日が不明な問題。**件数を報告するために取る。** 黙って捨てないこと。"""
    return [i for i in items if i.published_at is None]


# --------------------------------------------------------------------------
# DEV / HOLDOUT 分割 ★ 固定評価系の一部。定義後は変更しない
# --------------------------------------------------------------------------


def unit_hash(salt: str, item_id: str) -> float:
    """(salt, id) を [0, 1) の一様な値に潰す。

    組み込みの `hash()` は文字列に対してプロセスごとにランダム化されるので使わない
    (`perturb.rng_for` と同じ理由)。区切りの NUL は ("a", "bc") と ("ab", "c") が
    同じ値にならないようにするため。
    """
    digest = hashlib.sha256(f"{salt}\x00{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_dev_holdout(
    items: Sequence[Item],
    fraction: float = HOLDOUT_FRACTION,
    salt: str = SPLIT_SALT,
) -> tuple[list[Item], list[Item]]:
    """(DEV, HOLDOUT) に分ける。

    **id だけで決まる。** 入力の順序にも件数にも依存しないので、問題を後から足しても
    既存の問題の所属は動かない。ここが「実行順に依存しない」の実質的な意味であり、
    ランのたびに分割が揺れて DEV の問題が HOLDOUT に混入する事故を防いでいる。

    HOLDOUT は摂動器の設計中に**一度も見てはいけない。** 見た時点でそれは DEV である。
    CLI の `--split` 既定値が `dev` なのはこのため。
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"HOLDOUT の割合は (0, 1) の範囲: {fraction}")

    dev: list[Item] = []
    holdout: list[Item] = []
    for item in items:
        (holdout if unit_hash(salt, item.id) < fraction else dev).append(item)
    return dev, holdout


def take_deterministic(
    items: Sequence[Item], n: int, salt: str = SAMPLE_SALT
) -> list[Item]:
    """決定論的に n 問だけ抜く。**検出力で決めた標本サイズを取り出すのに使う。**

    全問に API を投げると金が足りないので部分集合を使うことになるが、その選び方が
    実験のたびに変わると「都合のいい部分集合を引くまで回した」と区別がつかない。
    id のハッシュ順に取ることで、**誰が何回やっても同じ n 問**になる。

    n を増やしたときに前の集合が保たれる(prefix になる)ので、パイロットを本番の
    一部として再利用でき、キャッシュも無駄にならない。

    ハッシュが衝突した場合は id で並べる。順序が実行ごとに揺れないようにするため。
    """
    if n < 0:
        raise ValueError(f"件数は 0 以上: {n}")

    ordered = sorted(items, key=lambda i: (unit_hash(salt, i.id), i.id))
    return ordered[:n]
