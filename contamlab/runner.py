"""③ 実行層 — モデルに問い、応答を採点し、キャッシュする。

3つの規律をここに閉じ込めている。

1. **キャッシュは追記専用。** 一度得た応答は書き換えない(jsav2 の
   「`data/processed/` の parquet は消さない」と同じ)。同じ問いに違う応答が来たら
   それは**モデルが非決定的**という重大事なので、上書きせず衝突として記録する。

2. **採点はオリジナルと摂動版で完全に同一。** 採点規則が条件によって違えば、
   正答率の差は汚染ではなく採点のブレになる。同じ関数を通す。

3. **解釈できなかった応答は不正解として数え、かつ件数を残す。** 解釈不能率が
   条件間で大きく違えば比較が壊れているので、報告側でそれを見られるようにする。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

from .benchmark import CHOICE_LABELS, Item

# 「A」「(A)」「A.」「答え: A」「答えはA」「正解はAです」のような先頭のラベルを拾う。
#
# 2つの経路に分かれている(2026-08-07)。
#
# 1. **接頭辞あり**(`答え`/`正解` 等が実際に書かれている): ラベルの後ろは問わない。
#    「正解はAです。…」のように日本語の文末表現が続いても拾う。接頭辞が「これはラベルだ」と
#    宣言しているので、後ろの区切りに頼る必要がない。
# 2. **接頭辞なし**(いきなり英字1文字): ラベルの後ろに区切り文字・空白・文末を要求する。
#    緩めると「B は正しくない」のような**否定文の主語**を選択と誤読する。
#
# どちらの経路もラベル直後の**英数字は許さない**。ここを開けると "Hydrogen" の H を
# ラベルと誤読する。
#
# 経路1を後から足したのは、docstring が「`答えはA`」を拾うと宣言していたのに実装が
# ラベル直後に日本語を許さず、「正解はDです」で外れていたためである(preregister.md の
# 変更履歴 2026-08-07)。
#
# ⚠️ **経路2を変えていないから既存の判定も変わらない、とは言えない。** 手元のキャッシュ
# 942 応答で突き合わせたところ、33 件が新たに読めるようになった一方で **2 件は別の選択肢に
# 移った。** 規則3が当たるようになると、従来その応答を担当していた**下流の規則4の
# 出番が消える**ためである(どちらもモデルが「正解はDです」と明言しつつ本文で別の選択肢に
# 言及した自己矛盾応答で、規則3を優先する新判定のほうが docstring の宣言に忠実)。
_LABEL_PREFIX = r"(?:答え|回答|正解|正しい選択肢|Answer|answer)\s*[はがも]?\s*[:：]?\s*"
_OPEN_BRACKET = r"[（(\[「『]?"
_LEADING_LABEL = re.compile(
    rf"^\s*(?:{_LABEL_PREFIX}{_OPEN_BRACKET}([A-Za-z])(?![A-Za-z0-9])"
    rf"|{_OPEN_BRACKET}([A-Za-z])(?:[)）\]」』\.、,:：\s]|$))",
)
_WHITESPACE = re.compile(r"\s+")


def _leading_label(text: str) -> str | None:
    """先頭のラベル1文字。どちらの経路で当たったかを呼び出し側に見せない。"""
    match = _LEADING_LABEL.match(text)
    if match is None:
        return None
    return match.group(1) or match.group(2)


def normalize(text: str) -> str:
    """比較用の正規化。前後の空白を落とし、内部の連続空白を1つに潰す。"""
    return _WHITESPACE.sub(" ", text.strip())


def format_prompt(item: Item) -> str:
    """モデルに投げる文面。**問題文を先頭に置く**(応答から問題を逆引きするため)。

    オリジナルと摂動版で構造が完全に同じであること。ここが違うと、正答率の差が
    汚染ではなく書式の差になる。
    """
    if not item.is_multiple_choice:
        return f"{item.question}\n\n答えだけを簡潔に書いてください。"

    lines = [item.question, ""]
    for label, choice in zip(CHOICE_LABELS, item.choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("正しい選択肢の記号だけを答えてください。")
    return "\n".join(lines)


@dataclass(frozen=True)
class Response:
    item_id: str
    raw: str
    selected: str | None
    correct: bool

    @property
    def parsed(self) -> bool:
        """応答から選択を読み取れたか。**読み取れないのに不正解と数えた件数は報告する。**"""
        return self.selected is not None


def select(item: Item, raw: str) -> str | None:
    """応答から「どれを選んだか」を読み取る。読み取れなければ None。

    `item` は**提示された形**(摂動後なら摂動後の並び)であること。ラベルは提示順で
    解釈される。

    順に試す:
        1. 応答が選択肢の1つと完全一致
        2. 応答がラベル1文字だけ
        3. 応答の先頭がラベル(「A.」「(A)」「答え: A」「正解はAです」など)
        4. 選択肢のうち**ちょうど1つ**が応答に含まれる
    """
    text = normalize(raw)
    if not text:
        return None

    if not item.is_multiple_choice:
        return text

    normalized_choices = [normalize(c) for c in item.choices]

    # 1. 完全一致
    for choice, normalized in zip(item.choices, normalized_choices):
        if text.casefold() == normalized.casefold():
            return choice

    # 2. ラベル1文字だけ
    if len(text) == 1 and text.upper() in CHOICE_LABELS:
        index = CHOICE_LABELS.index(text.upper())
        if index < len(item.choices):
            return item.choices[index]

    # 3. 先頭のラベル
    label = _leading_label(text)
    if label and label.upper() in CHOICE_LABELS:
        index = CHOICE_LABELS.index(label.upper())
        if index < len(item.choices):
            return item.choices[index]

    # 4. 選択肢が応答に含まれる。複数当たっても、それらが入れ子(「H」と「He」)なら
    #    最も長いものを採る。入れ子でなければ曖昧なので None(= 解釈不能)にする。
    folded = text.casefold()
    contained = [
        choice
        for choice, normalized in zip(item.choices, normalized_choices)
        if normalized and normalized.casefold() in folded
    ]
    if contained:
        longest = max(contained, key=len)
        longest_folded = normalize(longest).casefold()
        if all(normalize(c).casefold() in longest_folded for c in contained):
            return longest

    return None


def grade(item: Item, raw: str) -> Response:
    """採点する。**オリジナルと摂動版で同じこの関数を通すこと。**"""
    selected = select(item, raw)
    if selected is None:
        correct = False
    elif item.is_multiple_choice:
        correct = selected == item.answer
    else:
        correct = normalize(selected).casefold() == normalize(item.answer).casefold()
    return Response(item_id=item.id, raw=raw, selected=selected, correct=correct)


@runtime_checkable
class Model(Protocol):
    """モデルのインターフェース。実 API クライアントはここに差し込む。

    **temperature は 0 相当にすること。** 非決定的な応答は、汚染ではないのに
    不一致ペアを生み、検出力を無駄に食う。
    """

    name: str

    def answer(self, prompt: str) -> str: ...


class ResponseCache:
    """追記専用の応答キャッシュ(JSONL)。

    実験のたびに API を叩き直すと金がかかるうえ、モデル側の更新で過去の数字が
    再現しなくなる。一度得た応答は消さない。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, str] = {}
        self.conflicts: list[str] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_number} 壊れている: {exc}") from exc
                self._entries.setdefault(record["key"], record["response"])

    @staticmethod
    def make_key(model_name: str, prompt: str) -> str:
        return hashlib.sha256(f"{model_name}\x00{prompt}".encode("utf-8")).hexdigest()

    def get(self, model_name: str, prompt: str) -> str | None:
        return self._entries.get(self.make_key(model_name, prompt))

    def put(self, model_name: str, prompt: str, response: str) -> bool:
        """新規なら書き込んで True。既存なら**上書きせず** False。

        既存と中身が違えば `conflicts` に記録する。モデルが非決定的である証拠なので、
        黙って握り潰さない。
        """
        key = self.make_key(model_name, prompt)
        existing = self._entries.get(key)
        if existing is not None:
            if existing != response:
                self.conflicts.append(key)
            return False

        self._entries[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"key": key, "model": model_name, "prompt": prompt, "response": response},
                    ensure_ascii=False,
                )
                + "\n"
            )
        return True

    def __len__(self) -> int:
        return len(self._entries)


class CachedModel:
    """モデルをキャッシュで包む。キャッシュにあれば API を叩かない。"""

    def __init__(self, model: Model, cache: ResponseCache) -> None:
        self._model = model
        self._cache = cache
        self.name = model.name
        self.api_calls = 0

    def answer(self, prompt: str) -> str:
        cached = self._cache.get(self.name, prompt)
        if cached is not None:
            return cached
        response = self._model.answer(prompt)
        self.api_calls += 1
        self._cache.put(self.name, prompt, response)
        return response


class FakeModel:
    """テスト用のモデル。**汚染をパラメータで注入できる。**

    `memorized_ids` の問題は、**オリジナルの提示形と完全一致したときだけ**必ず正解する。
    摂動版は表層が変わるので記憶が発火せず、素の能力(`base_accuracy`)に落ちる。
    これが「暗記」の最小限のモデル化である。

    用途は `verify_cache.py` と同じ ——「検出器が、汚染があるときに検出し、
    無いときに検出しないこと」を確かめる。**測定装置が壊れていれば全実験が無価値。**
    """

    def __init__(
        self,
        name: str,
        items: Sequence[Item],
        base_accuracy: float = 0.5,
        memorized_ids: Iterable[str] = (),
        seed: str = "fake",
    ) -> None:
        if not 0.0 <= base_accuracy <= 1.0:
            raise ValueError(f"素の正答率は [0, 1] の範囲: {base_accuracy}")

        self.name = name
        self.base_accuracy = base_accuracy
        self.memorized_ids = frozenset(memorized_ids)
        self.seed = seed
        # 問題文はどの摂動でも保存されるので、これで逆引きできる。
        # 短い問題文が長い問題文の一部になっている場合に備えて長い順に見る。
        self._items = sorted(items, key=lambda i: len(i.question), reverse=True)
        self._original_prompts = {item.id: format_prompt(item) for item in items}

    def _lookup(self, prompt: str) -> Item | None:
        for item in self._items:
            if item.question in prompt:
                return item
        return None

    def answer(self, prompt: str) -> str:
        item = self._lookup(prompt)
        if item is None:
            return ""

        if item.id in self.memorized_ids and prompt == self._original_prompts[item.id]:
            return item.answer

        digest = hashlib.sha256(f"{self.seed}\x00{self.name}\x00{prompt}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        if draw < self.base_accuracy:
            return item.answer

        wrong = [c for c in item.choices if c != item.answer]
        if not wrong:
            return "(不明)"
        return wrong[int.from_bytes(digest[8:16], "big") % len(wrong)]


def run_items(model: Model, items: Sequence[Item]) -> list[Response]:
    """全問を解かせて採点する。**入力の順序を保つ**(対応付けに使うため)。"""
    return [grade(item, model.answer(format_prompt(item))) for item in items]
