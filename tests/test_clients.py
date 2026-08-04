"""実 API クライアント — 偽トランスポートで検査する。**ネットワークに触らない。**

課金が発生する層なので、送っている中身・リトライ・予算の上限を全部固定しておく。
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from contamlab.clients import (
    AnthropicModel,
    ApiError,
    BudgetExceededError,
    CallBudget,
    ClientOptions,
    OpenAICompatibleModel,
    RateLimiter,
    RetryPolicy,
    build_api_model,
    is_api_spec,
)
from contamlab.runner import Model


class FakeTransport:
    """応答を順に返す。要素が尽きたら最後の応答を繰り返す。"""

    def __init__(self, *responses) -> None:
        self.calls: list[dict] = []
        self._queue = list(responses)
        self._last = responses[-1] if responses else {}

    def post_json(self, url, payload, headers, timeout):
        self.calls.append(
            {"url": url, "payload": payload, "headers": dict(headers), "timeout": timeout}
        )
        item = self._queue.pop(0) if self._queue else self._last
        if isinstance(item, Exception):
            raise item
        return item


def _http_error(code: int, body: str = "エラー本文") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test", code, "err", {}, io.BytesIO(body.encode("utf-8"))
    )


class RecordingSleeper:
    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _anthropic(transport, budget=None, **kwargs) -> AnthropicModel:
    return AnthropicModel(
        "claude", "claude-opus-5", "test-key",
        budget=budget or CallBudget(100), transport=transport, **kwargs
    )


def _openai(transport, budget=None, **kwargs) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        "gpt", "gpt-test", "test-key",
        budget=budget or CallBudget(100), transport=transport, **kwargs
    )


class TestAnthropic:
    def test_プロトコルを満たす(self) -> None:
        assert isinstance(_anthropic(FakeTransport({})), Model)

    def test_応答からテキストを取り出す(self) -> None:
        transport = FakeTransport({"content": [{"type": "text", "text": "  A  "}]})

        assert _anthropic(transport).answer("Q") == "A"

    def test_テキスト以外のブロックは無視する(self) -> None:
        transport = FakeTransport(
            {"content": [{"type": "thinking", "thinking": "…"}, {"type": "text", "text": "B"}]}
        )

        assert _anthropic(transport).answer("Q") == "B"

    def test_中身が無ければ空文字(self) -> None:
        assert _anthropic(FakeTransport({"content": []})).answer("Q") == ""

    def test_送信内容(self) -> None:
        transport = FakeTransport({"content": [{"type": "text", "text": "A"}]})
        _anthropic(transport).answer("これは問題文")

        call = transport.calls[0]
        assert call["url"] == "https://api.anthropic.com/v1/messages"
        assert call["payload"]["model"] == "claude-opus-5"
        assert call["payload"]["messages"] == [{"role": "user", "content": "これは問題文"}]
        assert call["headers"]["x-api-key"] == "test-key"
        assert call["headers"]["anthropic-version"] == "2023-06-01"

    def test_温度は既定で0(self) -> None:
        """★ 非決定的な応答は、汚染でないのに不一致ペアを生んで検出力を食う。"""
        transport = FakeTransport({"content": [{"type": "text", "text": "A"}]})
        _anthropic(transport).answer("Q")

        assert transport.calls[0]["payload"]["temperature"] == 0.0


class TestOpenAICompatible:
    def test_応答からテキストを取り出す(self) -> None:
        transport = FakeTransport({"choices": [{"message": {"content": " C "}}]})

        assert _openai(transport).answer("Q") == "C"

    def test_choicesが空なら空文字(self) -> None:
        assert _openai(FakeTransport({"choices": []})).answer("Q") == ""

    def test_contentがnullでも落ちない(self) -> None:
        transport = FakeTransport({"choices": [{"message": {"content": None}}]})

        assert _openai(transport).answer("Q") == ""

    def test_既定は本家のURL(self) -> None:
        transport = FakeTransport({"choices": [{"message": {"content": "A"}}]})
        _openai(transport).answer("Q")

        assert transport.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"

    def test_base_urlを差し替えられる(self) -> None:
        transport = FakeTransport({"choices": [{"message": {"content": "A"}}]})
        model = OpenAICompatibleModel(
            "local", "llama", "", budget=CallBudget(10), transport=transport,
            base_url="http://localhost:11434/v1/",
        )

        model.answer("Q")

        assert transport.calls[0]["url"] == "http://localhost:11434/v1/chat/completions"

    def test_キーが空なら認証ヘッダを付けない(self) -> None:
        """ローカルモデルはキー不要なことが多い。"""
        transport = FakeTransport({"choices": [{"message": {"content": "A"}}]})
        OpenAICompatibleModel(
            "local", "llama", "", budget=CallBudget(10), transport=transport
        ).answer("Q")

        assert "Authorization" not in transport.calls[0]["headers"]


class TestRetry:
    def test_429は再試行して成功する(self) -> None:
        transport = FakeTransport(
            _http_error(429), _http_error(503),
            {"content": [{"type": "text", "text": "A"}]},
        )
        sleeper = RecordingSleeper()

        result = _anthropic(transport, sleeper=sleeper).answer("Q")

        assert result == "A"
        assert len(transport.calls) == 3

    def test_バックオフは指数的に伸びる(self) -> None:
        transport = FakeTransport(
            _http_error(429), _http_error(429),
            {"content": [{"type": "text", "text": "A"}]},
        )
        sleeper = RecordingSleeper()

        _anthropic(transport, sleeper=sleeper).answer("Q")

        assert sleeper.waits == [1.0, 2.0]

    def test_バックオフに上限がある(self) -> None:
        transport = FakeTransport(_http_error(429))
        sleeper = RecordingSleeper()
        model = _anthropic(
            transport,
            retry=RetryPolicy(max_attempts=8, initial_backoff=10.0, max_backoff=30.0),
            sleeper=sleeper,
        )

        with pytest.raises(ApiError):
            model.answer("Q")

        assert max(sleeper.waits) == 30.0

    def test_再試行しても駄目なら諦める(self) -> None:
        transport = FakeTransport(_http_error(503))
        model = _anthropic(
            transport, retry=RetryPolicy(max_attempts=3), sleeper=RecordingSleeper()
        )

        with pytest.raises(ApiError, match="3 回試して失敗"):
            model.answer("Q")

        assert len(transport.calls) == 3

    def test_401は即座に諦める(self) -> None:
        """認証エラーを再試行しても無駄。**しかも課金上限を空回りで食う。**"""
        transport = FakeTransport(_http_error(401, "invalid api key"))

        with pytest.raises(ApiError, match="HTTP 401"):
            _anthropic(transport, sleeper=RecordingSleeper()).answer("Q")

        assert len(transport.calls) == 1

    def test_ネットワーク断も再試行する(self) -> None:
        transport = FakeTransport(
            urllib.error.URLError("接続できない"),
            {"content": [{"type": "text", "text": "A"}]},
        )

        assert _anthropic(transport, sleeper=RecordingSleeper()).answer("Q") == "A"


class TestBudget:
    def test_上限に達したら止まる(self) -> None:
        """★ 走らせっぱなしで請求が来る事故を、コード側で止める。"""
        budget = CallBudget(max_calls=2)
        model = _anthropic(
            FakeTransport({"content": [{"type": "text", "text": "A"}]}), budget=budget
        )

        model.answer("Q1")
        model.answer("Q2")

        with pytest.raises(BudgetExceededError, match="上限 2 回"):
            model.answer("Q3")

    def test_モデル間で共有される(self) -> None:
        budget = CallBudget(max_calls=2)
        transport = FakeTransport({"content": [{"type": "text", "text": "A"}]})
        a = _anthropic(transport, budget=budget)
        b = _anthropic(transport, budget=budget)

        a.answer("Q")
        b.answer("Q")

        with pytest.raises(BudgetExceededError):
            a.answer("Q")

    def test_残りを数えられる(self) -> None:
        budget = CallBudget(max_calls=3)
        _anthropic(
            FakeTransport({"content": [{"type": "text", "text": "A"}]}), budget=budget
        ).answer("Q")

        assert budget.used == 1
        assert budget.remaining == 2

    def test_失敗した呼び出しも1回として数える(self) -> None:
        """リトライで空回りしても、上限の意味が消えないようにする。"""
        budget = CallBudget(max_calls=5)
        model = _anthropic(
            FakeTransport(_http_error(503)),
            budget=budget,
            retry=RetryPolicy(max_attempts=2),
            sleeper=RecordingSleeper(),
        )

        with pytest.raises(ApiError):
            model.answer("Q")

        assert budget.used == 1


class TestRateLimiter:
    def test_間隔が空いていれば待たない(self) -> None:
        sleeper = RecordingSleeper()
        limiter = RateLimiter(60, clock=lambda: 0.0, sleeper=sleeper)

        limiter.wait()

        assert sleeper.waits == []

    def test_間隔が足りなければ待つ(self) -> None:
        times = iter([0.0, 0.0, 0.2, 1.0])
        sleeper = RecordingSleeper()
        limiter = RateLimiter(60, clock=lambda: next(times), sleeper=sleeper)

        limiter.wait()
        limiter.wait()

        assert sleeper.waits == pytest.approx([0.8])

    def test_未設定なら待たない(self) -> None:
        sleeper = RecordingSleeper()
        limiter = RateLimiter(None, clock=lambda: 0.0, sleeper=sleeper)

        limiter.wait()
        limiter.wait()

        assert sleeper.waits == []


class TestBuildApiModel:
    def _options(self, **env) -> ClientOptions:
        return ClientOptions(budget=CallBudget(10), transport=FakeTransport({}), env=env)

    def test_anthropic(self) -> None:
        model = build_api_model(
            "anthropic:claude:claude-opus-5", self._options(ANTHROPIC_API_KEY="k")
        )

        assert isinstance(model, AnthropicModel)
        assert model.name == "claude"
        assert model.model_id == "claude-opus-5"

    def test_openai(self) -> None:
        model = build_api_model("openai:gpt:gpt-4o", self._options(OPENAI_API_KEY="k"))

        assert isinstance(model, OpenAICompatibleModel)
        assert model.base_url == "https://api.openai.com/v1"

    def test_compat_はコロンを含むURLを扱える(self) -> None:
        model = build_api_model(
            "compat:local:swallow:http://localhost:11434/v1", self._options()
        )

        assert model.name == "local"
        assert model.model_id == "swallow"
        assert model.base_url == "http://localhost:11434/v1"

    def test_APIキーが無ければ落とす(self) -> None:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            build_api_model("anthropic:claude:claude-opus-5", self._options())

    def test_未知の種別を落とす(self) -> None:
        with pytest.raises(ValueError, match="未知のモデル種別"):
            build_api_model("bedrock:x:y", self._options())

    @pytest.mark.parametrize(
        "spec", ["anthropic:claude", "openai:gpt", "compat:a:b", "anthropic::model"]
    )
    def test_形式が違えば落とす(self, spec: str) -> None:
        options = self._options(ANTHROPIC_API_KEY="k", OPENAI_API_KEY="k")

        with pytest.raises(ValueError, match="モデル指定が読めない"):
            build_api_model(spec, options)

    def test_オプションが伝わる(self) -> None:
        options = ClientOptions(
            budget=CallBudget(10),
            transport=FakeTransport({}),
            temperature=0.7,
            max_tokens=32,
            env={"OPENAI_API_KEY": "k"},
        )

        model = build_api_model("openai:gpt:gpt-4o", options)

        assert model.temperature == 0.7
        assert model.max_tokens == 32


class TestIsApiSpec:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("fake:x:0.5", False),
            ("anthropic:c:claude-opus-5", True),
            ("openai:g:gpt-4o", True),
            ("compat:l:m:http://x", True),
        ],
    )
    def test_課金される指定かどうか(self, spec: str, expected: bool) -> None:
        assert is_api_spec(spec) is expected
