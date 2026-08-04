"""実 API クライアント。**ここだけが金を使う。**

依存パッケージを持たない方針を守るため、HTTP は `urllib.request` で書いている。
SDK を入れないのは、この層が課金と再現性の要だからでもある —— 何が送られて何が
返っているかを全部読める状態にしておきたい。

## この層が引き受けている4つの安全装置

1. **予算の上限。** `CallBudget` が API 呼び出し回数を数え、上限で例外を投げる。
   走らせっぱなしで請求が来る事故を、コード側で止める。
2. **レート制限。** 毎分の上限を守って待つ。429 を叩き続けない。
3. **リトライ。** 429 と 5xx は指数バックオフで再試行する。ネットワークの揺れで
   実験全体が落ちないように。
4. **温度は 0 固定が既定。** 非決定的な応答は、汚染でないのに不一致ペアを生み、
   検出力をただ食う。これは実験の前提条件であって好みの問題ではない。

## 測定条件は事前確約の一部である

`temperature` / `max_tokens` / 拡張思考の有無を変えると、**測っているものが変わる。**
同じモデルでも別の測定になるので、`preregister.md` に書いてから変えること。
既定値を黙って動かさない。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# 既定値。**変えると測っているものが変わる。**
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 120.0


class ApiError(RuntimeError):
    """API 呼び出しが最終的に失敗した。"""


class BudgetExceededError(RuntimeError):
    """API 呼び出しの上限に達した。**実験を止める。**"""


@dataclass
class CallBudget:
    """API 呼び出し回数の上限。**モデル間で共有する。**

    `contamlab run` は実行前に必要な呼び出し回数を計算し、その値を上限にする。
    想定より多く呼ばれたら、それはどこかが壊れている合図なので止める。
    """

    max_calls: int
    used: int = 0

    def consume(self) -> None:
        if self.used >= self.max_calls:
            raise BudgetExceededError(
                f"API 呼び出しが上限 {self.max_calls} 回に達した。"
                "想定より多く呼ばれている = どこかが壊れている可能性がある。"
            )
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 60.0
    # 429(レート制限)、529(過負荷)、5xx は再試行する。4xx の大半は再試行しても無駄。
    retry_status: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class RateLimiter:
    """毎分の呼び出し回数を守る。呼び出しの間隔を空けるだけの素朴な実装。"""

    def __init__(
        self,
        per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / per_minute if per_minute and per_minute > 0 else 0.0
        self._clock = clock
        self._sleeper = sleeper
        self._last: float | None = None

    def wait(self) -> None:
        if self._interval <= 0.0:
            return
        now = self._clock()
        if self._last is not None:
            elapsed = now - self._last
            if elapsed < self._interval:
                self._sleeper(self._interval - elapsed)
        self._last = self._clock()


@runtime_checkable
class HttpTransport(Protocol):
    """HTTP の口。テストでは偽物を差し込んでネットワークに触らない。"""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict: ...


class UrllibTransport:
    """標準ライブラリだけの HTTP。"""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class _BaseApiModel:
    """API モデルの共通部分。呼び出し回数・レート制限・リトライをここで面倒みる。"""

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: str,
        budget: CallBudget,
        transport: HttpTransport | None = None,
        rate_limiter: RateLimiter | None = None,
        retry: RetryPolicy | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._api_key = api_key
        self._budget = budget
        self._transport = transport or UrllibTransport()
        self._limiter = rate_limiter or RateLimiter()
        self._retry = retry or RetryPolicy()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._timeout = timeout
        self._sleeper = sleeper

    # --- 下位クラスが実装する ---------------------------------------------

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _payload(self, prompt: str) -> dict:
        raise NotImplementedError

    def _extract(self, data: Mapping[str, Any]) -> str:
        raise NotImplementedError

    # --- 共通 --------------------------------------------------------------

    def answer(self, prompt: str) -> str:
        self._budget.consume()
        self._limiter.wait()
        return self._extract(self._post_with_retry(self._payload(prompt)))

    def _post_with_retry(self, payload: Mapping[str, Any]) -> dict:
        backoff = self._retry.initial_backoff
        last_error: Exception | None = None

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return self._transport.post_json(
                    self._endpoint(), payload, self._headers(), self._timeout
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in self._retry.retry_status:
                    raise ApiError(
                        f"{self.name}: HTTP {exc.code} {_safe_body(exc)}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc

            if attempt < self._retry.max_attempts:
                self._sleeper(backoff)
                backoff = min(backoff * 2.0, self._retry.max_backoff)

        raise ApiError(
            f"{self.name}: {self._retry.max_attempts} 回試して失敗した: {last_error}"
        ) from last_error


def _safe_body(exc: urllib.error.HTTPError) -> str:
    """エラー本文を読む。読めなくても落ちない。**API キーは本文に出ない前提。**"""
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001 - 診断用なので握り潰してよい
        return "(本文を読めなかった)"


class AnthropicModel(_BaseApiModel):
    """Anthropic Messages API。

    拡張思考は使わない(既定)。使うと測っているものが変わるので、必要なら
    `preregister.md` に書いてから有効にすること。
    """

    def _endpoint(self) -> str:
        return ANTHROPIC_ENDPOINT

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

    def _extract(self, data: Mapping[str, Any]) -> str:
        blocks = data.get("content", [])
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "".join(texts).strip()


class OpenAICompatibleModel(_BaseApiModel):
    """OpenAI の Chat Completions 形式。

    OpenAI 本体のほか、ローカルの Ollama / vLLM / LM Studio、および同形式を出す
    国内モデルのホストにも `base_url` の差し替えだけで当たる。
    """

    def __init__(self, *args, base_url: str = OPENAI_BASE_URL, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = base_url.rstrip("/")

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        # ローカルモデルはキー不要なことが多いので、空なら付けない。
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

    def _extract(self, data: Mapping[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        return (choices[0].get("message", {}).get("content") or "").strip()


# --------------------------------------------------------------------------
# CLI のモデル指定を組み立てる
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientOptions:
    budget: CallBudget
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    rate_limit_per_minute: int | None = None
    timeout: float = DEFAULT_TIMEOUT
    transport: HttpTransport | None = None
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)


def build_api_model(spec: str, options: ClientOptions):
    """モデル指定から実クライアントを組み立てる。`fake:` はここでは扱わない。

        anthropic:NAME:MODEL_ID
        openai:NAME:MODEL_ID
        compat:NAME:MODEL_ID:BASE_URL      ← ローカル/自前ホスト

    API キーは環境変数から読む。**指定文字列にキーを書かない**(履歴とログに残るため)。
        anthropic → ANTHROPIC_API_KEY
        openai    → OPENAI_API_KEY
        compat    → CONTAMLAB_API_KEY(無ければキー無しで送る)
    """
    kind, _, rest = spec.partition(":")

    common = dict(
        budget=options.budget,
        transport=options.transport,
        rate_limiter=RateLimiter(options.rate_limit_per_minute),
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        timeout=options.timeout,
    )

    if kind == "anthropic":
        name, model_id = _split_exactly(rest, 2, spec, "anthropic:NAME:MODEL_ID")
        return AnthropicModel(
            name, model_id, _require_key(options.env, "ANTHROPIC_API_KEY", name), **common
        )

    if kind == "openai":
        name, model_id = _split_exactly(rest, 2, spec, "openai:NAME:MODEL_ID")
        return OpenAICompatibleModel(
            name,
            model_id,
            _require_key(options.env, "OPENAI_API_KEY", name),
            base_url=OPENAI_BASE_URL,
            **common,
        )

    if kind == "compat":
        # BASE_URL に ":" が含まれるので、最初の2つだけ切って残りを URL とみなす。
        parts = rest.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"モデル指定が読めない: {spec!r}(形式は compat:NAME:MODEL_ID:BASE_URL)"
            )
        name, model_id, base_url = parts
        return OpenAICompatibleModel(
            name,
            model_id,
            options.env.get("CONTAMLAB_API_KEY", ""),
            base_url=base_url,
            **common,
        )

    raise ValueError(
        f"未知のモデル種別: {kind!r}(利用可能: fake, anthropic, openai, compat)"
    )


def is_api_spec(spec: str) -> bool:
    """課金が発生する指定かどうか。CLI の確認プロンプトの判断に使う。"""
    return spec.split(":", 1)[0] in {"anthropic", "openai", "compat"}


def _split_exactly(rest: str, count: int, spec: str, form: str) -> list[str]:
    parts = rest.split(":")
    if len(parts) != count or not all(parts):
        raise ValueError(f"モデル指定が読めない: {spec!r}(形式は {form})")
    return parts


def _require_key(env: Mapping[str, str], variable: str, name: str) -> str:
    key = env.get(variable, "")
    if not key:
        raise ValueError(
            f"{name}: 環境変数 {variable} が設定されていない。"
            ".env に書くか、シェルで設定すること(指定文字列にキーを書かない)。"
        )
    return key


def load_dotenv(path) -> None:
    """.env を環境変数に流し込む(既存の環境変数は上書きしない)。"""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
