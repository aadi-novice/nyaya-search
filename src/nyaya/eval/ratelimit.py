"""Pacing and 429 backoff for the free LLM tiers used by the Phase 3 runs.

Groq's free tier limits requests and tokens per minute (and per day). The
evaluation runs therefore:

*   space calls at least `60 / rpm` seconds apart (`Throttle`);
*   on HTTP 429, wait and retry with exponential backoff, honouring a
    `Retry-After` header when the provider sends one (`RateLimitedCompleter`);
*   after `retries` failed retries, raise `RateLimitExhausted`, so a run stops
    cleanly and can be resumed later (for example the next day).

Time spent waiting here is excluded from the pipeline's request clock
(`PausableClock`), so pacing never eats into `timeouts.request_s` or the
answer deadline. `nyaya-eval latency` paces between requests instead, outside
the timed region.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import openai

from nyaya.answer.llm_client import JSONCompleter, LLMBadJSON, LLMUnavailable


class RateLimitExhausted(LLMUnavailable):
    """Still rate-limited after every retry: stop the run and resume later."""


class PausableClock:
    """`time.monotonic()`, minus the time spent in `sleep()`.

    Used as the pipeline's `Deps.clock` during evaluation runs, so throttle
    and backoff waits do not count against the per-request time limits.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleeper
        self.paused = 0.0

    def __call__(self) -> float:
        return self._clock() - self.paused

    def wall(self) -> float:
        return self._clock()

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._sleep(seconds)
        self.paused += seconds


class Throttle:
    """At most `rpm` calls per minute: each `wait()` starts at least 60/rpm s after the last."""

    def __init__(self, rpm: float, clock: PausableClock) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock.wall()
        if self._last is not None and self.interval > 0:
            self._clock.sleep(self._last + self.interval - now)
        self._last = self._clock.wall()


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in out:
        out.append(cur)
        cur = cur.__cause__ or cur.__context__
    return out


def is_rate_limited(exc: BaseException) -> bool:
    """True for HTTP 429 anywhere in the exception's cause chain."""
    for e in _chain(exc):
        if isinstance(e, openai.RateLimitError) or getattr(e, "status_code", None) == 429:
            return True
    return False


def retry_after(exc: BaseException) -> float | None:
    """Seconds from a `Retry-After` header in the cause chain, when numeric."""
    for e in _chain(exc):
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        value = headers.get("retry-after")
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    return None


class RateLimitedCompleter:
    """A `JSONCompleter` that paces calls, backs off on 429, and counts tokens.

    `tokens` sums the `last_usage` of the wrapped client (LLMClient) over
    every reply, including replies that failed to parse; `calls` counts
    requests sent, including those rejected with 429.
    """

    def __init__(
        self,
        inner: JSONCompleter,
        *,
        clock: PausableClock,
        rpm: float,
        retries: int,
        backoff_base_s: float,
        backoff_max_s: float,
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.clock = clock
        self.throttle = Throttle(rpm, clock)
        self.retries = retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.calls = 0
        self.rate_limited = 0
        self.exhausted = False
        self.tokens: dict[str, int] = {}

    def _count(self) -> None:
        usage = getattr(self.inner, "last_usage", None)
        if isinstance(usage, dict):
            for k, v in usage.items():
                self.tokens[k] = self.tokens.get(k, 0) + int(v)

    def backoff(self, attempt: int, exc: BaseException) -> float:
        hinted = retry_after(exc)
        wait = self.backoff_base_s * (2**attempt)
        if hinted is not None:
            wait = max(wait, hinted)
        return min(wait, self.backoff_max_s)

    def complete_json(
        self,
        system: str,
        user: str,
        timeout: float,
        history: Sequence[dict[str, str]] = (),
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            self.throttle.wait()
            self.calls += 1
            try:
                out = self.inner.complete_json(system, user, timeout=timeout, history=history)
            except LLMBadJSON:
                self._count()
                raise
            except LLMUnavailable as exc:
                if not is_rate_limited(exc):
                    raise
                self.rate_limited += 1
                if attempt >= self.retries:
                    self.exhausted = True
                    raise RateLimitExhausted(
                        f"still rate-limited after {self.retries} retries: {exc}"
                    ) from exc
                self.clock.sleep(self.backoff(attempt, exc))
                attempt += 1
                continue
            self._count()
            return out
