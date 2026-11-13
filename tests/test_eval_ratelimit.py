"""Pacing, 429 backoff and token counting for the Phase 3 LLM calls (no network)."""

from __future__ import annotations

import httpx
import openai
import pytest

from nyaya.answer.llm_client import LLMBadJSON, LLMUnavailable
from nyaya.eval.ratelimit import (
    PausableClock,
    RateLimitedCompleter,
    RateLimitExhausted,
    Throttle,
    is_rate_limited,
    retry_after,
)

REQ = httpx.Request("POST", "http://llm/v1/chat/completions")


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make_clock():
    t = FakeTime()
    return t, PausableClock(t.clock, t.sleep)


def rate_limit_error(headers=None):
    """What LLMClient raises on HTTP 429: LLMUnavailable from openai.RateLimitError."""
    cause = openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=REQ, headers=headers or {}), body=None
    )
    try:
        raise LLMUnavailable(f"RateLimitError: {cause}") from cause
    except LLMUnavailable as exc:
        return exc


class ScriptedLLM:
    """Raises or returns the scripted items in order; records token usage per reply."""

    model = "fake"

    def __init__(self, script, usage=None):
        self.script = list(script)
        self.calls = 0
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.last_usage = None

    def complete_json(self, system, user, timeout, history=()):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            self.last_usage = self.usage if isinstance(item, LLMBadJSON) else None
            raise item
        self.last_usage = self.usage
        return item


def wrapper(inner, clock, rpm=0, retries=3):
    return RateLimitedCompleter(
        inner, clock=clock, rpm=rpm, retries=retries, backoff_base_s=5, backoff_max_s=60
    )


def test_pausable_clock_excludes_sleeps():
    t, clock = make_clock()
    start = clock()
    clock.sleep(30)
    t.now += 2
    assert clock() - start == pytest.approx(2)
    assert clock.wall() - start == pytest.approx(32)
    clock.sleep(0)
    assert t.sleeps == [30]


def test_throttle_spaces_calls():
    t, clock = make_clock()
    th = Throttle(rpm=20, clock=clock)  # every 3 s
    th.wait()
    t.now += 1
    th.wait()
    th.wait()
    assert t.sleeps == [pytest.approx(2), pytest.approx(3)]
    assert Throttle(rpm=0, clock=clock).interval == 0


def test_is_rate_limited_and_retry_after():
    exc = rate_limit_error({"retry-after": "7"})
    assert is_rate_limited(exc)
    assert retry_after(exc) == 7
    assert retry_after(rate_limit_error({"retry-after": "soon"})) is None
    assert retry_after(rate_limit_error()) is None
    assert not is_rate_limited(LLMUnavailable("ConnectError: refused"))


def test_429_backs_off_then_succeeds():
    t, clock = make_clock()
    inner = ScriptedLLM([rate_limit_error(), rate_limit_error(), {"ok": 1}])
    w = wrapper(inner, clock)
    assert w.complete_json("s", "u", timeout=5) == {"ok": 1}
    assert t.sleeps == [5, 10]  # exponential
    assert (w.calls, w.rate_limited, w.exhausted) == (3, 2, False)
    assert w.tokens == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_retry_after_header_lengthens_the_wait_up_to_the_cap():
    t, clock = make_clock()
    inner = ScriptedLLM([rate_limit_error({"retry-after": "30"}), rate_limit_error(), {"ok": 1}])
    w = wrapper(inner, clock)
    w.backoff_max_s = 20
    w.complete_json("s", "u", timeout=5)
    assert t.sleeps == [20, 10]


def test_429_exhausted_raises_and_flags():
    _, clock = make_clock()
    inner = ScriptedLLM([rate_limit_error()] * 3)
    w = wrapper(inner, clock, retries=2)
    with pytest.raises(RateLimitExhausted):
        w.complete_json("s", "u", timeout=5)
    assert w.exhausted and inner.calls == 3
    # RateLimitExhausted is an LLMUnavailable, so the pipeline answers `passages`
    assert issubclass(RateLimitExhausted, LLMUnavailable)


def test_other_errors_pass_through_and_bad_json_tokens_count():
    _, clock = make_clock()
    w = wrapper(ScriptedLLM([LLMUnavailable("ConnectError")]), clock)
    with pytest.raises(LLMUnavailable) as info:
        w.complete_json("s", "u", timeout=5)
    assert not isinstance(info.value, RateLimitExhausted)
    w = wrapper(ScriptedLLM([LLMBadJSON("bad", "x")]), clock)
    with pytest.raises(LLMBadJSON):
        w.complete_json("s", "u", timeout=5)
    assert w.tokens["total_tokens"] == 15


def test_pacing_applies_to_every_call():
    t, clock = make_clock()
    w = wrapper(ScriptedLLM([{"a": 1}, {"a": 2}]), clock, rpm=30)
    w.complete_json("s", "u", timeout=5)
    w.complete_json("s", "u", timeout=5)
    assert t.sleeps == [pytest.approx(2)]
