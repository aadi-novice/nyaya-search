from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from nyaya.answer.llm_client import (
    LLMBadJSON,
    LLMClient,
    LLMUnavailable,
    generate_answer,
    parse_json_object,
)
from nyaya.config import load_settings

SETTINGS = load_settings(env_file=None)
GOOD = {"sentences": [{"sentence": "s", "citations": [{"label": "P1", "quote": "q"}]}], "gaps": []}
REQ = httpx.Request("POST", "http://llm.test/v1/chat/completions")


def reply(content, finish_reason="stop"):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])


class FakeOpenAI:
    """Stands in for openai.OpenAI: returns or raises the scripted results in order."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def client(*results):
    fake = FakeOpenAI(*results)
    return LLMClient(model="m", max_output_tokens=800, client=fake), fake


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


# --- complete_json -------------------------------------------------------------


def test_complete_json_sends_budget_temperature_and_json_mode():
    llm, fake = client(reply(json.dumps(GOOD)))
    assert llm.complete_json("SYS", "USER", timeout=7) == GOOD
    call = fake.calls[0]
    assert call["model"] == "m"
    assert call["temperature"] == 0
    assert call["max_tokens"] == 800
    assert call["timeout"] == 7
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


def test_json_mode_can_be_switched_off():
    fake = FakeOpenAI(reply("{}"))
    LLMClient(model="m", max_output_tokens=10, client=fake, json_mode=False).complete_json(
        "s", "u", timeout=1
    )
    assert "response_format" not in fake.calls[0]


@pytest.mark.parametrize(
    "exc",
    [
        openai.APITimeoutError(request=REQ),
        openai.APIConnectionError(request=REQ),
        openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None),
        openai.InternalServerError("boom", response=httpx.Response(503, request=REQ), body=None),
        openai.BadRequestError("bad", response=httpx.Response(400, request=REQ), body=None),
        httpx.ReadTimeout("read timeout"),
    ],
)
def test_transport_and_server_errors_are_unavailable(exc):
    llm, _ = client(exc)
    with pytest.raises(LLMUnavailable):
        llm.complete_json("s", "u", timeout=1)


@pytest.mark.parametrize(
    "result",
    [
        reply("not json"),
        reply(None),
        reply("[1, 2]"),
        reply('{"sentences": ['),
        SimpleNamespace(choices=[]),
    ],
)
def test_bad_replies_raise_bad_json_with_raw_text(result):
    llm, _ = client(result)
    with pytest.raises(LLMBadJSON) as info:
        llm.complete_json("s", "u", timeout=1)
    assert info.value.error


def test_truncated_reply_says_so():
    llm, _ = client(reply('{"sentences": [', finish_reason="length"))
    with pytest.raises(LLMBadJSON, match="max_output_tokens"):
        llm.complete_json("s", "u", timeout=1)


def test_parse_strips_code_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_history_is_appended_after_user():
    llm, fake = client(reply("{}"))
    extra = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "fix"}]
    llm.complete_json("s", "u", timeout=1, history=extra)
    assert fake.calls[0]["messages"][2:] == extra


def test_from_settings_requires_llm_secrets(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMUnavailable, match="LLM_MODEL"):
        LLMClient.from_settings(load_settings(env_file=None))


def test_from_settings_builds_client(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    llm = LLMClient.from_settings(load_settings(env_file=None))
    assert llm.model == "gemini-test"
    assert llm.max_output_tokens == SETTINGS.llm.max_output_tokens


# --- generate_answer (timeouts and the repair retry) ----------------------------


def test_generate_answer_first_try():
    llm, fake = client(reply(json.dumps(GOOD)))
    clock = Clock()
    result = generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)
    assert result.output == GOOD
    assert result.attempts == 1 and not result.repaired
    assert fake.calls[0]["timeout"] == SETTINGS.timeouts.llm_s


def test_first_call_timeout_capped_by_request_limit():
    llm, fake = client(reply(json.dumps(GOOD)))
    clock = Clock(100.0)
    started = 100.0 - (SETTINGS.timeouts.request_s - 5)  # 5 s of the hard limit left
    generate_answer(llm, "s", "u", started_at=started, settings=SETTINGS, clock=clock)
    assert fake.calls[0]["timeout"] == pytest.approx(5)


def test_no_time_left_is_unavailable():
    llm, fake = client()
    clock = Clock(100.0)
    started = 100.0 - SETTINGS.timeouts.request_s
    with pytest.raises(LLMUnavailable, match="no time"):
        generate_answer(llm, "s", "u", started_at=started, settings=SETTINGS, clock=clock)
    assert fake.calls == []


def test_repair_retry_sends_parse_error_back():
    llm, fake = client(reply("oops"), reply(json.dumps(GOOD)))
    clock = Clock()
    result = generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)
    assert result.output == GOOD and result.repaired and result.attempts == 2
    repair_msgs = fake.calls[1]["messages"]
    assert repair_msgs[2] == {"role": "assistant", "content": "oops"}
    assert repair_msgs[3]["role"] == "user" and "not valid" in repair_msgs[3]["content"]
    # the repair timeout is what is left before the answer deadline, at most llm_s
    assert fake.calls[1]["timeout"] <= SETTINGS.timeouts.llm_s


def test_wrong_shape_triggers_repair():
    llm, fake = client(reply('{"answer": "x"}'), reply(json.dumps(GOOD)))
    clock = Clock()
    result = generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)
    assert result.repaired
    assert "sentences" in fake.calls[1]["messages"][3]["content"]


def test_missing_gaps_and_bad_gaps_shape():
    llm, _ = client(reply('{"sentences": []}'))
    clock = Clock()
    out = generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)
    assert out.output == {"sentences": [], "gaps": []}
    llm, _ = client(reply('{"sentences": [], "gaps": "none"}'), reply('{"sentences": []}'))
    out = generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)
    assert out.repaired


def test_no_repair_when_too_little_time_remains():
    llm, fake = client(reply("oops"))
    clock = Clock(100.0)
    left = SETTINGS.llm.repair_min_seconds_left - 0.5
    started = 100.0 - (SETTINGS.timeouts.answer_deadline_s - left)
    with pytest.raises(LLMBadJSON):
        generate_answer(llm, "s", "u", started_at=started, settings=SETTINGS, clock=clock)
    assert len(fake.calls) == 1


def test_only_one_repair():
    llm, fake = client(reply("oops"), reply("still bad"))
    clock = Clock()
    with pytest.raises(LLMBadJSON):
        generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)
    assert len(fake.calls) == 2


def test_unavailable_during_repair_propagates():
    llm, _ = client(reply("oops"), openai.APITimeoutError(request=REQ))
    clock = Clock()
    with pytest.raises(LLMUnavailable):
        generate_answer(llm, "s", "u", started_at=clock(), settings=SETTINGS, clock=clock)


def test_reasoning_effort_is_sent_only_when_set():
    llm, fake = client(reply(json.dumps(GOOD)))
    llm.complete_json("s", "u", timeout=1)
    assert "reasoning_effort" not in fake.calls[0]
    fake2 = FakeOpenAI(reply(json.dumps(GOOD)))
    LLMClient(model="m", max_output_tokens=10, client=fake2, reasoning_effort="low").complete_json(
        "s", "u", timeout=1
    )
    assert fake2.calls[0]["reasoning_effort"] == "low"


def test_last_usage_is_kept_for_the_eval_harness():
    with_usage = reply(json.dumps(GOOD))
    with_usage.usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150)
    bad = reply("not json")
    bad.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105)
    llm, _ = client(with_usage, reply(json.dumps(GOOD)), bad)
    llm.complete_json("s", "u", timeout=5)
    assert llm.last_usage == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    llm.complete_json("s", "u", timeout=5)
    assert llm.last_usage is None  # the provider sent no usage
    with pytest.raises(LLMBadJSON):
        llm.complete_json("s", "u", timeout=5)
    assert llm.last_usage["total_tokens"] == 105  # a bad reply still used tokens
