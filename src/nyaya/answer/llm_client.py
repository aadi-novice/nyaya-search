"""The answer LLM: one OpenAI-compatible client, temperature 0, JSON object output.

Design, "Answer generation" and file 33.

* A network error, rate limit, timeout or server error (any API error, in
  fact) raises `LLMUnavailable`; the pipeline then returns `passages`.
* A reply that is not a JSON object raises `LLMBadJSON`. `generate_answer`
  allows one repair retry, sending the parse error back, but only if at least
  `llm.repair_min_seconds_left` seconds remain before
  `timeouts.answer_deadline_s`, measured from when the request arrived.
* No call is given more than `timeouts.llm_s` seconds, nor more than what is
  left of the `timeouts.request_s` hard limit.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import openai

from nyaya.config import Settings


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """Network error, rate limit, timeout, server error, or no time left."""


class LLMBadJSON(LLMError):
    """The reply was not a JSON object of the expected shape."""

    def __init__(self, error: str, raw: str | None = None) -> None:
        super().__init__(error)
        self.error = error
        self.raw = raw


class JSONCompleter(Protocol):
    model: str

    def complete_json(
        self,
        system: str,
        user: str,
        timeout: float,
        history: Sequence[dict[str, str]] = (),
    ) -> dict[str, Any]: ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object; a surrounding ``` fence is tolerated."""
    text = raw.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMBadJSON(f"invalid JSON: {exc}", raw) from None
    if not isinstance(value, dict):
        raise LLMBadJSON(f"expected a JSON object, got {type(value).__name__}", raw)
    return value


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        max_output_tokens: int,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        json_mode: bool = True,
        reasoning_effort: str = "",
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        # token counts of the last reply (None if the provider sent none); the
        # evaluation harness reads them to record tokens used per run
        self.last_usage: dict[str, int] | None = None
        # max_retries=0: the time budget is ours to manage, not the SDK's.
        self._client = client or openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> LLMClient:
        s = settings.secrets
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", s.llm_base_url),
                ("LLM_API_KEY", s.llm_api_key),
                ("LLM_MODEL", s.llm_model),
            )
            if not value
        ]
        if missing:
            raise LLMUnavailable(f"answer LLM not configured: {', '.join(missing)} missing")
        assert s.llm_api_key is not None and s.llm_model is not None
        return cls(
            model=s.llm_model,
            max_output_tokens=settings.llm.max_output_tokens,
            reasoning_effort=settings.llm.reasoning_effort,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value(),
            **kwargs,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        timeout: float,
        history: Sequence[dict[str, str]] = (),
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            *history,
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "timeout": timeout,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        self.last_usage = None
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        self.last_usage = usage_dict(getattr(response, "usage", None))
        if not response.choices:
            raise LLMBadJSON("empty reply (no choices)")
        choice = response.choices[0]
        raw = choice.message.content
        if not raw:
            raise LLMBadJSON("empty reply", raw)
        try:
            return parse_json_object(raw)
        except LLMBadJSON as exc:
            if choice.finish_reason == "length":
                raise LLMBadJSON(
                    f"reply cut off at max_output_tokens ({self.max_output_tokens}): {exc.error}",
                    raw,
                ) from None
            raise


def usage_dict(usage: Any) -> dict[str, int] | None:
    """`prompt_tokens`, `completion_tokens` and `total_tokens` from a reply's usage."""
    if usage is None:
        return None
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out or None


def validate_answer_shape(obj: dict[str, Any]) -> dict[str, Any]:
    """`{"sentences": [...], "gaps": [...]}`; items are checked by the citation check."""
    sentences = obj.get("sentences")
    if not isinstance(sentences, list):
        raise LLMBadJSON('the object needs a "sentences" list', json.dumps(obj))
    gaps = obj.get("gaps", [])
    if not isinstance(gaps, list):
        raise LLMBadJSON('"gaps" must be a list of strings', json.dumps(obj))
    return {"sentences": sentences, "gaps": gaps}


@dataclass(frozen=True)
class LLMResult:
    output: dict[str, Any]
    attempts: int
    repaired: bool


REPAIR_MESSAGE = (
    "Your reply was not valid: {error}. Reply again with only the JSON object "
    '{{"sentences": [{{"sentence": "...", "citations": [{{"label": "P1", "quote": "..."}}]}}], '
    '"gaps": ["..."]}}, following the same rules.'
)


def generate_answer(
    llm: JSONCompleter,
    system: str,
    user: str,
    *,
    started_at: float,
    settings: Settings,
    clock: Callable[[], float] = time.monotonic,
) -> LLMResult:
    """Call the LLM, with one repair retry on a bad reply if time allows."""
    t = settings.timeouts
    hard_left = started_at + t.request_s - clock()
    if hard_left <= 0:
        raise LLMUnavailable("no time left before the request limit")
    try:
        obj = llm.complete_json(system, user, timeout=min(t.llm_s, hard_left))
        return LLMResult(validate_answer_shape(obj), attempts=1, repaired=False)
    except LLMBadJSON as exc:
        first = exc
    left = started_at + t.answer_deadline_s - clock()
    if left < settings.llm.repair_min_seconds_left:
        raise first
    history = [
        {"role": "assistant", "content": first.raw or ""},
        {"role": "user", "content": REPAIR_MESSAGE.format(error=first.error)},
    ]
    obj = llm.complete_json(system, user, timeout=min(t.llm_s, left), history=history)
    return LLMResult(validate_answer_shape(obj), attempts=2, repaired=True)
