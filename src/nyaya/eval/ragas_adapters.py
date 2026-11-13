"""Our clients behind the RAGAS 0.4 interfaces (imported only by `nyaya-eval ragas`).

*   `CompleterLLM` puts a `JSONCompleter` (the judge's `LLMClient`, wrapped in
    our `RateLimitedCompleter`) behind ragas's `InstructorBaseRagasLLM`. Each
    ragas prompt already carries its output's JSON schema; the reply is
    validated against the ragas response model. A bad reply is retried once
    with the error sent back (as in `judge.py`); a second bad reply raises
    `BadReply`. `RateLimitExhausted` is raised unchanged, so the run can stop
    and resume.
*   `cpu_embeddings` is ragas's sentence-transformers wrapper, on the CPU.

Importing this module imports ragas (the `ragas` extra), with its usage
analytics switched off.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from pydantic import BaseModel, ValidationError
from ragas.llms.base import InstructorBaseRagasLLM

from nyaya.answer.llm_client import JSONCompleter, LLMBadJSON

M = TypeVar("M", bound=BaseModel)

SYSTEM = (
    "You are an evaluator. Follow the instructions in the user message exactly "
    "and reply with only the JSON object it asks for."
)
REPAIR_MESSAGE = (
    "Your reply was not valid: {error}. Reply again with only a JSON object "
    "that matches the schema in the instructions."
)


class BadReply(ValueError):
    """The model's reply did not match the ragas response model, twice."""


class CompleterLLM(InstructorBaseRagasLLM):
    def __init__(self, llm: JSONCompleter, timeout_s: float) -> None:
        self.llm = llm
        self.model = llm.model
        self.timeout_s = timeout_s

    def generate(self, prompt: str, response_model: type[M]) -> M:
        history: list[dict[str, str]] = []
        for attempt in (1, 2):
            obj: Any = None
            try:
                obj = self.llm.complete_json(
                    SYSTEM, prompt, timeout=self.timeout_s, history=history
                )
                return response_model.model_validate(obj)
            except LLMBadJSON as exc:
                error, raw = exc.error, exc.raw or ""
            except ValidationError as exc:
                error, raw = _short(exc), json.dumps(obj, ensure_ascii=False)
            if attempt == 2:
                raise BadReply(f"bad reply after one retry: {error}")
            history = [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_MESSAGE.format(error=error)},
            ]
        raise AssertionError("unreachable")  # pragma: no cover

    async def agenerate(self, prompt: str, response_model: type[M]) -> M:
        # One call at a time: the rate limiter paces with blocking sleeps.
        return self.generate(prompt, response_model)


def _short(exc: ValidationError) -> str:
    errs = exc.errors()
    parts = [f"{'.'.join(str(x) for x in e['loc']) or '(root)'}: {e['msg']}" for e in errs[:3]]
    more = f" (and {len(errs) - 3} more)" if len(errs) > 3 else ""
    return "; ".join(parts) + more


def cpu_embeddings(model: str, device: str = "cpu") -> Any:
    """ragas's sentence-transformers embeddings (loads, or downloads once, `model`)."""
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=model, device=device)
