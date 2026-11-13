"""The G2 judge and `nyaya-eval judge-validate` (design file 45).

*   `Judge.judge(sentence, passage)` returns `supported`, `partial` or
    `unsupported` for one (sentence, passage) pair, from the `JUDGE_*` model
    with the `judge.prompt` template (prompts/judge_v1.md).
*   The reply must be a JSON object `{"label": ..., "reason": ...}` with no
    other keys and a known label (`parse_verdict`). A bad reply is retried
    once, with the parse error sent back; a second bad reply gives `error`.
    A network or server error also gives `error`; running out of free-tier
    quota (`RateLimitExhausted`) is raised, so the run can stop and resume.
*   `cohen_kappa` and `validate` compare the judge with the author's hand
    labels (`eval/datasets/hand_labels_v1.jsonl`). Pairs the judge could not
    label are left out of kappa and counted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nyaya.answer.llm_client import JSONCompleter, LLMBadJSON, LLMClient, LLMUnavailable
from nyaya.answer.prompt import PROMPTS_DIR, Template, load_template
from nyaya.config import Settings
from nyaya.eval.datasets import HandLabel, read_jsonl
from nyaya.eval.ratelimit import RateLimitExhausted

LABELS = ("supported", "partial", "unsupported")
ERROR = "error"
# best first: a sentence takes the best verdict over its citations
RANK = {"supported": 0, "partial": 1, "unsupported": 2, ERROR: 3}

REPAIR_MESSAGE = (
    "Your reply was not valid: {error}. Reply again with only the JSON object "
    '{{"label": "supported" | "partial" | "unsupported", "reason": "..."}}.'
)


class JudgeNotConfigured(RuntimeError):
    pass


class BadVerdict(ValueError):
    pass


def parse_verdict(obj: Any) -> tuple[str, str]:
    """Strict parse of a judge reply object: `(label, reason)`."""
    if not isinstance(obj, dict):
        raise BadVerdict(f"expected a JSON object, got {type(obj).__name__}")
    extra = set(obj) - {"label", "reason"}
    if extra:
        raise BadVerdict(f"unexpected keys {sorted(extra)}")
    label = obj.get("label")
    if label not in LABELS:
        raise BadVerdict(f"label must be one of {list(LABELS)}, got {label!r}")
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        raise BadVerdict("reason must be a string")
    return label, reason


@dataclass(frozen=True)
class Verdict:
    label: str  # supported | partial | unsupported | error
    reason: str = ""
    attempts: int = 1
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def best(labels: Iterable[str]) -> str:
    """The best label of several (supported > partial > unsupported > error)."""
    return min(labels, key=lambda lab: RANK[lab], default=ERROR)


class Judge:
    def __init__(self, llm: JSONCompleter, template: Template, timeout_s: float) -> None:
        self.llm = llm
        self.template = template
        self.timeout_s = timeout_s
        self.model = llm.model

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        wrap: Callable[[JSONCompleter], JSONCompleter] | None = None,
        prompts_dir: Path = PROMPTS_DIR,
    ) -> Judge:
        """The judge from `JUDGE_BASE_URL`, `JUDGE_API_KEY` and `JUDGE_MODEL`."""
        s = settings.secrets
        missing = [
            name
            for name, value in (
                ("JUDGE_BASE_URL", s.judge_base_url),
                ("JUDGE_API_KEY", s.judge_api_key),
                ("JUDGE_MODEL", s.judge_model),
            )
            if not value
        ]
        if missing:
            raise JudgeNotConfigured(f"judge not configured: {', '.join(missing)} missing")
        assert s.judge_api_key is not None and s.judge_model is not None
        j = settings.judge
        llm: JSONCompleter = LLMClient(
            model=s.judge_model,
            max_output_tokens=j.max_output_tokens,
            reasoning_effort=j.reasoning_effort,
            base_url=s.judge_base_url,
            api_key=s.judge_api_key.get_secret_value(),
        )
        if wrap is not None:
            llm = wrap(llm)
        return cls(llm, load_template(j.prompt, prompts_dir), j.timeout_s)

    def prompt(self, sentence: str, passage: str) -> tuple[str, str]:
        user = self.template.user.replace("{{sentence}}", sentence)
        user = user.replace("{{passage}}", passage)
        return self.template.system, user

    def judge(self, sentence: str, passage: str) -> Verdict:
        system, user = self.prompt(sentence, passage)
        history: list[dict[str, str]] = []
        for attempt in (1, 2):
            raw = ""
            try:
                obj = self.llm.complete_json(system, user, timeout=self.timeout_s, history=history)
                label, reason = parse_verdict(obj)
                return Verdict(label, reason, attempts=attempt)
            except RateLimitExhausted:
                raise
            except LLMUnavailable as exc:
                return Verdict(ERROR, attempts=attempt, error=f"{type(exc).__name__}: {exc}")
            except LLMBadJSON as exc:
                error, raw = exc.error, exc.raw or ""
            except BadVerdict as exc:
                error, raw = str(exc), json.dumps(obj)
            if attempt == 2:
                return Verdict(ERROR, attempts=2, error=f"bad reply: {error}")
            history = [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_MESSAGE.format(error=error)},
            ]
        raise AssertionError("unreachable")  # pragma: no cover


# --- judge-validate -----------------------------------------------------------------


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Unweighted Cohen's kappa of two raters over the same items.

    When chance agreement is 1 (both raters used one and the same label for
    every item) kappa is undefined; this returns 1.0 for full agreement.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lists differ in length: {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("no items to compare")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def confusion(hand: Sequence[str], judge: Sequence[str]) -> dict[str, dict[str, int]]:
    """Rows: the hand label; columns: the judge's label."""
    out = {h: dict.fromkeys(LABELS, 0) for h in LABELS}
    for h, j in zip(hand, judge, strict=True):
        out[h][j] += 1
    return out


class HandLabelsMissing(FileNotFoundError):
    pass


def read_hand_labels(path: Path) -> list[HandLabel]:
    if not path.exists():
        raise HandLabelsMissing(
            f"{path} not found: judge-validate needs the author's hand labels "
            '(one {"id", "sentence", "chunk_id", "label", "labeller"} per line; '
            "see eval/datasets/README.md)"
        )
    labels = read_jsonl(path, HandLabel)
    if not labels:
        raise HandLabelsMissing(
            f"{path} is empty: label 30 to 50 (sentence, passage) pairs first "
            "(see eval/datasets/README.md, 'Hand-label format')"
        )
    return labels


def validate(
    labels: Sequence[HandLabel],
    passages: Mapping[str, str],
    judge: Judge,
    *,
    min_kappa: float,
    min_labels: int,
) -> dict[str, Any]:
    """Judge every hand-labelled pair and compare. `passages`: chunk_id -> passage text."""
    rows: list[dict[str, Any]] = []
    missing = [lab.id for lab in labels if lab.chunk_id not in passages]
    for lab in labels:
        if lab.chunk_id not in passages:
            continue
        v = judge.judge(lab.sentence, passages[lab.chunk_id])
        rows.append({"id": lab.id, "hand": lab.label, "judge": v.label, **_verdict_extra(v)})
    scored = [r for r in rows if r["judge"] != ERROR]
    hand = [r["hand"] for r in scored]
    got = [r["judge"] for r in scored]
    kappa = cohen_kappa(hand, got) if scored else None
    enough = len(scored) >= min_labels
    return {
        "labels": len(labels),
        "scored": len(scored),
        "judge_errors": len(rows) - len(scored),
        "missing_chunks": missing,
        "kappa": kappa,
        "agreement": (sum(h == j for h, j in zip(hand, got, strict=True)) / len(scored))
        if scored
        else None,
        "confusion": confusion(hand, got),
        "min_kappa": min_kappa,
        "min_labels": min_labels,
        "enough_labels": enough,
        "passed": bool(enough and kappa is not None and kappa >= min_kappa),
        "judge_model": judge.model,
        "rows": rows,
    }


def _verdict_extra(v: Verdict) -> dict[str, Any]:
    out: dict[str, Any] = {"reason": v.reason}
    if v.error:
        out["error"] = v.error
    return out
