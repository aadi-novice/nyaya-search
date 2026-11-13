"""`nyaya-eval latency`: G7, end-to-end time of a checked answer (design file 45).

*   A fixed query set: the first `latency.query_count` questions of the
    dataset, sorted by id. `latency.warmup` requests run first and are not
    counted, so the models are warm.
*   One request at a time. Each is timed on the wall clock around
    `pipeline.ask`; the pipeline's own `timings_ms` give the stage split.
*   Requests are paced to `answers.llm_rpm` *between* requests, outside the
    timed region. A request whose LLM call was rate-limited (HTTP 429) is
    not a latency sample: it is retried after a backoff, up to
    `answers.rate_limit_retries` times, then the run stops.
*   A request that takes longer than `timeouts.request_s`, or ends in
    `error`, is a failure, and counts as an infinitely slow sample, so more
    than 5% failures fail the p95 target.
*   Percentiles use the nearest-rank method (an observed value, never an
    interpolation).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from nyaya.contracts import AskRequest, AskResponse, Status
from nyaya.eval.ratelimit import PausableClock, Throttle

# pipeline stage -> reported group (anything else, and the remainder, is "other")
STAGE_GROUPS = {
    "route": "retrieve",
    "lookup": "retrieve",
    "embed": "retrieve",
    "keyword": "retrieve",
    "vector": "retrieve",
    "fusion": "retrieve",
    "expand": "retrieve",
    "fetch": "retrieve",
    "fetch_judgments": "retrieve",
    "rerank": "rerank",
    "prompt": "llm",
    "llm": "llm",
    "check": "check",
}
GROUPS = ("retrieve", "rerank", "llm", "check", "other")

AskFn = Callable[[AskRequest], tuple[AskResponse, Any]]


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile (`p` in 0-100). `inf` values sort last."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def stage_split(timings_ms: dict[str, float]) -> dict[str, float]:
    """Seconds per group; `other` is the total minus the named stages."""
    out = dict.fromkeys(GROUPS, 0.0)
    for stage, ms in timings_ms.items():
        if stage == "total":
            continue
        out[STAGE_GROUPS.get(stage, "other")] += ms / 1000.0
    total = timings_ms.get("total")
    if total is not None:
        named = sum(v for k, v in out.items() if k != "other")
        out["other"] = max(0.0, total / 1000.0 - named)
    return out


def rate_limited(trace: Any) -> bool:
    err = getattr(trace, "llm_error", None) or ""
    return "RateLimit" in err or "429" in err


@dataclass
class Sample:
    id: str
    seconds: float
    status: str
    failed: bool
    stages: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seconds": self.seconds,
            "status": self.status,
            "failed": self.failed,
            "stages": self.stages,
        }


class LatencyAborted(RuntimeError):
    pass


def run_latency(
    questions: Sequence[tuple[str, AskRequest]],
    ask: AskFn,
    *,
    warmup: int,
    request_limit_s: float,
    throttle: Throttle,
    clock: PausableClock,
    retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
) -> list[Sample]:
    """Time each question once, warm, one at a time. Warm-up requests reuse the first questions."""
    samples: list[Sample] = []
    for i in range(min(warmup, len(questions))):
        throttle.wait()
        ask(questions[i][1])
    for qid, request in questions:
        attempt = 0
        while True:
            throttle.wait()
            t0 = clock.wall()
            response, trace = ask(request)
            seconds = clock.wall() - t0
            if not rate_limited(trace):
                break
            if attempt >= retries:
                raise LatencyAborted(f"{qid}: still rate-limited after {retries} retries")
            clock.sleep(min(backoff_base_s * 2**attempt, backoff_max_s))
            attempt += 1
        failed = response.status is Status.ERROR or seconds > request_limit_s
        samples.append(
            Sample(qid, seconds, response.status.value, failed, stage_split(response.timings_ms))
        )
    return samples


def summarize(samples: Sequence[Sample], *, p95_target_s: float) -> dict[str, Any]:
    values = [math.inf if s.failed else s.seconds for s in samples]
    p50, p95 = percentile(values, 50), percentile(values, 95)
    ok = [s for s in samples if not s.failed]
    stages = {
        g: {
            "p50": percentile([s.stages.get(g, 0.0) for s in ok], 50),
            "p95": percentile([s.stages.get(g, 0.0) for s in ok], 95),
        }
        for g in GROUPS
    }
    return {
        "requests": len(samples),
        "failures": sum(1 for s in samples if s.failed),
        "p50_s": None if p50 is None or math.isinf(p50) else p50,
        "p95_s": None if p95 is None or math.isinf(p95) else p95,
        "p95_target_s": p95_target_s,
        "passed": p95 is not None and not math.isinf(p95) and p95 <= p95_target_s,
        "statuses": {
            st: sum(1 for s in samples if s.status == st)
            for st in sorted({s.status for s in samples})
        },
        "stages_s": stages,
        "samples": [s.as_dict() for s in samples],
    }


def fixed_query_set(items: Sequence[Any], count: int) -> list[tuple[str, AskRequest]]:
    """The first `count` items by id, as (id, request)."""
    chosen = sorted(items, key=lambda it: it.id)[:count]
    return [(it.id, AskRequest(question=it.question, filters=it.filters)) for it in chosen]
