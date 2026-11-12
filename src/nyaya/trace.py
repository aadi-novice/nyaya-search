"""One `traces` row per request (design, file 36). A write failure is logged and
never fails the request."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from psycopg.types.json import Jsonb

from nyaya.contracts import AskRequest, AskResponse

log = logging.getLogger(__name__)

_JSON_COLUMNS = frozenset(
    {"filters", "versions", "retrieved", "rows_returned", "rule_breaks", "timings_ms"}
)


@dataclass
class TraceRecord:
    request_id: str
    question: str
    route: str
    filters: dict[str, Any]
    versions: dict[str, Any]
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    rows_returned: dict[str, Any] = field(default_factory=dict)
    top_score: float | None = None
    threshold: float | None = None
    status: str = "error"
    rule_breaks: list[dict[str, Any]] = field(default_factory=list)
    unverified_count: int = 0
    gaps_count: int = 0
    reranker_fallback_used: bool = False
    llm_error: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None  # why an `error` response failed (migration 002)


_COLUMNS = list(TraceRecord.__dataclass_fields__)
_INSERT = (
    f"INSERT INTO traces ({', '.join(_COLUMNS)}) "  # fixed column names
    f"VALUES ({', '.join(['%s'] * len(_COLUMNS))})"
)


def record_from_response(
    request: AskRequest, response: AskResponse, error: str | None = None
) -> TraceRecord:
    """A trace row for a response built outside the pipeline (the API's timeout path)."""
    return TraceRecord(
        request_id=response.request_id,
        question=request.question,
        route=response.route.value,
        filters=response.filters_applied,
        versions=response.versions.model_dump(),
        status=response.status.value,
        unverified_count=len(response.unverified),
        gaps_count=len(response.gaps),
        reranker_fallback_used=response.versions.reranker_fallback_used,
        timings_ms=response.timings_ms,
        error=error,
    )


def write_trace(connect: Callable[[], Any], record: TraceRecord) -> bool:
    """Insert `record` on a fresh connection from `connect`; True if it was written."""
    conn = None
    try:
        conn = connect()
        values = asdict(record)
        params = tuple(Jsonb(values[c]) if c in _JSON_COLUMNS else values[c] for c in _COLUMNS)
        conn.execute(_INSERT, params)
        conn.commit()
        return True
    except Exception as exc:
        log.error("trace write failed for request %s: %s", record.request_id, exc)
        return False
    finally:
        if conn is not None:
            conn.close()
