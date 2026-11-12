from __future__ import annotations

import logging

from psycopg.types.json import Jsonb

from nyaya.trace import TraceRecord, write_trace


def record(**kw) -> TraceRecord:
    base = {
        "request_id": "r1",
        "question": "q",
        "route": "hybrid",
        "filters": {"year_from": 2020},
        "versions": {"index_version": "v1"},
        "retrieved": [{"chunk_id": "a#judgment#1", "rerank_score": 0.9}],
        "rows_returned": {"keyword": {"requested": 50, "returned": 3}},
        "top_score": 0.9,
        "threshold": 0.35,
        "status": "answered",
        "rule_breaks": [{"kind": "quote_not_found"}],
        "unverified_count": 1,
        "gaps_count": 2,
        "reranker_fallback_used": False,
        "llm_error": None,
        "timings_ms": {"total": 12.5},
    }
    base.update(kw)
    return TraceRecord(**base)


class FakeConn:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.executed: list[tuple[str, tuple]] = []
        self.committed = False
        self.closed = False

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("db down")
        self.executed.append((sql, params))

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_write_trace_inserts_one_row_with_json_columns():
    conn = FakeConn()
    assert write_trace(lambda: conn, record()) is True
    ((sql, params),) = conn.executed
    assert sql.strip().startswith("INSERT INTO traces")
    assert sql.count("%s") == len(params) == 17 - 1  # every column but created_at
    assert params[0] == "r1"
    jsonb = [p for p in params if isinstance(p, Jsonb)]
    assert len(jsonb) == 6  # filters, versions, retrieved, rows_returned, rule_breaks, timings
    assert conn.committed and conn.closed


def test_write_failure_is_logged_never_raised(caplog):
    conn = FakeConn(fail=True)
    with caplog.at_level(logging.ERROR, logger="nyaya.trace"):
        assert write_trace(lambda: conn, record()) is False
    assert "r1" in caplog.text
    assert conn.closed


def test_connect_failure_is_logged(caplog):
    def boom():
        raise OSError("no database")

    with caplog.at_level(logging.ERROR, logger="nyaya.trace"):
        assert write_trace(boom, record()) is False
    assert "no database" in caplog.text
