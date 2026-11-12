from contextlib import contextmanager

import pytest

from nyaya.eval.datasets import DevItem, Gold
from nyaya.eval.retrieval import Ranking, RetrievalReport
from nyaya.eval.size_steps import (
    StepResult,
    build_step_schema,
    compare_steps,
    drop_step_schema,
    nested_steps,
    parse_targets,
    run_size_steps,
    step_name,
)

COLUMNS = {
    "judgments": ["index_version", "path", "title"],
    "chunks": ["index_version", "chunk_id", "judgment_id", "zone", "text"],
    "chunk_vectors": ["index_version", "chunk_id", "embedding"],
    "links": ["index_version", "from_chunk_id", "to_chunk_id"],
    "refs": ["index_version", "path", "kind", "value", "value_norm"],
}


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, counts=None):
        self.sql: list[tuple[str, object]] = []
        self.counts = counts or {}
        self.transactions = 0

    @contextmanager
    def transaction(self):
        self.transactions += 1
        yield

    def execute(self, query, params=None):
        self.sql.append((query, params))
        if "information_schema.columns" in query:
            return Result([(c,) for c in COLUMNS[params[0]]])
        if "GROUP BY judgment_id" in query:
            return Result(list(self.counts.items()))
        return Result([])


def test_parse_targets():
    assert parse_targets("10000, 50_000,full") == [10000, 50000, None]


def test_nested_steps_contain_gold_and_nest():
    counts = {f"p{i:02d}": 10 for i in range(20)}
    steps = nested_steps(counts, ["p05", "p17", "unknown"], [30, 100, None], seed=7)
    assert [len(s) for s in steps] == [3, 10, 20]
    for s in steps:
        assert {"p05", "p17"} <= set(s)
    assert steps[0] == steps[1][:3] and steps[1] == steps[2][:10]
    assert steps == nested_steps(counts, ["p17", "p05"], [30, 100, None], seed=7)


def test_nested_steps_keeps_all_gold_even_above_target():
    counts = {"a": 100, "b": 100, "c": 1}
    steps = nested_steps(counts, ["a", "b"], [50], seed=1)
    assert sorted(steps[0]) == ["a", "b"]


def test_nested_steps_targets_must_increase():
    with pytest.raises(ValueError, match="increase"):
        nested_steps({"a": 1}, [], [10, 5], seed=1)


def test_build_step_schema_sql():
    conn = FakeConn()
    build_step_schema(conn, "g4_step_2", "v2025", ["p1", "p2"])
    text = [q for q, _ in conn.sql]
    assert text[0] == "DROP SCHEMA IF EXISTS g4_step_2 CASCADE"
    assert text[1] == "CREATE SCHEMA g4_step_2"
    assert (
        "CREATE TABLE g4_step_2.chunks (LIKE public.chunks INCLUDING DEFAULTS INCLUDING GENERATED)"
        in text
    )
    inserts = [(q, p) for q, p in conn.sql if q.startswith("INSERT")]
    assert len(inserts) == 5
    q, p = next((q, p) for q, p in inserts if "g4_step_2.chunks" in q)
    assert "SELECT %(name)s, chunk_id, judgment_id, zone, text FROM public.chunks" in q
    assert "judgment_id = ANY(%(paths)s)" in q
    assert p == {"name": "g4_step_2", "version": "v2025", "paths": ["p1", "p2"]}
    links = next(q for q, _ in inserts if "g4_step_2.links" in q)
    assert "to_chunk_id IN (SELECT chunk_id FROM g4_step_2.chunks)" in links
    assert any("USING hnsw (embedding vector_cosine_ops)" in q for q in text)
    assert any("USING gin (tsv)" in q for q in text)
    assert not any("index_versions" in q for q in text)


def test_schema_names_are_checked():
    with pytest.raises(ValueError):
        build_step_schema(FakeConn(), "public", "v", [])
    with pytest.raises(ValueError):
        drop_step_schema(FakeConn(), "g4_step_1; DROP TABLE chunks")
    conn = FakeConn()
    drop_step_schema(conn, step_name(3))
    assert conn.sql == [("DROP SCHEMA IF EXISTS g4_step_3 CASCADE", None)]


def report(ids, values):
    """A report where every method, level and metric has the given per-question values."""
    metrics = {"recall@5": list(values), "recall@50": list(values), "rr": list(values)}
    per_q = {
        m: {lv: {k: list(v) for k, v in metrics.items()} for lv in ("judgment", "paragraph")}
        for m in ("hybrid", "hybrid_rerank")
    }
    return RetrievalReport(item_ids=ids, per_question=per_q)


def test_compare_steps_pass_and_fail():
    ids = [f"q{i}" for i in range(40)]
    small = StepResult("g4_step_1", 10, 100, report(ids, [1.0] * 40))
    same = StepResult("g4_step_2", 20, 200, report(ids, [1.0] * 39 + [0.0]))
    worse = StepResult("g4_step_2", 20, 200, report(ids, [1.0] * 20 + [0.0] * 20))
    ok = compare_steps([small, same], seed=7, samples=500, max_drop_points=5.0)
    drop = ok.drops["g4_step_1->g4_step_2"]["hybrid/judgment/recall@50"]
    assert drop["mean"] == pytest.approx(2.5) and drop["gated"]
    # one question in 40 lost: the interval's upper bound reaches 7.5 points
    assert drop["high"] > 5.0 and not ok.passed
    bad = compare_steps([small, worse], seed=7, samples=500, max_drop_points=5.0)
    assert not bad.passed
    flat = compare_steps(
        [small, StepResult("g4_step_2", 20, 200, report(ids, [1.0] * 40))],
        seed=7,
        samples=200,
        max_drop_points=5.0,
    )
    assert flat.passed
    assert flat.as_dict()["steps"][1]["chunks"] == 200
    assert compare_steps([small], seed=1, samples=10, max_drop_points=5).passed is False


class StepRetriever:
    def __init__(self, hits):
        self.hits = hits

    def run(self, method, question, filters):
        return Ranking(self.hits)


def test_run_size_steps_builds_evaluates_and_drops():
    from eval_fakes import FakeEngine, chunk

    counts = {"A": 2, "B": 2, "C": 2}
    conn = FakeConn(counts)
    eng = FakeEngine([chunk("A#judgment#001", labels=["1"])])
    made = []

    def make(name):
        made.append(name)
        return StepRetriever(["A#judgment#001"]), eng

    items = [
        DevItem(id="q1", question="x", type="answerable", gold=[Gold(path="A", para_labels=["1"])]),
        DevItem(id="u1", question="y", type="unanswerable"),
    ]
    res = run_size_steps(
        conn,
        items,
        source_version="v1",
        targets=[2, None],
        make_retriever=make,
        seed=7,
        bootstrap_samples=100,
        max_drop_points=5.0,
        methods=["hybrid", "hybrid_rerank"],
    )
    assert made == ["g4_step_1", "g4_step_2"]
    assert [s.judgments for s in res.steps] == [1, 3]
    assert res.passed
    drops = [q for q, _ in conn.sql if q.startswith("DROP SCHEMA")]
    assert len(drops) == 4  # one before each build, one after each run

    with pytest.raises(ValueError, match="gold judgments not in"):
        run_size_steps(
            FakeConn({"B": 1}),
            items,
            source_version="v1",
            targets=[None],
            make_retriever=make,
            seed=7,
            bootstrap_samples=10,
            max_drop_points=5.0,
        )
