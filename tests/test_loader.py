"""Loader: reading the build, the post-load checks, and load/activate flow.

The flow tests use a recording fake connection; the real-database tests are
in test_store_search_db.py and skip without Postgres.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nyaya.config import load_settings
from nyaya.store import loader as ld
from search_fakes import FakeEmbedder, FakeEngine, WordCountReranker, make_chunk

DIM = 1024


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def judgment_df(paths, **extra):
    n = len(paths)
    data = {
        "path": paths,
        "title": [f"{p} v. State" for p in paths],
        "petitioner": ["A"] * n,
        "respondent": ["State"] * n,
        "judges": [np.array(["Abhay S. Oka, J.", "Ujjal Bhuyan"])] * n,
        "author_judge": [None] * n,
        "scr_citation": ["[2025] 1 S.C.R. 230"] * n,
        "neutral_citation": ["2025INSC31"] * n,
        "cnr": ["ESCR010004242025"] * n,
        "disposal": ["Dismissed"] * n,
        "outcome": ["dismissed"] * n,
        "decision_date": [pd.Timestamp("2025-01-08")] * n,
        "decision_year": [2025] * n,
        "era": ["d"] * n,
        "has_pdf": [True] * n,
        "tar_file": ["x.tar"] * n,
        "offset_data": [512] * n,
        "byte_size": [1000] * n,
        "pages": [9] * n,
        "clean_text": ["clean"] * n,
        "position_map": [json.dumps([[0, 230], [50, 231]])] * n,
    }
    data.update(extra)
    return pd.DataFrame(data)


def write_build(root, chunks, vector_ids=None, metrics=None, aliases=None, links=None, refs=None):
    (root / "judgments").mkdir(parents=True)
    paths = sorted({c.judgment_id for c in chunks}) or ["p0"]
    judgment_df(paths).to_parquet(root / "judgments" / "2025.parquet")
    (root / "chunks").mkdir()
    with (root / "chunks" / "2025.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    (root / "vectors").mkdir()
    ids = vector_ids if vector_ids is not None else [c.chunk_id for c in chunks]
    vecs = np.random.default_rng(0).normal(size=(len(ids), DIM)).astype(np.float32)
    np.save(root / "vectors" / "2025.npy", vecs)
    (root / "vectors" / "2025.ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    pd.DataFrame(
        aliases or [{"alias_norm": "insc:2025:31", "kind": "neutral", "path": paths[0]}],
        columns=list(ld.ALIAS_COLUMNS),
    ).to_parquet(root / "aliases.parquet")
    (root / "links").mkdir()
    pd.DataFrame(links or [], columns=list(ld.LINK_COLUMNS)).to_parquet(
        root / "links" / "2025.parquet"
    )
    (root / "refs").mkdir()
    pd.DataFrame(
        refs
        or [{"path": paths[0], "kind": "act", "value": "IPC", "value_norm": "penal code, 1860"}],
        columns=list(ld.REF_COLUMNS),
    ).to_parquet(root / "refs" / "2025.parquet")
    m = metrics or {"paragraph_integrity": 0.99, "size_compliance": 0.995, "over_model_limit": 0}
    (root / "chunk_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    return root


CHUNKS = [
    make_chunk("2025_1_230_238", text="anticipatory bail section 438"),
    make_chunk("2025_1_230_238", zone="editorial", text="held bail"),
    make_chunk("2025_2_1_9", text="tenancy rent arrears"),
]


# --- reading -----------------------------------------------------------------------------


def test_judgment_rows_cleaning():
    rows = ld.judgment_rows(judgment_df(["p"], author_judge=[float("nan")]))
    r = rows[0]
    assert r["judges"] == ["Abhay S. Oka, J.", "Ujjal Bhuyan"]
    assert r["judges_norm"] == ["ABHAY S OKA", "UJJAL BHUYAN"]
    assert r["decision_date"] == date(2025, 1, 8)
    assert r["author_judge"] is None
    assert r["position_map"] == [[0, 230], [50, 231]]
    assert isinstance(r["decision_year"], int) and isinstance(r["has_pdf"], bool)


def test_judgment_rows_no_pdf_and_derived_year():
    df = judgment_df(
        ["p"],
        has_pdf=[False],
        clean_text=[None],
        position_map=[None],
        era=[None],
        decision_year=[None],
        decision_date=["2019-03-04"],
        judges_norm=[np.array(["GIVEN"])],
    )
    r = ld.judgment_rows(df)[0]
    assert r["has_pdf"] is False and r["clean_text"] is None and r["position_map"] is None
    assert r["decision_year"] == 2019 and r["judges_norm"] == ["GIVEN"]


def test_judgment_rows_position_map_as_list_and_missing_columns():
    df = pd.DataFrame({"path": ["p"], "has_pdf": [True], "position_map": [[[0, 1]]]})
    r = ld.judgment_rows(df)[0]
    assert r["position_map"] == [[0, 1]] and r["title"] is None and r["judges"] == []
    with pytest.raises(ValueError, match="has_pdf"):
        ld.judgment_rows(pd.DataFrame({"path": ["p"]}))


def test_read_build(tmp_path):
    other = {"alias_norm": "insc:2020:1", "kind": "neutral", "path": "not_loaded"}
    mine = {"alias_norm": "insc:2025:31", "kind": "neutral", "path": "2025_1_230_238"}
    write_build(
        tmp_path,
        CHUNKS,
        aliases=[mine, other],
        links=[{"from_chunk_id": CHUNKS[1].chunk_id, "to_chunk_id": CHUNKS[0].chunk_id}],
    )
    data = ld.read_build(tmp_path, DIM)
    assert data.paths == {"2025_1_230_238", "2025_2_1_9"}
    assert len(data.chunks) == 3 and data.vectors.shape == (3, DIM)
    assert data.aliases == [("insc:2025:31", "neutral", "2025_1_230_238")]  # loaded paths only
    assert data.links == [(CHUNKS[1].chunk_id, CHUNKS[0].chunk_id)]
    assert data.refs[0][3] == "penal code, 1860"


def test_read_build_errors(tmp_path):
    with pytest.raises(ValueError, match="no judgment files"):
        ld.read_build(tmp_path, DIM)


def test_read_build_orphan_chunks(tmp_path):
    write_build(tmp_path, CHUNKS)
    judgment_df(["2025_1_230_238"]).to_parquet(tmp_path / "judgments" / "2025.parquet")
    with pytest.raises(ValueError, match="not in the build"):
        ld.read_build(tmp_path, DIM)


def test_read_build_duplicate_judgment(tmp_path):
    write_build(tmp_path, CHUNKS)
    judgment_df(["2025_2_1_9"]).to_parquet(tmp_path / "judgments" / "2026.parquet")
    with pytest.raises(ValueError, match="twice"):
        ld.read_build(tmp_path, DIM)


def test_read_vectors_shape_mismatch(tmp_path):
    write_build(tmp_path, CHUNKS)
    (tmp_path / "vectors" / "2025.ids.txt").write_text("a\nb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ld.read_vectors(tmp_path, DIM)
    (tmp_path / "vectors" / "2025.ids.txt").unlink()
    with pytest.raises(ValueError, match="missing"):
        ld.read_vectors(tmp_path, DIM)


def test_read_chunks_bad_line(tmp_path):
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks" / "2025.jsonl").write_text('{"chunk_id": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"2025\.jsonl:1"):
        ld.read_chunks(tmp_path)


def test_parquet_missing_columns(tmp_path):
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "a.parquet")
    with pytest.raises(ValueError, match="missing columns"):
        ld._read_parquets([tmp_path / "a.parquet"], ld.LINK_COLUMNS)


# --- checks ------------------------------------------------------------------------------


def test_check_ids():
    assert ld.check_ids(["a", "b"], ["b", "a"]) == []
    f = ld.check_ids(["a", "b"], ["a", "a", "c"])
    assert any("duplicate" in x for x in f)
    assert any("no vector" in x for x in f)
    assert any("no chunk" in x for x in f)


def test_check_g6(tmp_path, settings):
    p = tmp_path / "m.json"
    assert ld.check_g6(p, settings) == [f"g6: {p} not found"]
    p.write_text(
        json.dumps({"paragraph_integrity": 0.99, "size_compliance": 0.99, "over_model_limit": 0})
    )
    assert ld.check_g6(p, settings) == []
    p.write_text(json.dumps({"paragraph_integrity": 0.5, "over_model_limit": 2}))
    f = ld.check_g6(p, settings)
    assert any("paragraph_integrity 0.5" in x for x in f)
    assert any("size_compliance missing" in x for x in f)
    assert any("2 chunks over" in x for x in f)
    p.write_text(json.dumps({"paragraph_integrity": 1, "size_compliance": 1}))
    assert ld.check_g6(p, settings) == ["g6: over_model_limit missing"]


def smoke_questions():
    return [
        {
            "id": "s1",
            "question": "2025 INSC 31",
            "expected_route": "citation_lookup",
            "expected_path": "2025_1_230_238",
        },
        {
            "id": "s2",
            "question": "anticipatory bail section 438",
            "expected_route": "hybrid",
            "expected_path": "2025_1_230_238",
        },
        {
            "id": "s3",
            "question": "rent arrears",
            "expected_route": "hybrid",
            "expected_path": "2025_2_1_9",
            "filters": {"year_from": 2025},
        },
        {
            "id": "s4",
            "question": "space law on mars",
            "expected_route": "hybrid",
            "expected_path": None,
        },
    ]


def smoke_settings(settings, min_hits):
    return settings.model_copy(
        update={"load": settings.load.model_copy(update={"smoke_min_hits": min_hits})}
    )


def test_smoke_check_passes(settings):
    engine = FakeEngine(CHUNKS, aliases={"insc:2025:31": "2025_1_230_238"}, vector_order=[])
    failures, out = ld.smoke_check(
        smoke_questions(), engine, FakeEmbedder(), WordCountReranker(), smoke_settings(settings, 4)
    )
    assert failures == []
    assert [o.hit for o in out] == [True, True, True, True]
    assert out[0].route == "citation_lookup" and out[0].found_paths == ["2025_1_230_238"]
    assert engine.calls[2][2].year_from == 2025  # filters reached the search


def test_smoke_unresolved_citation_falls_back_to_hybrid(settings):
    engine = FakeEngine(CHUNKS, aliases={}, vector_order=[])
    qs = [
        {
            "id": "c",
            "question": "2025 INSC 31 anticipatory bail",
            "expected_route": "citation_lookup",
            "expected_path": "2025_1_230_238",
        }
    ]
    failures, out = ld.smoke_check(
        qs, engine, FakeEmbedder(), WordCountReranker(), smoke_settings(settings, 1)
    )
    assert failures == [] and out[0].hit
    assert any(c[0] == "keyword" for c in engine.calls)


def test_smoke_check_failures(settings):
    engine = FakeEngine(CHUNKS, aliases={}, vector_order=[])
    qs = smoke_questions()
    qs[1]["expected_route"] = "citation_lookup"
    failures, _ = ld.smoke_check(
        qs, engine, FakeEmbedder(), WordCountReranker(), smoke_settings(settings, 4)
    )
    assert "smoke: route mismatch for ['s2']" in failures
    assert any("hits < 4" in f for f in failures)


# --- flow with a fake connection ------------------------------------------------------------


class Cur:
    def __init__(self, rows=None):
        self.rows = rows or []

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class Copy:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write_row(self, row):
        self.sink.append(row)


class Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def copy(self, sql):
        rows = []
        self.conn.copies.append((sql, rows))
        return Copy(rows)


class Tx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, responder=None):
        self.executed = []
        self.copies = []
        self.responder = responder or (lambda sql, params: [])

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return Cur(self.responder(sql, params))

    def cursor(self):
        return Cursor(self)

    def transaction(self):
        return Tx()


def id_check_ok(sql, params):
    return [(0, 0)] if "NOT EXISTS" in sql else []


def write_smoke(path, qs):
    path.write_text("\n".join(json.dumps(q) for q in qs) + "\n", encoding="utf-8")


def engine_for(_v):
    return FakeEngine(CHUNKS, aliases={"insc:2025:31": "2025_1_230_238"}, vector_order=[])


def test_load_flow_success(tmp_path, settings):
    build = write_build(tmp_path / "build", CHUNKS)
    smoke = tmp_path / "smoke.jsonl"
    write_smoke(smoke, smoke_questions())
    conn = FakeConn(id_check_ok)
    rep = ld.load(
        "v1",
        build,
        settings=smoke_settings(settings, 4),
        conn=conn,
        embedder=FakeEmbedder(),
        reranker=WordCountReranker(),
        smoke_path=smoke,
        engine_factory=engine_for,
    )
    assert rep.status == "ready"
    assert rep.counts == {
        "judgments": 2,
        "chunks": 3,
        "chunk_vectors": 3,
        "aliases": 1,
        "links": 0,
        "refs": 1,
    }
    tables = [sql.split()[1] for sql, _ in conn.copies]
    assert tables == ["judgments", "chunks", "chunk_vectors", "aliases", "links", "refs"]
    insert_sql, insert_params = conn.executed[0]
    assert "'building'" in insert_sql
    assert insert_params[0] == "v1"
    assert insert_params[7:9] == (
        settings.refusal.threshold_qwen,
        settings.refusal.threshold_cohere,
    )
    assert "SET status = 'ready'" in conn.executed[-1][0]
    # vectors go in as pgvector text; chunk arrays as lists
    vec_row = conn.copies[2][1][0]
    assert vec_row[2].startswith("[") and vec_row[2].count(",") == DIM - 1
    chunk_row = conn.copies[1][1][0]
    assert chunk_row[0] == "v1" and chunk_row[6] == ["1"]


def test_load_flow_check_failure_leaves_building(tmp_path, settings):
    build = write_build(
        tmp_path / "build",
        CHUNKS,
        metrics={"paragraph_integrity": 0.1, "size_compliance": 1.0, "over_model_limit": 0},
    )
    conn = FakeConn(lambda sql, p: [(1, 0)] if "NOT EXISTS" in sql else [])
    with pytest.raises(ld.LoadCheckFailed) as exc:
        ld.load(
            "v2",
            build,
            settings=settings,
            conn=conn,
            embedder=FakeEmbedder(),
            reranker=WordCountReranker(),
            smoke_path=tmp_path / "absent.jsonl",
            engine_factory=engine_for,
        )
    assert exc.value.exit_code == 5
    msg = str(exc.value)
    assert "ids: 1 loaded chunks have no vector" in msg
    assert "paragraph_integrity" in msg and "smoke:" in msg
    assert not any("'ready'" in sql for sql, _ in conn.executed)


def test_load_refuses_inconsistent_build(tmp_path, settings):
    build = write_build(tmp_path / "build", CHUNKS, vector_ids=[c.chunk_id for c in CHUNKS[:2]])
    conn = FakeConn()
    with pytest.raises(ValueError, match="nothing loaded"):
        ld.load(
            "v3",
            build,
            settings=settings,
            conn=conn,
            embedder=FakeEmbedder(),
            reranker=WordCountReranker(),
            engine_factory=engine_for,
        )
    assert conn.executed == []


def test_activate_flow():
    state = {"v1": "active", "v0": "retired", "v2": "ready"}

    def responder(sql, params):
        if sql.startswith("SELECT status"):
            return [(state[params[0]],)] if params[0] in state else []
        if "status = 'active' AND id <>" in sql:
            return [(k,) for k, v in state.items() if v == "active" and k != params[0]]
        if sql.startswith("DELETE"):
            return [(k,) for k, v in state.items() if v == "retired" and k not in params[0]]
        return []

    conn = FakeConn(responder)
    out = ld.activate(conn, "v2")
    assert out == {"active": "v2", "retired": ["v1"], "deleted": ["v0"]}
    with pytest.raises(LookupError):
        ld.activate(conn, "nope")
    state["v3"] = "building"
    with pytest.raises(ValueError, match="building"):
        ld.activate(conn, "v3")


def test_activate_first_version_deletes_nothing():
    def responder(sql, params):
        if sql.startswith("SELECT status"):
            return [("ready",)]
        return []

    conn = FakeConn(responder)
    assert ld.activate(conn, "v1") == {"active": "v1", "retired": [], "deleted": []}
    assert not any(sql.startswith("DELETE") for sql, _ in conn.executed)
