"""Load, search and activate against the docker-compose database.

Skipped when DATABASE_URL is unset or the database is unreachable; it never
starts Docker. Everything runs in a throwaway schema.
"""

import uuid

import numpy as np
import pytest

from nyaya.config import load_settings
from nyaya.contracts import Filters
from nyaya.store import db
from nyaya.store import loader as ld
from search_fakes import WordCountReranker
from test_db import database_url
from test_loader import CHUNKS, smoke_settings, write_build, write_smoke


@pytest.fixture
def conn():
    psycopg = pytest.importorskip("psycopg")
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set; the docker-compose database is not configured")
    try:
        connection = psycopg.connect(url, connect_timeout=3, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database unreachable (is Docker running?): {exc}")
    schema = f"test_load_{uuid.uuid4().hex[:8]}"
    connection.execute(f"CREATE SCHEMA {schema}")
    connection.execute(f"SET search_path TO {schema}, public")
    try:
        db.apply_migrations(connection)
        from pgvector.psycopg import register_vector

        register_vector(connection)
        yield connection, schema
    finally:
        connection.execute(f"DROP SCHEMA {schema} CASCADE")
        connection.close()


class VecEmbedder:
    """Embeds every question as the vector stored for the first chunk."""

    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, question):
        return self.vec


def test_load_search_activate(conn, tmp_path):
    connection, schema = conn
    settings = smoke_settings(load_settings(env_file=None), 2)
    build = write_build(
        tmp_path / "build",
        CHUNKS,
        links=[{"from_chunk_id": CHUNKS[1].chunk_id, "to_chunk_id": CHUNKS[0].chunk_id}],
    )
    first_vec = np.load(build / "vectors" / "2025.npy")[0]
    smoke = tmp_path / "smoke.jsonl"
    write_smoke(
        smoke,
        [
            {
                "id": "a",
                "question": "2025 INSC 31",
                "expected_route": "citation_lookup",
                "expected_path": "2025_1_230_238",
            },
            {
                "id": "b",
                "question": "anticipatory bail",
                "expected_route": "hybrid",
                "expected_path": "2025_1_230_238",
            },
        ],
    )
    from nyaya.search.postgres import PostgresSearch

    def factory(v):
        return PostgresSearch(index_version=v, schema=schema, settings=settings, conn=connection)

    rep = ld.load(
        "v1",
        build,
        settings=settings,
        conn=connection,
        embedder=VecEmbedder(first_vec),
        reranker=WordCountReranker(),
        smoke_path=smoke,
        engine_factory=factory,
    )
    assert rep.status == "ready"

    eng = factory("v1")
    kw = eng.keyword('"anticipatory bail" 438', None, 10)
    assert kw[0].chunk_id == CHUNKS[0].chunk_id
    vec = eng.vector(first_vec, None, 3)
    assert vec[0].chunk_id == CHUNKS[0].chunk_id
    exact = eng.vector(first_vec, None, 3, exact=True)
    assert [h.chunk_id for h in exact] == [h.chunk_id for h in vec]
    assert eng.vector(first_vec, Filters(year_from=2030), 3) == []
    assert eng.keyword("bail", Filters(act="penal code, 1860"), 10)
    assert eng.keyword("bail", Filters(judge="ABHAY S. OKA, J."), 10)
    assert eng.keyword("bail", Filters(outcome="allowed"), 10) == []
    assert eng.resolve_alias("insc:2025:31") == "2025_1_230_238"
    assert eng.links_from([CHUNKS[1].chunk_id]) == {CHUNKS[1].chunk_id: [CHUNKS[0].chunk_id]}
    j = eng.get_judgment("2025_1_230_238")
    assert j.decision_year == 2025 and j.position_map == [(0, 230), (50, 231)]
    assert [c.chunk_id for c in eng.get_chunks([CHUNKS[2].chunk_id])] == [CHUNKS[2].chunk_id]
    assert eng.thresholds() == {"qwen": 0.35, "cohere": 0.20}

    out = ld.activate(connection, "v1")
    assert out == {"active": "v1", "retired": [], "deleted": []}
    assert PostgresSearch(schema=schema, settings=settings, conn=connection).index_version == "v1"
