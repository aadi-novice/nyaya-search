"""Migration test against the docker-compose database.

Skipped when the database is unreachable (Docker absent or not started). It
never starts Docker itself. The migrations run in a throwaway schema, so the
real tables are not touched.
"""

import os
import uuid

import pytest

from nyaya.store import db

EXPECTED_TABLES = {
    "index_versions",
    "judgments",
    "chunks",
    "chunk_vectors",
    "aliases",
    "links",
    "refs",
    "traces",
    "eval_runs",
    "schema_migrations",
}


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        from nyaya.config import load_settings

        return load_settings().secrets.database_url
    except Exception:
        return None


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
    schema = f"test_mig_{uuid.uuid4().hex[:8]}"
    connection.execute(f"CREATE SCHEMA {schema}")
    connection.execute(f"SET search_path TO {schema}, public")
    try:
        yield connection, schema
    finally:
        connection.execute(f"DROP SCHEMA {schema} CASCADE")
        connection.close()


def test_apply_migrations_creates_schema_and_is_idempotent(conn):
    connection, schema = conn
    applied = db.apply_migrations(connection)
    assert applied == ["001_schema.sql", "002_trace_error.sql"]
    tables = {
        r[0]
        for r in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (schema,),
        )
    }
    assert tables >= EXPECTED_TABLES
    ext = connection.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    version = ext.fetchone()
    assert version is not None
    major, minor = (int(x) for x in version[0].split(".")[:2])
    assert (major, minor) >= (0, 8)
    # second run applies nothing
    assert db.apply_migrations(connection) == []
    recorded = [r[0] for r in connection.execute("SELECT filename FROM schema_migrations")]
    assert recorded == ["001_schema.sql", "002_trace_error.sql"]


def test_generated_tsv_and_cascade(conn):
    connection, _ = conn
    db.apply_migrations(connection)
    with connection.transaction():
        connection.execute(
            "INSERT INTO index_versions (id, status, clean_text_version, chunker_version,"
            " config_version, embed_model) VALUES ('t1', 'building', 'ct1', 'ch1', 'cfg1', 'm')"
        )
        connection.execute(
            "INSERT INTO judgments (index_version, path, has_pdf) VALUES ('t1', 'p', false)"
        )
        connection.execute(
            "INSERT INTO chunks (index_version, chunk_id, judgment_id, zone, text, header,"
            " char_start, char_end, clean_text_version, chunker_version)"
            " VALUES ('t1', 'p#judgment#001', 'p', 'judgment', 'bail granted', 'Header',"
            " 0, 12, 'ct1', 'ch1')"
        )
    tsv = connection.execute("SELECT tsv::text FROM chunks").fetchone()[0]
    assert "bail" in tsv and "header" in tsv
    with connection.transaction():
        connection.execute("DELETE FROM index_versions WHERE id = 't1'")
    assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM judgments").fetchone()[0] == 0
