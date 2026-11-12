"""Database connection and migrations (PostgreSQL 17 + pgvector)."""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from nyaya.config import ConfigError, Settings, load_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "db" / "migrations"

_CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def _database_url(settings: Settings) -> str:
    url = settings.secrets.database_url
    if not url:
        raise ConfigError("DATABASE_URL: missing (set it in the environment or .env)")
    return url


def connect(settings: Settings, **kwargs: object) -> psycopg.Connection:
    """Open a psycopg connection and register the pgvector type."""
    conn = psycopg.connect(_database_url(settings), **kwargs)  # type: ignore[arg-type]
    register_vector(conn)
    return conn


def apply_migrations(
    conn: psycopg.Connection | None = None,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[str]:
    """Run the not-yet-applied `*.sql` files in name order.

    Each file runs in its own transaction and is recorded in
    `schema_migrations`. Returns the names of the files applied.
    Without `conn`, connects using DATABASE_URL from the settings.
    """
    own = conn is None
    if conn is None:
        # pgvector is not registered here: the extension may not exist yet.
        conn = psycopg.connect(_database_url(load_settings()))
    applied: list[str] = []
    try:
        with conn.transaction():
            conn.execute(_CREATE_MIGRATIONS_TABLE)
        done = {r[0] for r in conn.execute("SELECT filename FROM schema_migrations")}
        if not conn.autocommit:
            conn.commit()
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(sql)  # type: ignore[arg-type]  # no params: multi-statement
                conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            applied.append(path.name)
    finally:
        if own:
            conn.close()
    return applied


if __name__ == "__main__":
    names = apply_migrations()
    print(f"applied {len(names)} migration(s): {', '.join(names) or 'none'}")
