"""PostgreSQL + pgvector search engine (design file 31).

Every query is limited to one `index_version`:

* `PostgresSearch()` reads the `active` version once, at construction;
* the loader passes its `building` version explicitly;
* the G4 size steps pass a temporary schema, applied through `search_path`.

The SQL builders are module-level functions so they can be unit-tested
without a database. All user values are bound parameters; the only text
spliced into SQL is fixed fragments and a validated schema identifier.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nyaya.config import OutcomeMap, Settings
from nyaya.contracts import ChunkRecord, Filters, JudgmentRecord, SearchHit
from nyaya.text.normalize import normalize_judge, normalize_text

log = logging.getLogger(__name__)

_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_QUOTED_RE = re.compile(r"[\"“”]([^\"“”]+)[\"“”]")
_WORD_RE = re.compile(r"[^\W_]+")

CHUNK_COLUMNS = (
    "chunk_id",
    "judgment_id",
    "zone",
    "opinion_author",
    "opinion_type",
    "para_labels",
    "para_refs",
    "section_label",
    "text",
    "header",
    "char_start",
    "char_end",
    "pages",
    "flags",
    "clean_text_version",
    "chunker_version",
)

JUDGMENT_COLUMNS = (
    "path",
    "title",
    "petitioner",
    "respondent",
    "judges",
    "author_judge",
    "scr_citation",
    "neutral_citation",
    "cnr",
    "disposal",
    "outcome",
    "decision_date",
    "decision_year",
    "era",
    "has_pdf",
    "pages",
    "clean_text",
    "position_map",
)

# Text columns the contract requires as strings; null in the database becomes "".
_JUDGMENT_TEXT_DEFAULTS = (
    "title",
    "petitioner",
    "respondent",
    "scr_citation",
    "cnr",
    "disposal",
    "outcome",
)


# --- SQL builders (pure) ---------------------------------------------------------


def validate_schema(schema: str) -> str:
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"invalid schema name {schema!r}")
    return schema


def outcome_values(outcome_map: OutcomeMap) -> set[str]:
    return set(outcome_map.values) | {outcome_map.default}


def resolve_outcome(value: str, outcome_map: OutcomeMap) -> str:
    """An outcome filter value: an outcome name ("allowed") or a raw disposal text."""
    v = value.strip()
    if v.lower() in outcome_values(outcome_map):
        return v.lower()
    for outcome, raw in outcome_map.values.items():
        if any(v.lower() == r.lower() for r in raw):
            return outcome
    raise ValueError(
        f"unknown outcome filter {value!r}; expected one of {sorted(outcome_values(outcome_map))}"
    )


@dataclass(frozen=True)
class WhereClause:
    sql: str  # fragment starting with " AND ...", or ""
    params: dict[str, Any]

    @property
    def needs_judgments(self) -> bool:
        return "j." in self.sql


def build_filter_sql(
    filters: Filters | None,
    outcome_map: OutcomeMap,
    act_normalizer: Callable[[str], str] = normalize_text,
) -> WhereClause:
    """The filter part of the WHERE clause, over chunks `c` and judgments `j`."""
    if filters is None:
        return WhereClause("", {})
    parts: list[str] = []
    params: dict[str, Any] = {}
    if filters.year_from is not None:
        parts.append("j.decision_year >= %(year_from)s")
        params["year_from"] = filters.year_from
    if filters.year_to is not None:
        parts.append("j.decision_year <= %(year_to)s")
        params["year_to"] = filters.year_to
    if filters.judge is not None and filters.judge.strip():
        parts.append("j.judges_norm @> ARRAY[%(judge)s]::text[]")
        params["judge"] = normalize_judge(filters.judge)
    if filters.act is not None and filters.act.strip():
        parts.append(
            "EXISTS (SELECT 1 FROM refs r WHERE r.index_version = c.index_version"
            " AND r.path = c.judgment_id AND r.kind = 'act' AND r.value_norm = %(act)s)"
        )
        params["act"] = act_normalizer(filters.act)
    if filters.outcome is not None and filters.outcome.strip():
        parts.append("j.outcome = %(outcome)s")
        params["outcome"] = resolve_outcome(filters.outcome, outcome_map)
    sql = "".join(f" AND {p}" for p in parts)
    return WhereClause(sql, params)


def keyword_terms(question: str) -> tuple[list[str], list[str]]:
    """(words, phrases): normalised words for the OR query, and quoted phrases.

    Words are runs of letters and digits only, so they are safe inside
    `to_tsquery` syntax; duplicates are dropped, order kept.
    """
    phrases = [p.strip() for p in _QUOTED_RE.findall(question) if p.strip()]
    text = normalize_text(question)
    words: list[str] = []
    for w in _WORD_RE.findall(text):
        if w not in words:
            words.append(w)
    return words, phrases


def build_tsquery(words: Sequence[str], phrases: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """A tsquery expression: OR of the words, OR each quoted phrase as an exact phrase."""
    parts: list[str] = []
    params: dict[str, Any] = {}
    if words:
        parts.append("to_tsquery('english', %(or_words)s)")
        params["or_words"] = " | ".join(words)
    for i, phrase in enumerate(phrases):
        parts.append(f"phraseto_tsquery('english', %(phrase_{i})s)")
        params[f"phrase_{i}"] = phrase
    if not parts:
        raise ValueError("no searchable words in the question")
    return "(" + " || ".join(parts) + ")", params


_JOIN_JUDGMENTS = (
    " JOIN judgments j ON j.index_version = c.index_version AND j.path = c.judgment_id"
)


def build_keyword_sql(
    question: str, where: WhereClause, index_version: str, k: int
) -> tuple[str, dict[str, Any]] | None:
    """The keyword query, or None when the question has nothing searchable."""
    words, phrases = keyword_terms(question)
    if not words and not phrases:
        return None
    tsq, params = build_tsquery(words, phrases)
    join = _JOIN_JUDGMENTS if where.needs_judgments else ""
    sql = (
        f"WITH q AS (SELECT {tsq} AS query)"
        " SELECT c.chunk_id, ts_rank_cd(c.tsv, q.query) AS score"
        f" FROM q, chunks c{join}"
        " WHERE c.index_version = %(index_version)s AND c.tsv @@ q.query"
        f"{where.sql}"
        " ORDER BY score DESC, c.chunk_id"
        " LIMIT %(k)s"
    )
    return sql, {**params, **where.params, "index_version": index_version, "k": k}


def build_vector_sql(where: WhereClause, index_version: str, k: int) -> tuple[str, dict[str, Any]]:
    """The vector query. Bind `qvec` separately (a pgvector value).

    With `hnsw.iterative_scan = relaxed_order` rows may come back slightly out
    of order, so the outer query re-sorts by distance (pgvector's advice).
    """
    join = _JOIN_JUDGMENTS if where.needs_judgments else ""
    sql = (
        "WITH r AS MATERIALIZED ("
        " SELECT v.chunk_id, v.embedding <=> %(qvec)s AS distance"
        " FROM chunk_vectors v"
        " JOIN chunks c ON c.index_version = v.index_version AND c.chunk_id = v.chunk_id"
        f"{join}"
        " WHERE v.index_version = %(index_version)s"
        f"{where.sql}"
        " ORDER BY distance LIMIT %(k)s)"
        " SELECT chunk_id, distance FROM r ORDER BY distance, chunk_id"
    )
    return sql, {**where.params, "index_version": index_version, "k": k}


def vector_settings_sql(ef_search: int, exact: bool) -> tuple[str, dict[str, Any]]:
    """Transaction-local planner settings for the vector query (SET LOCAL equivalents)."""
    if exact:
        return (
            "SELECT set_config('enable_indexscan', 'off', true),"
            " set_config('enable_bitmapscan', 'off', true)",
            {},
        )
    return (
        "SELECT set_config('hnsw.ef_search', %(ef)s, true),"
        " set_config('hnsw.iterative_scan', 'relaxed_order', true)",
        {"ef": str(int(ef_search))},
    )


def row_to_chunk(row: Sequence[Any]) -> ChunkRecord:
    data = dict(zip(CHUNK_COLUMNS, row, strict=True))
    for key in ("para_labels", "para_refs", "pages", "flags"):
        data[key] = list(data[key] or [])
    return ChunkRecord.model_validate(data)


def row_to_judgment(row: Sequence[Any]) -> JudgmentRecord:
    data = dict(zip(JUDGMENT_COLUMNS, row, strict=True))
    for key in _JUDGMENT_TEXT_DEFAULTS:
        if data[key] is None:
            data[key] = ""
    data["judges"] = list(data["judges"] or [])
    if data["position_map"] is not None:
        data["position_map"] = [tuple(p) for p in data["position_map"]]
    return JudgmentRecord.model_validate(data)


# --- engine -----------------------------------------------------------------------


class NoActiveVersion(LookupError):
    pass


class PostgresSearch:
    """`SearchEngine` over the Nyaya schema, scoped to one index version.

    Pass an open psycopg connection (autocommit, pgvector registered), or
    settings to connect with `store.db.connect`. `act_normalizer` maps an Act
    filter value to `refs.value_norm` form; the pipeline should pass the
    zoner's alias mapping (design file 20). It defaults to `normalize_text`.
    """

    def __init__(
        self,
        index_version: str | None = None,
        schema: str | None = None,
        *,
        settings: Settings | None = None,
        conn: Any = None,
        act_normalizer: Callable[[str], str] = normalize_text,
    ) -> None:
        if settings is None:
            from nyaya.config import load_settings

            settings = load_settings()
        self.settings = settings
        self.act_normalizer = act_normalizer
        if conn is None:
            from nyaya.store.db import connect

            conn = connect(settings, autocommit=True)
        self.conn = conn
        self.schema = validate_schema(schema) if schema else None
        if self.schema:
            self.conn.execute(
                "SELECT set_config('search_path', %s, false)", (f"{self.schema}, public",)
            )
        self.index_version = index_version or self._active_version()

    def _active_version(self) -> str:
        row = self.conn.execute(
            "SELECT id FROM index_versions WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise NoActiveVersion("no index version is active; run nyaya-build activate")
        return row[0]

    def _where(self, filters: Filters | None) -> WhereClause:
        return build_filter_sql(filters, self.settings.outcome_map, self.act_normalizer)

    def keyword(self, question: str, filters: Filters | None, k: int) -> list[SearchHit]:
        built = build_keyword_sql(question, self._where(filters), self.index_version, k)
        if built is None:
            return []
        sql, params = built
        rows = self.conn.execute(sql, params).fetchall()
        log.info("keyword search: %d rows returned of %d requested", len(rows), k)
        return [
            SearchHit(chunk_id=r[0], keyword_rank=i, fused_rank=float(i))
            for i, r in enumerate(rows, 1)
        ]

    def vector(
        self, qvec: Sequence[float], filters: Filters | None, k: int, exact: bool = False
    ) -> list[SearchHit]:
        sql, params = build_vector_sql(self._where(filters), self.index_version, k)
        params["qvec"] = np.asarray(qvec, dtype=np.float32)
        set_sql, set_params = vector_settings_sql(self.settings.retrieval.ef_search, exact)
        with self.conn.transaction():
            self.conn.execute(set_sql, set_params)
            rows = self.conn.execute(sql, params).fetchall()
        log.info("vector search: %d rows returned of %d requested", len(rows), k)
        return [
            SearchHit(chunk_id=r[0], vector_rank=i, fused_rank=float(i))
            for i, r in enumerate(rows, 1)
        ]

    def get_chunks(self, ids: Sequence[str]) -> list[ChunkRecord]:
        if not ids:
            return []
        cols = ", ".join(CHUNK_COLUMNS)
        rows = self.conn.execute(
            f"SELECT {cols} FROM chunks WHERE index_version = %s AND chunk_id = ANY(%s)",
            (self.index_version, list(ids)),
        ).fetchall()
        by_id = {r[0]: row_to_chunk(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def chunk_ids_for(self, path: str) -> list[str]:
        """All chunk IDs of one judgment, in chunk_id order (zone, then sequence)."""
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE index_version = %s AND judgment_id = %s"
            " ORDER BY chunk_id",
            (self.index_version, path),
        ).fetchall()
        return [r[0] for r in rows]

    def get_judgment(self, path: str) -> JudgmentRecord | None:
        cols = ", ".join(JUDGMENT_COLUMNS)
        row = self.conn.execute(
            f"SELECT {cols} FROM judgments WHERE index_version = %s AND path = %s",
            (self.index_version, path),
        ).fetchone()
        return row_to_judgment(row) if row else None

    def resolve_alias(self, canonical: str) -> str | None:
        row = self.conn.execute(
            "SELECT path FROM aliases WHERE index_version = %s AND alias_norm = %s"
            " ORDER BY path LIMIT 1",
            (self.index_version, canonical),
        ).fetchone()
        return row[0] if row else None

    def links_from(self, chunk_ids: Sequence[str]) -> dict[str, list[str]]:
        if not chunk_ids:
            return {}
        rows = self.conn.execute(
            "SELECT from_chunk_id, to_chunk_id FROM links"
            " WHERE index_version = %s AND from_chunk_id = ANY(%s)"
            " ORDER BY from_chunk_id, to_chunk_id",
            (self.index_version, list(chunk_ids)),
        ).fetchall()
        out: dict[str, list[str]] = {}
        for src, dst in rows:
            out.setdefault(src, []).append(dst)
        return out

    def thresholds(self) -> dict[str, float | None]:
        """Refusal thresholds of this version: {"qwen": ..., "cohere": ...}."""
        row = self.conn.execute(
            "SELECT threshold_qwen, threshold_cohere FROM index_versions WHERE id = %s",
            (self.index_version,),
        ).fetchone()
        if row is None:
            raise LookupError(f"index version {self.index_version!r} not found")
        return {"qwen": row[0], "cohere": row[1]}
