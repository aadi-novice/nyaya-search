"""Load a build into a new index version, check it, and activate it (design
step 8 "Load", file 29).

`load(version, build_dir)`:

1. inserts the `index_versions` row as `building`, with `threshold_qwen` and
   `threshold_cohere` seeded from `refusal.*`;
2. COPYs, in order: judgments, chunks, vectors, aliases (only for the paths
   just loaded), links, refs;
3. refreshes planner statistics. The HNSW and GIN indexes are defined once by
   the migration over the shared tables and are maintained as rows arrive;
   dropping them per load would also take them away from the active version;
4. runs the post-load checks against the `building` version itself
   (`PostgresSearch(index_version=version)`), never the LLM: vector IDs equal
   chunk IDs, the G6 automatic metrics pass, and the smoke check;
5. sets `ready` if they pass. Otherwise the version stays `building` and
   `LoadCheckFailed` (exit code 5) names the failing checks.

`activate(version)` moves a `ready` (or, for a rollback, `retired`) version to
`active`, retires the previous active version, and deletes older retired
versions, so production keeps the active version and one retired version.

Build inputs (written by `nyaya-build parse-chunk` and `embed`; `years`
limits the per-year files read):

    judgments/<year>.parquet    one row per judgment, judgments-table columns
    chunks/<year>.jsonl         one ChunkRecord per line
    vectors/<year>.npy          float32 (n, embed.dim)
    vectors/<year>.ids.txt      n chunk IDs, one per line, in row order
    aliases.parquet             alias_norm, kind, path
    links/<year>.parquet        from_chunk_id, to_chunk_id
    refs/<year>.parquet         path, kind, value, value_norm
    chunk_metrics.json          paragraph_integrity, size_compliance, over_model_limit
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nyaya.config import Settings
from nyaya.contracts import ChunkRecord, Filters, Route
from nyaya.search.interface import SearchEngine
from nyaya.search.retrieve import QueryEmbedder, RerankService, hybrid_retrieve
from nyaya.search.router import lookup_canonicals, route
from nyaya.text.normalize import normalize_judge

log = logging.getLogger(__name__)

LOAD_CHECK_EXIT_CODE = 5

JUDGMENT_LOAD_COLUMNS = (
    "path",
    "title",
    "petitioner",
    "respondent",
    "judges",
    "judges_norm",
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
    "tar_file",
    "offset_data",
    "byte_size",
    "pages",
    "clean_text",
    "position_map",
)
CHUNK_LOAD_COLUMNS = (
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
ALIAS_COLUMNS = ("alias_norm", "kind", "path")
LINK_COLUMNS = ("from_chunk_id", "to_chunk_id")
REF_COLUMNS = ("path", "kind", "value", "value_norm")


class LoadCheckFailed(Exception):
    """A post-load check failed; the version stays `building`."""

    exit_code = LOAD_CHECK_EXIT_CODE

    def __init__(self, version: str, failures: Sequence[str]):
        self.version = version
        self.failures = list(failures)
        super().__init__(f"index version {version!r} failed: " + "; ".join(self.failures))


# --- reading the build (pure) -----------------------------------------------------------


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return v is pd.NaT


def _clean(v: Any) -> Any:
    """Parquet/pandas values to plain Python for COPY."""
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    if _is_missing(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.date()
    return v


def _as_date(v: Any) -> date | None:
    v = _clean(v)
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def _as_position_map(v: Any) -> list[list[int]] | None:
    v = _clean(v)
    if v is None:
        return None
    if isinstance(v, str):
        v = json.loads(v)
    return [[int(a), int(b)] for a, b in v]


def judgment_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Judgment rows in JUDGMENT_LOAD_COLUMNS form. `judges_norm` is derived when absent."""
    for required in ("path", "has_pdf"):
        if required not in df.columns:
            raise ValueError(f"judgments parquet: missing column {required!r}")
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        row = {c: _clean(rec.get(c)) for c in JUDGMENT_LOAD_COLUMNS}
        row["judges"] = list(row["judges"] or [])
        if row["judges_norm"] is None:
            row["judges_norm"] = [normalize_judge(j) for j in row["judges"]]
        row["decision_date"] = _as_date(row["decision_date"])
        if row["decision_year"] is None and row["decision_date"] is not None:
            row["decision_year"] = row["decision_date"].year
        row["position_map"] = _as_position_map(rec.get("position_map"))
        row["has_pdf"] = bool(row["has_pdf"])
        rows.append(row)
    return rows


def _year_files(directory: Path, suffix: str, years: Iterable[int] | None = None) -> list[Path]:
    """`<year><suffix>` files in `directory`, all or only those of `years`."""
    if not directory.is_dir():
        return []
    files = sorted(directory.glob(f"*{suffix}"))
    if years is None:
        return files
    wanted = {str(y) for y in years}
    return [f for f in files if f.name[: -len(suffix)] in wanted]


def read_chunks(build: Path, years: Iterable[int] | None = None) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for path in _year_files(build / "chunks", ".jsonl", years):
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        chunks.append(ChunkRecord.model_validate_json(line))
                    except ValueError as exc:
                        raise ValueError(f"{path}:{n}: {exc}") from exc
    return chunks


def read_vectors(
    build: Path, dim: int, years: Iterable[int] | None = None
) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    for npy in _year_files(build / "vectors", ".npy", years):
        ids_file = npy.with_suffix(".ids.txt")
        if not ids_file.exists():
            raise ValueError(f"{npy}: missing {ids_file.name}")
        block_ids = [s for s in ids_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        block = np.load(npy).astype(np.float32, copy=False)
        if block.ndim != 2 or block.shape != (len(block_ids), dim):
            raise ValueError(
                f"{npy}: shape {block.shape} does not match {len(block_ids)} ids x {dim}"
            )
        ids.extend(block_ids)
        blocks.append(block)
    vectors = np.concatenate(blocks) if blocks else np.zeros((0, dim), dtype=np.float32)
    return ids, vectors


def _read_parquets(files: Iterable[Path], columns: Sequence[str]) -> list[tuple]:
    rows: list[tuple] = []
    for path in files:
        df = pd.read_parquet(path)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows.extend(tuple(_clean(v) for v in r) for r in df[list(columns)].itertuples(index=False))
    return rows


@dataclass
class BuildData:
    judgments: list[dict[str, Any]]
    chunks: list[ChunkRecord]
    vector_ids: list[str]
    vectors: np.ndarray
    aliases: list[tuple]
    links: list[tuple]
    refs: list[tuple]

    @property
    def paths(self) -> set[str]:
        return {j["path"] for j in self.judgments}


def read_build(build_dir: Path | str, dim: int, years: Iterable[int] | None = None) -> BuildData:
    """Read the build files (every year, or only `years`); aliases are kept only for
    the judgments being loaded."""
    build = Path(build_dir)
    years = list(years) if years is not None else None
    jfiles = _year_files(build / "judgments", ".parquet", years)
    if not jfiles:
        raise ValueError(f"{build / 'judgments'}: no judgment files")
    judgments = [r for f in jfiles for r in judgment_rows(pd.read_parquet(f))]
    seen: set[str] = set()
    for j in judgments:
        if j["path"] in seen:
            raise ValueError(f"judgment {j['path']!r} appears twice in the build")
        seen.add(j["path"])
    chunks = read_chunks(build, years)
    orphans = sorted({c.judgment_id for c in chunks} - seen)
    if orphans:
        raise ValueError(f"chunks reference judgments not in the build: {orphans[:5]}")
    ids, vectors = read_vectors(build, dim, years)
    alias_file = build / "aliases.parquet"
    aliases = _read_parquets([alias_file], ALIAS_COLUMNS) if alias_file.exists() else []
    aliases = [a for a in aliases if a[2] in seen]
    links = _read_parquets(_year_files(build / "links", ".parquet", years), LINK_COLUMNS)
    refs = _read_parquets(_year_files(build / "refs", ".parquet", years), REF_COLUMNS)
    return BuildData(judgments, chunks, ids, vectors, aliases, links, refs)


# --- checks (pure) ------------------------------------------------------------------------


def check_ids(chunk_ids: Iterable[str], vector_ids: Iterable[str]) -> list[str]:
    """Failures when the vector IDs are not exactly the chunk IDs."""
    chunks, vectors = list(chunk_ids), list(vector_ids)
    failures: list[str] = []
    if len(set(vectors)) != len(vectors):
        failures.append("ids: duplicate vector ids")
    missing = set(chunks) - set(vectors)
    extra = set(vectors) - set(chunks)
    if missing:
        failures.append(f"ids: {len(missing)} chunks have no vector, e.g. {sorted(missing)[:3]}")
    if extra:
        failures.append(f"ids: {len(extra)} vectors have no chunk, e.g. {sorted(extra)[:3]}")
    return failures


def check_g6(metrics_path: Path, settings: Settings) -> list[str]:
    """The G6 automatic metrics from build/chunk_metrics.json."""
    if not metrics_path.exists():
        return [f"g6: {metrics_path} not found"]
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for key, minimum in (
        ("paragraph_integrity", settings.g6.min_paragraph_integrity),
        ("size_compliance", settings.g6.min_size_compliance),
    ):
        value = m.get(key)
        if not isinstance(value, int | float):
            failures.append(f"g6: {key} missing")
        elif value < minimum:
            failures.append(f"g6: {key} {value} < {minimum}")
    over = m.get("over_model_limit")
    if not isinstance(over, int):
        failures.append("g6: over_model_limit missing")
    elif over > 0:
        failures.append(f"g6: {over} chunks over the model limit")
    return failures


@dataclass
class SmokeOutcome:
    id: str
    route_ok: bool
    hit: bool
    route: str
    found_paths: list[str] = field(default_factory=list)


def read_smoke(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def smoke_check(
    questions: Sequence[dict[str, Any]],
    engine: SearchEngine,
    embedder: QueryEmbedder,
    reranker: RerankService,
    settings: Settings,
) -> tuple[list[str], list[SmokeOutcome]]:
    """Route, retrieve and rerank each smoke question; no LLM, no refusal threshold.

    A citation question that resolves counts the resolved judgment; one that
    does not resolve falls back to hybrid, as the pipeline does. A question
    with `expected_path` null (unanswerable) counts as a hit: there is no
    judgment to find, and the refusal threshold is not part of this check.
    """
    outcomes: list[SmokeOutcome] = []
    for q in questions:
        question = q["question"]
        filters = Filters(**q["filters"]) if q.get("filters") else None
        r = route(question)
        found: list[str] = []
        if r is Route.CITATION_LOOKUP:
            found = [p for c in lookup_canonicals(question) if (p := engine.resolve_alias(c))]
        if not found:
            res = hybrid_retrieve(question, filters, engine, embedder, reranker, settings)
            chunks = engine.get_chunks([h.chunk_id for h in res.reranked])
            found = list(dict.fromkeys(c.judgment_id for c in chunks))
        expected = q.get("expected_path")
        outcomes.append(
            SmokeOutcome(
                id=str(q.get("id", question)),
                route_ok=r.value == q["expected_route"],
                hit=expected is None or expected in found,
                route=r.value,
                found_paths=found,
            )
        )
    failures: list[str] = []
    bad_routes = [o.id for o in outcomes if not o.route_ok]
    if bad_routes:
        failures.append(f"smoke: route mismatch for {bad_routes}")
    hits = sum(o.hit for o in outcomes)
    if hits < settings.load.smoke_min_hits:
        misses = [o.id for o in outcomes if not o.hit]
        failures.append(f"smoke: {hits} hits < {settings.load.smoke_min_hits} (missed {misses})")
    return failures, outcomes


# --- database -----------------------------------------------------------------------------


def _vector_text(v: np.ndarray) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _copy(cur: Any, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    n = 0
    cols = ", ".join(columns)
    with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)
            n += 1
    return n


def insert_version_row(conn: Any, version: str, settings: Settings) -> None:
    conn.execute(
        "INSERT INTO index_versions (id, status, clean_text_version, chunker_version,"
        " config_version, embed_model, embed_revision, reranker_model, threshold_qwen,"
        " threshold_cohere, snapshot_date)"
        " VALUES (%s, 'building', %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            version,
            settings.versions.clean_text,
            settings.versions.chunker,
            settings.versions.config,
            settings.embed.model,
            settings.embed.revision or None,
            settings.rerank.model,
            settings.refusal.threshold_qwen,
            settings.refusal.threshold_cohere,
            settings.snapshot_date,
        ),
    )


def chunk_row(c: ChunkRecord) -> tuple:
    d = c.model_dump(mode="json")
    return tuple(d[k] for k in CHUNK_LOAD_COLUMNS)


def copy_build(conn: Any, version: str, data: BuildData) -> dict[str, int]:
    """COPY every table in the design's order. Returns row counts."""
    from psycopg.types.json import Jsonb

    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        counts["judgments"] = _copy(
            cur,
            "judgments",
            ("index_version", *JUDGMENT_LOAD_COLUMNS),
            (
                (
                    version,
                    *(
                        Jsonb(j[c]) if c == "position_map" and j[c] is not None else j[c]
                        for c in JUDGMENT_LOAD_COLUMNS
                    ),
                )
                for j in data.judgments
            ),
        )
        counts["chunks"] = _copy(
            cur,
            "chunks",
            ("index_version", *CHUNK_LOAD_COLUMNS),
            ((version, *chunk_row(c)) for c in data.chunks),
        )
        counts["chunk_vectors"] = _copy(
            cur,
            "chunk_vectors",
            ("index_version", "chunk_id", "embedding"),
            (
                (version, cid, _vector_text(vec))
                for cid, vec in zip(data.vector_ids, data.vectors, strict=True)
            ),
        )
        counts["aliases"] = _copy(
            cur, "aliases", ("index_version", *ALIAS_COLUMNS), ((version, *a) for a in data.aliases)
        )
        counts["links"] = _copy(
            cur, "links", ("index_version", *LINK_COLUMNS), ((version, *lk) for lk in data.links)
        )
        counts["refs"] = _copy(
            cur, "refs", ("index_version", *REF_COLUMNS), ((version, *r) for r in data.refs)
        )
    return counts


def db_id_check(conn: Any, version: str) -> list[str]:
    """Vector IDs equal chunk IDs, as loaded."""
    row = conn.execute(
        "SELECT"
        " (SELECT count(*) FROM chunks c WHERE c.index_version = %(v)s AND NOT EXISTS"
        "   (SELECT 1 FROM chunk_vectors v WHERE v.index_version = %(v)s"
        "    AND v.chunk_id = c.chunk_id)),"
        " (SELECT count(*) FROM chunk_vectors v WHERE v.index_version = %(v)s AND NOT EXISTS"
        "   (SELECT 1 FROM chunks c WHERE c.index_version = %(v)s AND c.chunk_id = v.chunk_id))",
        {"v": version},
    ).fetchone()
    failures = []
    if row[0]:
        failures.append(f"ids: {row[0]} loaded chunks have no vector")
    if row[1]:
        failures.append(f"ids: {row[1]} loaded vectors have no chunk")
    return failures


@dataclass
class LoadReport:
    version: str
    status: str
    counts: dict[str, int]
    failures: list[str]
    smoke: list[SmokeOutcome]


def load(
    version: str,
    build_dir: Path | str,
    *,
    settings: Settings,
    conn: Any,
    embedder: QueryEmbedder,
    reranker: RerankService,
    smoke_path: Path | str = Path("eval") / "datasets" / "phase1_smoke.jsonl",
    engine_factory: Callable[[str], SearchEngine] | None = None,
    years: Iterable[int] | None = None,
) -> LoadReport:
    """Load `build_dir` as `version`; raise LoadCheckFailed (exit 5) if a check fails.

    `conn` is an autocommit psycopg connection with pgvector registered.
    `engine_factory(version)` builds the engine for the checks; it defaults to
    `PostgresSearch(index_version=version, conn=conn)`.
    """
    build = Path(build_dir)
    data = read_build(build, settings.embed.dim, years)
    pre = check_ids([c.chunk_id for c in data.chunks], data.vector_ids)
    if pre:
        raise ValueError("build is inconsistent, nothing loaded: " + "; ".join(pre))

    with conn.transaction():
        insert_version_row(conn, version, settings)
    with conn.transaction():
        counts = copy_build(conn, version, data)
    conn.execute("ANALYZE chunks")
    conn.execute("ANALYZE chunk_vectors")
    log.info("loaded %s: %s", version, counts)

    if engine_factory is None:
        from nyaya.search.postgres import PostgresSearch
        from nyaya.zone.zoner import act_normalizer

        def engine_factory(v: str) -> SearchEngine:
            return PostgresSearch(
                index_version=v,
                settings=settings,
                conn=conn,
                act_normalizer=act_normalizer(settings),
            )

    failures = db_id_check(conn, version)
    failures += check_g6(build / "chunk_metrics.json", settings)
    smoke: list[SmokeOutcome] = []
    smoke_file = Path(smoke_path)
    if smoke_file.exists():
        smoke_failures, smoke = smoke_check(
            read_smoke(smoke_file), engine_factory(version), embedder, reranker, settings
        )
        failures += smoke_failures
    else:
        failures.append(f"smoke: {smoke_file} not found")

    if failures:
        raise LoadCheckFailed(version, failures)
    with conn.transaction():
        conn.execute(
            "UPDATE index_versions SET status = 'ready' WHERE id = %s AND status = 'building'",
            (version,),
        )
    return LoadReport(version, "ready", counts, [], smoke)


def activate(conn: Any, version: str) -> dict[str, Any]:
    """Make `version` active; retire the old active version; drop older retired ones.

    Accepts a `ready` version, or a `retired` one (rolling back to the previous
    version, which stays possible until the next activation succeeds).
    """
    with conn.transaction():
        row = conn.execute(
            "SELECT status FROM index_versions WHERE id = %s FOR UPDATE", (version,)
        ).fetchone()
        if row is None:
            raise LookupError(f"index version {version!r} not found")
        if row[0] not in ("ready", "retired"):
            raise ValueError(f"index version {version!r} is {row[0]}; only ready can be activated")
        previous = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM index_versions WHERE status = 'active' AND id <> %s FOR UPDATE",
                (version,),
            ).fetchall()
        ]
        conn.execute("UPDATE index_versions SET status = 'retired' WHERE id = ANY(%s)", (previous,))
        conn.execute("UPDATE index_versions SET status = 'active' WHERE id = %s", (version,))
        deleted: list[str] = []
        if previous:
            deleted = [
                r[0]
                for r in conn.execute(
                    "DELETE FROM index_versions WHERE status = 'retired' AND NOT (id = ANY(%s))"
                    " RETURNING id",
                    (previous,),
                ).fetchall()
            ]
    return {"active": version, "retired": previous, "deleted": deleted}
