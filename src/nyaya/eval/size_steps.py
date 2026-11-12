"""`nyaya-eval size-steps`: does search hold as the index grows? (G4)

Design: Goal G4, "Evaluation harness" (G4 size steps), file 42.

*   Each step is built in its own temporary schema `g4_step_<n>`, one at a
    time, and dropped after its run. Tables are created `LIKE public.<t>
    INCLUDING DEFAULTS INCLUDING GENERATED` (no foreign keys), and rows are
    copied from the source (active) version for the step's paths, with
    `index_version` set to the schema name. No re-embedding is needed.
*   Steps are nested: each contains every gold judgment, plus other judgments
    added in a fixed order from `eval.seed`, until the step's chunk target
    is reached. The last step may be "full" (every judgment).
*   Each step gets its own HNSW and GIN indexes, and is queried through an
    engine made for that schema (`PostgresSearch(index_version=name,
    schema=name)`). Steps never pass through `index_versions`.
*   **Pass rule:** for the G3 numbers (judgment-level recall@50 of hybrid
    search before reranking, and recall@5 after reranking), the upper bound of
    the 95% paired-bootstrap interval of the drop from the smallest to the
    largest step is at most `g4.max_drop_points`.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nyaya.eval.datasets import DevItem
from nyaya.eval.retrieval import METHODS, RetrievalReport, Retriever, evaluate
from nyaya.eval.stats import paired_bootstrap

STEP_TABLES = ("judgments", "chunks", "chunk_vectors", "links", "refs")
_SCHEMA_RE = re.compile(r"^g4_step_\d{1,3}$")

# (method, level, metric) pairs that G4 gates; everything else is reported only.
GATED = (("hybrid", "judgment", "recall@50"), ("hybrid_rerank", "judgment", "recall@5"))


def step_name(n: int) -> str:
    return f"g4_step_{n}"


def _check_name(name: str) -> str:
    if not _SCHEMA_RE.match(name):
        raise ValueError(f"not a size-step schema name: {name!r}")
    return name


def nested_steps(
    chunk_counts: Mapping[str, int],
    gold_paths: Iterable[str],
    targets: Sequence[int | None],
    seed: int,
) -> list[list[str]]:
    """Paths for each step: gold first, then others in one seeded order.

    `targets` are chunk counts in increasing order; None means every path.
    Each step is a prefix of the same order, so steps are nested.
    """
    gold = sorted(p for p in set(gold_paths) if p in chunk_counts)
    others = sorted(p for p in chunk_counts if p not in set(gold))
    random.Random(seed).shuffle(others)
    order = gold + others
    steps: list[list[str]] = []
    prev_target = -1
    for t in targets:
        if t is not None and t <= prev_target:
            raise ValueError("step targets must increase")
        prev_target = t if t is not None else prev_target
        if t is None:
            steps.append(list(order))
            continue
        total, cut = 0, 0
        for p in order:
            if total >= t and cut >= len(gold):
                break
            total += chunk_counts[p]
            cut += 1
        steps.append(order[:cut])
    return steps


def parse_targets(text: str) -> list[int | None]:
    """'10000,50000,full' -> [10000, 50000, None]."""
    out: list[int | None] = []
    for part in text.split(","):
        part = part.strip().lower()
        out.append(None if part in ("full", "all") else int(part.replace("_", "")))
    return out


# --- schemas ---------------------------------------------------------------------------


def _columns(conn: Any, table: str) -> list[str]:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s AND is_generated = 'NEVER' "
        "ORDER BY ordinal_position",
        (table,),
    ).fetchall()
    return [r[0] for r in rows]


_ROW_FILTERS = {
    "judgments": "path = ANY(%(paths)s)",
    "chunks": "judgment_id = ANY(%(paths)s)",
    "chunk_vectors": "chunk_id IN (SELECT chunk_id FROM {s}.chunks)",
    "links": (
        "from_chunk_id IN (SELECT chunk_id FROM {s}.chunks) "
        "AND to_chunk_id IN (SELECT chunk_id FROM {s}.chunks)"
    ),
    "refs": "path = ANY(%(paths)s)",
}

_INDEXES = (
    "CREATE INDEX ON {s}.chunk_vectors USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX ON {s}.chunks USING gin (tsv)",
    "CREATE UNIQUE INDEX ON {s}.chunks (index_version, chunk_id)",
    "CREATE INDEX ON {s}.judgments (index_version, path)",
    "CREATE INDEX ON {s}.judgments (index_version, decision_year)",
    "CREATE INDEX ON {s}.judgments USING gin (judges_norm)",
    "CREATE INDEX ON {s}.refs (index_version, kind, value_norm)",
)


def build_step_schema(conn: Any, name: str, source_version: str, paths: Sequence[str]) -> None:
    """Create schema `name` holding `paths` copied from `source_version`."""
    s = _check_name(name)
    with conn.transaction():
        conn.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        conn.execute(f"CREATE SCHEMA {s}")
        for t in STEP_TABLES:
            conn.execute(
                f"CREATE TABLE {s}.{t} (LIKE public.{t} INCLUDING DEFAULTS INCLUDING GENERATED)"
            )
            cols = _columns(conn, t)
            if "index_version" not in cols:
                raise RuntimeError(f"public.{t} has no index_version column")
            col_list = ", ".join(cols)
            select = ", ".join("%(name)s" if c == "index_version" else c for c in cols)
            where = _ROW_FILTERS[t].format(s=s)
            conn.execute(
                f"INSERT INTO {s}.{t} ({col_list}) SELECT {select} FROM public.{t} "
                f"WHERE index_version = %(version)s AND {where}",
                {"name": s, "version": source_version, "paths": list(paths)},
            )
        for ddl in _INDEXES:
            conn.execute(ddl.format(s=s))
        for t in STEP_TABLES:
            conn.execute(f"ANALYZE {s}.{t}")


def drop_step_schema(conn: Any, name: str) -> None:
    s = _check_name(name)
    with conn.transaction():
        conn.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")


def chunk_counts(conn: Any, version: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT judgment_id, count(*) FROM chunks WHERE index_version = %s GROUP BY judgment_id",
        (version,),
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


# --- the run ---------------------------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    judgments: int
    chunks: int
    report: RetrievalReport


@dataclass
class SizeStepsResult:
    steps: list[StepResult]
    drops: dict[str, dict[str, Any]] = field(default_factory=dict)
    passed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "name": s.name,
                    "judgments": s.judgments,
                    "chunks": s.chunks,
                    "summary": s.report.summary(),
                }
                for s in self.steps
            ],
            "drops": self.drops,
            "passed": self.passed,
        }


MakeRetriever = Callable[[str], tuple[Retriever, Any]]


def run_size_steps(
    conn: Any,
    items: Sequence[DevItem],
    *,
    source_version: str,
    targets: Sequence[int | None],
    make_retriever: MakeRetriever,
    seed: int,
    bootstrap_samples: int,
    max_drop_points: float,
    methods: Sequence[str] = METHODS,
    keep_schemas: bool = False,
) -> SizeStepsResult:
    """Build, evaluate and drop each step in turn, then compare first and last."""
    answerable = [i for i in items if i.type == "answerable"]
    counts = chunk_counts(conn, source_version)
    gold = {p for i in answerable for p in i.gold_paths}
    missing = sorted(gold - counts.keys())
    if missing:
        raise ValueError(f"gold judgments not in {source_version}: {missing[:5]}")
    plan = nested_steps(counts, gold, targets, seed)
    steps: list[StepResult] = []
    for n, paths in enumerate(plan, start=1):
        name = step_name(n)
        build_step_schema(conn, name, source_version, paths)
        try:
            retriever, engine = make_retriever(name)
            report = evaluate(answerable, retriever, engine, methods)
        finally:
            if not keep_schemas:
                drop_step_schema(conn, name)
        steps.append(StepResult(name, len(paths), sum(counts[p] for p in paths), report))
    return compare_steps(
        steps, seed=seed, samples=bootstrap_samples, max_drop_points=max_drop_points
    )


def compare_steps(
    steps: list[StepResult], *, seed: int, samples: int, max_drop_points: float
) -> SizeStepsResult:
    """Drops (in points) from the first step to each later step, with the G4 verdict."""
    result = SizeStepsResult(steps=steps)
    if len(steps) < 2:
        return result
    first = steps[0].report
    passed = True
    gated_seen = False
    for later in steps[1:]:
        per_step: dict[str, Any] = {}
        for method, levels in first.per_question.items():
            for level, metrics in levels.items():
                for metric in metrics:
                    name = "mrr" if metric == "rr" else metric
                    a = first.vector(method, level, name)
                    b = later.report.vector(method, level, name)
                    ci = paired_bootstrap(a, b, samples=samples, seed=seed).scaled(100.0)
                    key = f"{method}/{level}/{name}"
                    gated = (method, level, name) in GATED
                    entry: dict[str, Any] = {**ci.as_dict(), "gated": gated}
                    if gated:
                        entry["passed"] = ci.high <= max_drop_points
                        if later is steps[-1]:
                            gated_seen = True
                            passed = passed and entry["passed"]
                    per_step[key] = entry
        result.drops[f"{steps[0].name}->{later.name}"] = per_step
    result.passed = passed and gated_seen  # no verdict without the gated methods
    return result
