"""`nyaya-eval bm25-trigger`: is Postgres full-text search good enough?

Design: "Hybrid retrieval" (BM25 trigger), file 42.

Postgres keyword search (`SearchEngine.keyword`) and an in-memory BM25
(`bm25s`) are run on the same chunk texts and headers, for the answerable
development questions, and scored with the retrieval scoring rules. The
trigger **fires** when BM25's judgment-level recall@50 is higher by at least
`bm25_trigger.min_gap_points` and the 95% paired-bootstrap interval of the
gap excludes zero. Filters are not applied to either side (BM25 has none).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nyaya.contracts import Filters
from nyaya.eval.datasets import DevItem
from nyaya.eval.retrieval import Engine, Ranking, evaluate
from nyaya.eval.stats import paired_bootstrap


def _stemmer() -> Any:
    try:
        import Stemmer  # PyStemmer, optional: Postgres's 'english' config also stems

        return Stemmer.Stemmer("english")
    except ImportError:
        return None


class Bm25Index:
    """bm25s over `header + " " + text`, the same string Postgres indexes."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str]):
        import bm25s

        if len(ids) != len(texts) or not ids:
            raise ValueError("need one text per id, and at least one")
        self._bm25s = bm25s
        self.ids = list(ids)
        self.stemmer = _stemmer()
        tokens = bm25s.tokenize(
            list(texts), stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=False)

    def search(self, question: str, k: int) -> list[str]:
        q = self._bm25s.tokenize(
            [question], stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        if not q.vocab:
            return []
        k = min(k, len(self.ids))
        docs, scores = self.retriever.retrieve(q, k=k, show_progress=False)
        return [self.ids[int(d)] for d, s in zip(docs[0], scores[0], strict=True) if s > 0]


@dataclass
class TriggerRetriever:
    engine: Engine
    index: Bm25Index
    k: int = 50

    def run(self, method: str, question: str, filters: Filters | None) -> Ranking:
        if method == "keyword":
            return Ranking([h.chunk_id for h in self.engine.keyword(question, None, self.k)])
        if method == "bm25":
            return Ranking(self.index.search(question, self.k))
        raise ValueError(f"unknown method {method!r}")


def load_corpus(conn: Any, index_version: str) -> tuple[list[str], list[str]]:
    rows = conn.execute(
        "SELECT chunk_id, header || ' ' || text FROM chunks "
        "WHERE index_version = %s ORDER BY chunk_id",
        (index_version,),
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def run_bm25_trigger(
    items: Sequence[DevItem],
    engine: Engine,
    index: Bm25Index,
    *,
    k: int,
    samples: int,
    seed: int,
    min_gap_points: float,
) -> dict[str, Any]:
    report = evaluate(items, TriggerRetriever(engine, index, k), engine, ["keyword", "bm25"])
    if not report.item_ids:
        raise ValueError("no answerable questions")
    bm = report.vector("bm25", "judgment", "recall@50")
    pg = report.vector("keyword", "judgment", "recall@50")
    gap = paired_bootstrap(bm, pg, samples=samples, seed=seed).scaled(100.0)
    fires = gap.mean >= min_gap_points and gap.low > 0.0
    return {
        "summary": report.summary(),
        "gap_points": gap.as_dict(),
        "min_gap_points": min_gap_points,
        "fires": fires,
        "stemmer": index.stemmer is not None,
        "questions": len(report.item_ids),
    }
