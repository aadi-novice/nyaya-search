"""Hybrid retrieval and the refusal-gate inputs (design: "Hybrid retrieval",
"Link expansion", "Reranking", "Refusal gate").

keyword + vector search -> RRF -> link expansion (cap `rerank_candidates`)
-> rerank (primary, fallback on failure) -> top `final_k`.

Shared by the query pipeline and the loader's smoke check, so both run the
same steps. No LLM is involved.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from nyaya.config import Settings
from nyaya.contracts import Filters, SearchHit
from nyaya.models.encoders import RerankOutcome, document_text
from nyaya.search.expand import expand
from nyaya.search.fusion import rrf
from nyaya.search.interface import SearchEngine


class QueryEmbedder(Protocol):
    def embed_query(self, question: str) -> Any: ...


class RerankService(Protocol):
    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> RerankOutcome: ...


@dataclass
class HybridResult:
    keyword_hits: list[SearchHit]
    vector_hits: list[SearchHit]
    fused: list[SearchHit]
    candidates: list[SearchHit]  # after expansion and the cap: the reranker input
    reranked: list[SearchHit]  # top final_k, with rerank_score, best first
    rerank: RerankOutcome | None  # None when there were no candidates
    rows_returned: dict[str, dict[str, int]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float | None:
        return self.reranked[0].rerank_score if self.reranked else None

    @property
    def reranker_kind(self) -> str | None:
        return self.rerank.kind if self.rerank else None


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def hybrid_retrieve(
    question: str,
    filters: Filters | None,
    engine: SearchEngine,
    embedder: QueryEmbedder,
    reranker: RerankService,
    settings: Settings,
    *,
    exact: bool = False,
) -> HybridResult:
    r = settings.retrieval
    timings: dict[str, float] = {}

    t = time.perf_counter()
    kw = engine.keyword(question, filters, r.keyword_k)
    timings["keyword"] = _ms(t)

    t = time.perf_counter()
    qvec = embedder.embed_query(question)
    timings["embed_query"] = _ms(t)

    t = time.perf_counter()
    vec = engine.vector(qvec, filters, r.vector_k, exact)
    timings["vector"] = _ms(t)

    t = time.perf_counter()
    fused = rrf([kw, vec], r.rrf_k)
    candidates = expand(fused, engine, r.expand_rank_offset, r.rerank_candidates)
    timings["fuse_expand"] = _ms(t)

    rows = {
        "keyword": {"requested": r.keyword_k, "returned": len(kw)},
        "vector": {"requested": r.vector_k, "returned": len(vec)},
    }
    if not candidates:
        return HybridResult(kw, vec, fused, [], [], None, rows, timings)

    t = time.perf_counter()
    chunks = {c.chunk_id: c for c in engine.get_chunks([h.chunk_id for h in candidates])}
    present = [h for h in candidates if h.chunk_id in chunks]
    docs = [document_text(chunks[h.chunk_id].header, chunks[h.chunk_id].text) for h in present]
    timings["get_chunks"] = _ms(t)

    t = time.perf_counter()
    outcome = reranker.rerank(question, docs, r.final_k)
    timings["rerank"] = _ms(t)

    reranked = [
        present[res.index].model_copy(update={"rerank_score": res.score})
        for res in outcome.results[: r.final_k]
    ]
    return HybridResult(kw, vec, fused, candidates, reranked, outcome, rows, timings)


def refusal_threshold(kind: str, thresholds: dict[str, float | None]) -> float:
    """The threshold for the reranker that ran, from the active index_versions row."""
    if kind not in ("qwen", "cohere"):
        raise ValueError(f"unknown reranker kind {kind!r}")
    value = thresholds.get(kind)
    if value is None:
        raise ValueError(f"index version has no threshold_{kind}")
    return float(value)


def should_refuse(top_score: float | None, kind: str | None, thresholds: dict[str, Any]) -> bool:
    """Refuse when the top rerank score is below the threshold of the reranker that ran.

    No candidates (top_score None) also refuses; the pipeline reports
    `no_match_with_filters` instead when filters emptied the candidate list.
    """
    if top_score is None or kind is None:
        return True
    return top_score < refusal_threshold(kind, thresholds)
