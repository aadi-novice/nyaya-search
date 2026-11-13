"""Connecting the evaluation code to the real search stack (imported lazily).

The search engine, fusion, expansion and encoders are built in other Phase 1
modules. Only the function names the design gives are relied on:

*   `nyaya.search.postgres.PostgresSearch(index_version=..., schema=...,
    act_normalizer=...)`;
*   `nyaya.search.fusion.rrf(lists, k)`;
*   `nyaya.search.expand.expand(hits, engine, offset, cap)`, bound here to
    `retrieval.expand_rank_offset` and `retrieval.rerank_candidates`;
*   `nyaya.models.encoders.Embedder`, `QwenReranker` and
    `CohereFallbackReranker`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from nyaya.config import Settings
from nyaya.contracts import ChunkRecord
from nyaya.eval.retrieval import EngineRetriever
from nyaya.models.encoders import document_text


def _method(obj: Any, names: Sequence[str]) -> Callable[..., Any]:
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn
    raise TypeError(f"{type(obj).__name__} has none of the methods {list(names)}")


def embed_query_fn(embedder: Any) -> Callable[[str], Sequence[float]]:
    """A one-question embedding function from an `Embedder`."""
    single = None
    for n in ("embed_query", "encode_query"):
        if callable(getattr(embedder, n, None)):
            single = getattr(embedder, n)
            break
    if single is not None:
        return lambda q: list(single(q))
    batch = _method(embedder, ("embed_queries", "encode_queries"))
    return lambda q: list(batch([q])[0])


def rerank_fn(reranker: Any) -> Callable[[str, list[ChunkRecord]], list[float]]:
    """One score per chunk, in chunk order, given `header`, a blank line, then `text`.

    Uses `scores(query, docs)` when the reranker has it (QwenReranker), else
    the results of `rerank(query, docs, top_n)` put back in input order.
    """
    fn = _method(reranker, ("scores", "rerank"))
    by_index = getattr(fn, "__name__", "") == "rerank"

    def run(question: str, chunks: list[ChunkRecord]) -> list[float]:
        docs = [document_text(c.header, c.text) for c in chunks]
        if not by_index:
            return [float(s) for s in fn(question, docs)]
        out = [0.0] * len(docs)
        for r in fn(question, docs, len(docs)):
            out[r.index] = float(r.score)
        return out

    return run


def active_version(conn: Any) -> str:
    row = conn.execute("SELECT id FROM index_versions WHERE status = 'active'").fetchone()
    if row is None:
        raise RuntimeError("no active index version (run `nyaya-build activate`)")
    return row[0]


def make_encoders(settings: Settings, reranker: str = "qwen") -> tuple[Any, Any]:
    from nyaya.models import encoders

    if reranker not in ("qwen", "cohere"):
        raise ValueError(f"unknown reranker {reranker!r}")
    embedder = encoders.Embedder(settings)
    if reranker == "qwen":
        rr: Any = encoders.QwenReranker(settings)
    else:
        rr = encoders.CohereFallbackReranker.from_settings(settings)
    return embedder, rr


def make_engine_retriever(
    settings: Settings,
    *,
    index_version: str | None,
    schema: str | None = None,
    embedder: Any,
    reranker: Any | None,
) -> tuple[EngineRetriever, Any]:
    from nyaya.search.expand import expand
    from nyaya.search.fusion import rrf
    from nyaya.search.postgres import PostgresSearch
    from nyaya.zone.zoner import act_normalizer

    engine = PostgresSearch(
        index_version=index_version,
        schema=schema,
        settings=settings,
        act_normalizer=act_normalizer(settings),
    )
    r = settings.retrieval

    def bound_expand(hits: Any, eng: Any) -> Any:
        return expand(hits, eng, r.expand_rank_offset, r.rerank_candidates)

    retriever = EngineRetriever(
        engine=engine,
        embed_query=embed_query_fn(embedder),
        fuse=rrf,
        rerank=rerank_fn(reranker) if reranker is not None else None,
        expand=bound_expand,
        keyword_k=r.keyword_k,
        vector_k=r.vector_k,
        rrf_k=r.rrf_k,
        rerank_candidates=r.rerank_candidates,
    )
    return retriever, engine
