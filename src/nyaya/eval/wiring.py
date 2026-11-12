"""Connecting the evaluation code to the real search stack (imported lazily).

The search engine, fusion, expansion and encoders are built in other Phase 1
modules. Only the function names the design gives are relied on:

*   `nyaya.search.postgres.PostgresSearch(index_version=..., schema=...)`;
*   `nyaya.search.fusion.rrf(lists, k)`;
*   `nyaya.search.expand.expand(hits, engine)`;
*   `nyaya.models.encoders.Embedder`, `Reranker` and `FallbackReranker`.

The encoder method names are not fixed by the design, so the adapters below
accept the likely spellings and fail with a clear message otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from nyaya.config import Settings
from nyaya.contracts import ChunkRecord
from nyaya.eval.retrieval import EngineRetriever


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
    """One score per chunk from a `Reranker`, given `header`, a blank line, then `text`."""
    fn = _method(reranker, ("score", "rerank", "scores"))

    def run(question: str, chunks: list[ChunkRecord]) -> list[float]:
        out = fn(question, [f"{c.header}\n\n{c.text}" for c in chunks])
        return [float(s) for s in out]

    return run


def active_version(conn: Any) -> str:
    row = conn.execute("SELECT id FROM index_versions WHERE status = 'active'").fetchone()
    if row is None:
        raise RuntimeError("no active index version (run `nyaya-build activate`)")
    return row[0]


def make_encoders(settings: Settings, reranker: str = "qwen") -> tuple[Any, Any]:
    from nyaya.models import encoders

    embedder = encoders.Embedder(settings)
    if reranker == "qwen":
        rr = encoders.Reranker(settings)
    elif reranker == "cohere":
        rr = encoders.FallbackReranker(settings)
    else:
        raise ValueError(f"unknown reranker {reranker!r}")
    return embedder, rr


def make_engine_retriever(
    settings: Settings,
    *,
    index_version: str | None,
    schema: str | None = None,
    embedder: Any,
    reranker: Any | None,
) -> tuple[EngineRetriever, Any]:
    from nyaya.search.fusion import rrf
    from nyaya.search.postgres import PostgresSearch

    try:
        from nyaya.search.expand import expand
    except ImportError:  # pragma: no cover - expansion not built yet
        expand = None
    engine = PostgresSearch(index_version=index_version, schema=schema)
    r = settings.retrieval
    retriever = EngineRetriever(
        engine=engine,
        embed_query=embed_query_fn(embedder),
        fuse=rrf,
        rerank=rerank_fn(reranker) if reranker is not None else None,
        expand=expand,
        keyword_k=r.keyword_k,
        vector_k=r.vector_k,
        rrf_k=r.rrf_k,
        rerank_candidates=r.rerank_candidates,
    )
    return retriever, engine
