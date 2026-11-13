"""`nyaya-eval retrieval`: recall@5, recall@50 and MRR (G3), per search method.

Design: Goals G3 and G4, "Evaluation harness", file 42.

*   **Methods:** exact vector, HNSW vector, keyword, hybrid (fused and
    link-expanded), and hybrid plus rerank.
*   **Scoring rules**, applied to every ranked list before it is scored:
    1.  a retrieved Held (editorial) chunk is replaced, in place, by the
        judgment chunks it links to (unless `editorial` is itself scored);
    2.  chunks of judgments in the item's `exclude` are removed;
    3.  only chunks whose zone is in `scored_zones` (default `judgment`) count;
    4.  repeated chunks are kept once, at their first position.
*   **Levels:** a judgment-level hit is any scored chunk of a gold judgment; a
    paragraph-level hit is a scored chunk of a gold judgment whose
    `para_labels` contain a gold label (as "N", "N.x", or the parent of "N.x").
*   **Ranks** are positions in the scored chunk list, so "top 5" means the
    first five scored chunks.

Search, fusion, expansion and reranking are injected (`EngineRetriever`), so
this module depends only on the contracts and the `SearchEngine` methods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from nyaya.contracts import ChunkRecord, Filters, SearchHit, Zone
from nyaya.eval.datasets import DevItem

METHODS = ("exact", "hnsw", "keyword", "hybrid", "hybrid_rerank")
LEVELS = ("judgment", "paragraph")
KS = (5, 50)


class Engine(Protocol):
    """The `SearchEngine` methods (file 30) used here."""

    def keyword(self, question: str, filters: Filters | None, k: int) -> list[SearchHit]: ...

    def vector(
        self, qvec: Sequence[float], filters: Filters | None, k: int, exact: bool
    ) -> list[SearchHit]: ...

    def get_chunks(self, ids: Sequence[str]) -> list[ChunkRecord]: ...

    def links_from(self, chunk_ids: Sequence[str]) -> dict[str, list[str]]: ...


Fuse = Callable[[list[list[SearchHit]], int], list[SearchHit]]
Expand = Callable[[list[SearchHit], Any], list[SearchHit]]
EmbedQuery = Callable[[str], Sequence[float]]
Rerank = Callable[[str, list[ChunkRecord]], list[float]]


@dataclass
class Ranking:
    chunk_ids: list[str]
    top_score: float | None = None  # the top rerank score (hybrid_rerank only)


class Retriever(Protocol):
    def run(self, method: str, question: str, filters: Filters | None) -> Ranking: ...


@dataclass
class EngineRetriever:
    """Runs each method against a `SearchEngine`, as the query pipeline does.

    `fuse` is `search.fusion.rrf`; `expand` is `search.expand.expand` (or None
    to skip link expansion); `rerank` returns one score per chunk.
    """

    engine: Engine
    embed_query: EmbedQuery
    fuse: Fuse
    rerank: Rerank | None = None
    expand: Expand | None = None
    keyword_k: int = 50
    vector_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 100
    _qvec_cache: dict[str, Sequence[float]] = field(default_factory=dict, repr=False)

    def _qvec(self, question: str) -> Sequence[float]:
        if question not in self._qvec_cache:
            self._qvec_cache[question] = self.embed_query(question)
        return self._qvec_cache[question]

    def _hybrid(self, question: str, filters: Filters | None) -> list[SearchHit]:
        kw = self.engine.keyword(question, filters, self.keyword_k)
        vec = self.engine.vector(self._qvec(question), filters, self.vector_k, False)
        fused = self.fuse([kw, vec], self.rrf_k)
        if self.expand is not None:
            fused = self.expand(fused, self.engine)
        return sorted(fused, key=lambda h: h.fused_rank)[: self.rerank_candidates]

    def run(self, method: str, question: str, filters: Filters | None) -> Ranking:
        if method == "keyword":
            hits = self.engine.keyword(question, filters, self.keyword_k)
        elif method in ("exact", "hnsw"):
            hits = self.engine.vector(
                self._qvec(question), filters, self.vector_k, method == "exact"
            )
        elif method == "hybrid":
            hits = self._hybrid(question, filters)
        elif method == "hybrid_rerank":
            if self.rerank is None:
                raise ValueError("hybrid_rerank needs a reranker")
            ids = [h.chunk_id for h in self._hybrid(question, filters)]
            chunks = _ordered_chunks(self.engine, ids)
            if not chunks:
                return Ranking([], None)
            scores = self.rerank(question, chunks)
            order = sorted(range(len(chunks)), key=lambda i: -scores[i])
            return Ranking([chunks[i].chunk_id for i in order], float(scores[order[0]]))
        else:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        return Ranking([h.chunk_id for h in hits])


def _ordered_chunks(engine: Engine, ids: Sequence[str]) -> list[ChunkRecord]:
    got = {c.chunk_id: c for c in engine.get_chunks(list(ids))}
    return [got[i] for i in ids if i in got]


# --- scoring ---------------------------------------------------------------------------


def label_matches(label: str, gold: str) -> bool:
    return label == gold or label.startswith(gold + ".") or gold.startswith(label + ".")


def scored_view(
    ranked_ids: Sequence[str],
    chunks: Mapping[str, ChunkRecord],
    links: Mapping[str, Sequence[str]],
    item: DevItem,
) -> list[ChunkRecord]:
    """Apply the scoring rules (module docstring) to one ranked list."""
    zones = set(item.scored_zones)
    exclude = set(item.exclude)
    replace_held = Zone.EDITORIAL not in zones
    out: list[ChunkRecord] = []
    seen: set[str] = set()

    def add(cid: str) -> None:
        c = chunks.get(cid)
        if c is None or cid in seen or c.judgment_id in exclude or c.zone not in zones:
            return
        seen.add(cid)
        out.append(c)

    for cid in ranked_ids:
        c = chunks.get(cid)
        if c is not None and replace_held and c.zone is Zone.EDITORIAL:
            for to in links.get(cid, ()):
                add(to)
        else:
            add(cid)
    return out


def first_hit_rank(view: Sequence[ChunkRecord], item: DevItem, level: str) -> int | None:
    """1-based rank of the first hit, or None."""
    gold = {g.path: g.para_labels for g in item.gold}
    for rank, c in enumerate(view, start=1):
        if c.judgment_id not in gold:
            continue
        if level == "judgment":
            return rank
        if any(label_matches(lab, g) for lab in c.para_labels for g in gold[c.judgment_id]):
            return rank
    return None


def question_metrics(rank: int | None, ks: Iterable[int] = KS) -> dict[str, float]:
    m = {f"recall@{k}": float(rank is not None and rank <= k) for k in ks}
    m["rr"] = 1.0 / rank if rank is not None else 0.0
    return m


@dataclass
class RetrievalReport:
    """Per-question metrics and their means, by method and level."""

    # per_question[method][level][metric] -> list aligned with `item_ids`
    item_ids: list[str]
    per_question: dict[str, dict[str, dict[str, list[float]]]]
    top_scores: dict[str, float | None] = field(default_factory=dict)  # hybrid_rerank

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for method, levels in self.per_question.items():
            out[method] = {}
            for level, metrics in levels.items():
                out[method][level] = {
                    ("mrr" if name == "rr" else name): (sum(v) / len(v) if v else 0.0)
                    for name, v in metrics.items()
                }
        return out

    def vector(self, method: str, level: str, metric: str) -> list[float]:
        return self.per_question[method][level]["rr" if metric == "mrr" else metric]


def evaluate(
    items: Iterable[DevItem],
    retriever: Retriever,
    engine: Engine,
    methods: Sequence[str] = METHODS,
    ks: Sequence[int] = KS,
) -> RetrievalReport:
    """Run every method for every answerable item and score it."""
    answerable = [i for i in items if i.type == "answerable"]
    report = RetrievalReport(
        item_ids=[i.id for i in answerable],
        per_question={
            m: {lv: {**{f"recall@{k}": [] for k in ks}, "rr": []} for lv in LEVELS} for m in methods
        },
    )
    for item in answerable:
        for method in methods:
            ranking = retriever.run(method, item.question, item.filters)
            if method == "hybrid_rerank":
                report.top_scores[item.id] = ranking.top_score
            view = scored_view(ranking.chunk_ids, *_lookups(engine, ranking.chunk_ids), item)
            for level in LEVELS:
                for name, v in question_metrics(first_hit_rank(view, item, level), ks).items():
                    report.per_question[method][level][name].append(v)
    return report


def _lookups(
    engine: Engine, ids: Sequence[str]
) -> tuple[dict[str, ChunkRecord], dict[str, list[str]]]:
    chunks = {c.chunk_id: c for c in engine.get_chunks(list(ids))}
    held = [cid for cid, c in chunks.items() if c.zone is Zone.EDITORIAL]
    links = engine.links_from(held) if held else {}
    missing = [t for tos in links.values() for t in tos if t not in chunks]
    if missing:
        chunks.update({c.chunk_id: c for c in engine.get_chunks(missing)})
    return chunks, links
