"""Link expansion (design: "Link expansion", file 32).

A retrieved Held (editorial) chunk adds the judgment chunks it links to, each
ranked at its parent's fused rank plus `retrieval.expand_rank_offset`. The
list is then cut to `retrieval.rerank_candidates`; that cap is the only limit
on reranker input.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nyaya.contracts import SearchHit


class LinkSource(Protocol):
    def links_from(self, chunk_ids: Sequence[str]) -> dict[str, list[str]]: ...


def expand(
    hits: Sequence[SearchHit], engine: LinkSource, offset: float, cap: int
) -> list[SearchHit]:
    """Add linked chunks, re-sort by fused rank, and keep the first `cap`.

    A linked chunk that is already in the list keeps the better (lower) of its
    own rank and parent rank + offset. A chunk linked from two parents takes
    the better parent. Order ties keep the input order, and added children
    follow in link order.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h.fused_rank)
    by_id: dict[str, SearchHit] = {}
    for h in ordered:
        by_id.setdefault(h.chunk_id, h)
    links = engine.links_from(list(by_id))
    for parent in list(by_id.values()):
        for child_id in links.get(parent.chunk_id, []):
            if child_id == parent.chunk_id:
                continue
            rank = parent.fused_rank + offset
            current = by_id.get(child_id)
            if current is None:
                by_id[child_id] = SearchHit(chunk_id=child_id, fused_rank=rank)
            elif rank < current.fused_rank:
                by_id[child_id] = current.model_copy(update={"fused_rank": rank})
    # dict order = input order, then children in link order: a stable sort keeps it for ties
    return sorted(by_id.values(), key=lambda h: h.fused_rank)[:cap]
