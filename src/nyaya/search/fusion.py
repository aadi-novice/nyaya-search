"""Reciprocal rank fusion (design: "Hybrid retrieval", file 32).

score(d) = sum over lists of 1 / (k + rank), where rank is d's 1-based
position in that list. `k` comes from `retrieval.rrf_k`.
"""

from __future__ import annotations

from collections.abc import Sequence

from nyaya.contracts import SearchHit


def rrf_scores(lists: Sequence[Sequence[SearchHit]], k: int) -> dict[str, float]:
    """The RRF score of every chunk in `lists`. A chunk repeated in one list counts once."""
    if k < 0:
        raise ValueError(f"rrf k must be >= 0, got {k}")
    scores: dict[str, float] = {}
    for hits in lists:
        seen: set[str] = set()
        for pos, hit in enumerate(hits, 1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + pos)
    return scores


def rrf(lists: Sequence[Sequence[SearchHit]], k: int) -> list[SearchHit]:
    """Fuse ranked lists into one, best first, with `fused_rank` = 1, 2, 3, ...

    Each fused hit keeps the `keyword_rank` and `vector_rank` it had in the
    inputs. Equal scores are ordered by the best single-list position, then
    by chunk_id, so the result is deterministic.
    """
    scores = rrf_scores(lists, k)
    keyword_rank: dict[str, int] = {}
    vector_rank: dict[str, int] = {}
    best_pos: dict[str, int] = {}
    for hits in lists:
        for pos, hit in enumerate(hits, 1):
            cid = hit.chunk_id
            best_pos[cid] = min(best_pos.get(cid, pos), pos)
            if hit.keyword_rank is not None and cid not in keyword_rank:
                keyword_rank[cid] = hit.keyword_rank
            if hit.vector_rank is not None and cid not in vector_rank:
                vector_rank[cid] = hit.vector_rank
    order = sorted(scores, key=lambda c: (-scores[c], best_pos[c], c))
    return [
        SearchHit(
            chunk_id=cid,
            keyword_rank=keyword_rank.get(cid),
            vector_rank=vector_rank.get(cid),
            fused_rank=float(i),
        )
        for i, cid in enumerate(order, 1)
    ]
