"""Reciprocal rank fusion: rank arithmetic."""

import pytest

from nyaya.contracts import SearchHit
from nyaya.search.fusion import rrf, rrf_scores


def kw(*ids):
    return [SearchHit(chunk_id=c, keyword_rank=i, fused_rank=i) for i, c in enumerate(ids, 1)]


def vec(*ids):
    return [SearchHit(chunk_id=c, vector_rank=i, fused_rank=i) for i, c in enumerate(ids, 1)]


def test_scores_follow_formula():
    scores = rrf_scores([kw("a", "b"), vec("b", "c")], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_fused_order_and_ranks():
    fused = rrf([kw("a", "b"), vec("b", "c")], k=60)
    assert [h.chunk_id for h in fused] == ["b", "a", "c"]
    assert [h.fused_rank for h in fused] == [1.0, 2.0, 3.0]


def test_component_ranks_are_carried():
    fused = {h.chunk_id: h for h in rrf([kw("a", "b"), vec("b", "c")], k=60)}
    assert (fused["a"].keyword_rank, fused["a"].vector_rank) == (1, None)
    assert (fused["b"].keyword_rank, fused["b"].vector_rank) == (2, 1)
    assert (fused["c"].keyword_rank, fused["c"].vector_rank) == (None, 2)
    assert all(h.rerank_score is None for h in fused.values())


def test_ties_broken_by_best_rank_then_id():
    # "a" and "x" both sit at rank 1 in one list each: equal scores
    fused = rrf([kw("x", "b"), vec("a", "c")], k=60)
    assert [h.chunk_id for h in fused] == ["a", "x", "b", "c"]


def test_position_not_carried_rank_is_used():
    # the rank used is the position in the list, whatever fused_rank says
    odd = [SearchHit(chunk_id="z", keyword_rank=9, fused_rank=9)]
    assert rrf_scores([odd], k=60)["z"] == pytest.approx(1 / 61)


def test_duplicate_in_one_list_counts_once():
    dup = [SearchHit(chunk_id="a", fused_rank=1), SearchHit(chunk_id="a", fused_rank=2)]
    assert rrf_scores([dup], k=10)["a"] == pytest.approx(1 / 11)


def test_empty_inputs():
    assert rrf([], k=60) == []
    assert rrf([[], []], k=60) == []


def test_small_k_weights_top_more():
    scores = rrf_scores([kw("a", "b")], k=0)
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(0.5)


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        rrf([kw("a")], k=-1)
