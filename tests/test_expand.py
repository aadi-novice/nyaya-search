"""Link expansion: rank arithmetic and the reranker-input cap."""

import pytest

from nyaya.contracts import SearchHit
from nyaya.search.expand import expand


class FakeLinks:
    def __init__(self, links):
        self.links = links
        self.calls = []

    def links_from(self, chunk_ids):
        self.calls.append(list(chunk_ids))
        return {c: self.links[c] for c in chunk_ids if c in self.links}


def hits(*ids):
    return [SearchHit(chunk_id=c, fused_rank=float(i)) for i, c in enumerate(ids, 1)]


def test_children_sit_at_parent_rank_plus_offset():
    engine = FakeLinks({"p#editorial#001": ["p#judgment#003", "p#judgment#004"]})
    out = expand(hits("a", "p#editorial#001", "b"), engine, offset=0.5, cap=100)
    assert [(h.chunk_id, h.fused_rank) for h in out] == [
        ("a", 1.0),
        ("p#editorial#001", 2.0),
        ("p#judgment#003", 2.5),
        ("p#judgment#004", 2.5),
        ("b", 3.0),
    ]
    assert engine.calls == [["a", "p#editorial#001", "b"]]


def test_child_already_present_keeps_better_rank():
    engine = FakeLinks({"h": ["a", "z"]})
    out = expand(hits("a", "b", "h", "z"), engine, offset=0.5, cap=100)
    ranks = {h.chunk_id: h.fused_rank for h in out}
    assert ranks["a"] == 1.0  # already better than 3.5
    assert ranks["z"] == 3.5  # improved from 4.0
    assert len(out) == 4
    assert [h.chunk_id for h in out] == ["a", "b", "h", "z"]


def test_child_present_keeps_its_component_ranks():
    base = [
        SearchHit(chunk_id="h", keyword_rank=1, fused_rank=1.0),
        SearchHit(chunk_id="c", vector_rank=7, fused_rank=2.0),
    ]
    out = expand(base, FakeLinks({"h": ["c"]}), offset=0.5, cap=100)
    c = next(h for h in out if h.chunk_id == "c")
    assert c.fused_rank == 1.5 and c.vector_rank == 7


def test_child_of_two_parents_takes_the_better_one():
    engine = FakeLinks({"h1": ["c"], "h2": ["c"]})
    out = expand(hits("h1", "x", "h2"), engine, offset=0.5, cap=100)
    assert [(h.chunk_id, h.fused_rank) for h in out] == [
        ("h1", 1.0),
        ("c", 1.5),
        ("x", 2.0),
        ("h2", 3.0),
    ]


def test_cap_applies_after_expansion():
    engine = FakeLinks({"h": ["c1", "c2"]})
    out = expand(hits("h", "a", "b"), engine, offset=0.5, cap=3)
    assert [h.chunk_id for h in out] == ["h", "c1", "c2"]


def test_unsorted_input_is_ordered_by_rank():
    base = [SearchHit(chunk_id="b", fused_rank=2.0), SearchHit(chunk_id="a", fused_rank=1.0)]
    out = expand(base, FakeLinks({}), offset=0.5, cap=10)
    assert [h.chunk_id for h in out] == ["a", "b"]


def test_added_children_have_no_component_ranks():
    out = expand(hits("h"), FakeLinks({"h": ["c"]}), offset=0.5, cap=10)
    c = out[1]
    assert c.keyword_rank is None and c.vector_rank is None and c.rerank_score is None


def test_empty_hits_skip_the_engine():
    engine = FakeLinks({})
    assert expand([], engine, offset=0.5, cap=10) == []
    assert engine.calls == []


def test_self_link_is_ignored():
    out = expand(hits("h"), FakeLinks({"h": ["h"]}), offset=0.5, cap=10)
    assert [(h.chunk_id, h.fused_rank) for h in out] == [("h", 1.0)]


def test_bad_cap_rejected():
    with pytest.raises(ValueError):
        expand(hits("a"), FakeLinks({}), offset=0.5, cap=0)
