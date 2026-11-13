"""Hybrid retrieval wiring and the refusal gate, over in-memory fakes."""

import pytest

from nyaya.config import load_settings
from nyaya.contracts import Filters
from nyaya.search.retrieve import hybrid_retrieve, refusal_threshold, should_refuse
from search_fakes import FakeEmbedder, FakeEngine, WordCountReranker, make_chunk


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def corpus():
    return [
        make_chunk("a", text="anticipatory bail under section 438"),
        make_chunk("b", text="tenancy and rent arrears"),
        make_chunk("c", zone="editorial", text="held bail is discretionary"),
        make_chunk("c", seq=7, text="paragraph seven on custody"),
    ]


def test_pipeline_steps_and_scores(settings):
    engine = FakeEngine(corpus(), links={"c#editorial#001": ["c#judgment#007"]})
    emb, rr = FakeEmbedder(), WordCountReranker()
    f = Filters(year_from=2020)
    res = hybrid_retrieve("anticipatory bail 438", f, engine, emb, rr, settings)
    assert emb.questions == ["anticipatory bail 438"]
    assert engine.calls[0] == ("keyword", "anticipatory bail 438", f, settings.retrieval.keyword_k)
    assert engine.calls[1] == ("vector", f, settings.retrieval.vector_k, False)
    # the linked judgment chunk joined the candidates
    assert "c#judgment#007" in [h.chunk_id for h in res.candidates]
    # reranker saw header + blank line + text
    _, docs, top_n = rr.calls[0]
    assert top_n == settings.retrieval.final_k
    assert docs[0].startswith("Header\n\n")
    assert res.reranked[0].chunk_id == "a#judgment#001"
    assert res.top_score == pytest.approx(1.0)
    assert res.reranker_kind == "qwen"
    assert all(h.rerank_score is not None for h in res.reranked)
    assert res.rows_returned["keyword"]["returned"] == 2
    assert set(res.timings_ms) >= {"keyword", "embed_query", "vector", "fuse_expand", "rerank"}


def test_exact_flag_passed(settings):
    engine = FakeEngine(corpus())
    hybrid_retrieve("bail", None, engine, FakeEmbedder(), WordCountReranker(), settings, exact=True)
    assert engine.calls[1][3] is True


def test_no_candidates(settings):
    engine = FakeEngine([], vector_order=[])
    rr = WordCountReranker()
    res = hybrid_retrieve("bail", None, engine, FakeEmbedder(), rr, settings)
    assert res.candidates == [] and res.reranked == [] and res.rerank is None
    assert res.top_score is None and res.reranker_kind is None
    assert rr.calls == []


def test_candidates_missing_from_store_are_skipped(settings):
    engine = FakeEngine(corpus(), vector_order=["ghost#judgment#001", "b#judgment#001"])
    res = hybrid_retrieve("rent", None, engine, FakeEmbedder(), WordCountReranker(), settings)
    assert "ghost#judgment#001" in [h.chunk_id for h in res.candidates]
    assert "ghost#judgment#001" not in [h.chunk_id for h in res.reranked]


def test_fallback_kind_reported(settings):
    res = hybrid_retrieve(
        "bail",
        None,
        FakeEngine(corpus()),
        FakeEmbedder(),
        WordCountReranker(kind="cohere", fallback_used=True),
        settings,
    )
    assert res.reranker_kind == "cohere" and res.rerank.fallback_used


# --- refusal gate ----------------------------------------------------------------------

TH = {"qwen": 0.35, "cohere": 0.20}


@pytest.mark.parametrize(
    ("score", "kind", "refuse"),
    [
        (0.34, "qwen", True),
        (0.35, "qwen", False),
        (0.9, "qwen", False),
        (0.25, "cohere", False),
        (0.19, "cohere", True),
        (None, "qwen", True),
        (0.9, None, True),
    ],
)
def test_should_refuse(score, kind, refuse):
    assert should_refuse(score, kind, TH) is refuse


def test_threshold_errors():
    with pytest.raises(ValueError, match="unknown reranker"):
        refusal_threshold("bge", TH)
    with pytest.raises(ValueError, match="no threshold_cohere"):
        refusal_threshold("cohere", {"qwen": 0.3, "cohere": None})
    assert refusal_threshold("qwen", TH) == 0.35
