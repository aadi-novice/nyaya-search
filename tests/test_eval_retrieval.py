import pytest

from eval_fakes import FakeEngine, chunk, simple_rrf
from nyaya.eval.datasets import DevItem, Gold
from nyaya.eval.retrieval import (
    EngineRetriever,
    Ranking,
    evaluate,
    first_hit_rank,
    label_matches,
    question_metrics,
    scored_view,
)

A, B, C = "2025_1_1_10", "2025_1_11_20", "2025_1_21_30"


def item(**kw):
    base = {
        "id": "q1",
        "question": "q",
        "type": "answerable",
        "gold": [Gold(path=A, para_labels=["12"])],
    }
    base.update(kw)
    return DevItem(**base)


CHUNKS = [
    chunk(f"{A}#editorial#001", refs=["12"]),
    chunk(f"{A}#issue#000"),
    chunk(f"{A}#judgment#005", labels=["3", "4"]),
    chunk(f"{A}#judgment#009", labels=["12.1", "12.2"]),
    chunk(f"{B}#judgment#001", labels=["12"]),
    chunk(f"{C}#judgment#001", labels=["1"]),
]
BY_ID = {c.chunk_id: c for c in CHUNKS}
LINKS = {f"{A}#editorial#001": [f"{A}#judgment#009"]}


def test_label_matches():
    assert label_matches("12", "12")
    assert label_matches("12.1", "12")
    assert label_matches("12", "12.1")
    assert not label_matches("120", "12")
    assert not label_matches("1", "12")


def test_held_chunk_replaced_by_linked_judgment_chunks():
    ranked = [f"{B}#judgment#001", f"{A}#editorial#001", f"{A}#issue#000"]
    view = scored_view(ranked, BY_ID, LINKS, item())
    assert [c.chunk_id for c in view] == [f"{B}#judgment#001", f"{A}#judgment#009"]


def test_editorial_kept_when_scored():
    ranked = [f"{A}#editorial#001"]
    view = scored_view(ranked, BY_ID, LINKS, item(scored_zones=["judgment", "editorial"]))
    assert [c.chunk_id for c in view] == [f"{A}#editorial#001"]


def test_exclude_and_duplicates_and_unknown_ids():
    ranked = [f"{B}#judgment#001", "missing#judgment#1", f"{A}#judgment#005", f"{A}#judgment#005"]
    view = scored_view(ranked, BY_ID, LINKS, item(exclude=[B]))
    assert [c.chunk_id for c in view] == [f"{A}#judgment#005"]


def test_first_hit_rank_levels():
    view = [BY_ID[f"{B}#judgment#001"], BY_ID[f"{A}#judgment#005"], BY_ID[f"{A}#judgment#009"]]
    assert first_hit_rank(view, item(), "judgment") == 2
    assert first_hit_rank(view, item(), "paragraph") == 3
    assert first_hit_rank(view[:1], item(), "judgment") is None


def test_question_metrics():
    assert question_metrics(3) == {"recall@5": 1.0, "recall@50": 1.0, "rr": pytest.approx(1 / 3)}
    assert question_metrics(7) == {"recall@5": 0.0, "recall@50": 1.0, "rr": pytest.approx(1 / 7)}
    assert question_metrics(None) == {"recall@5": 0.0, "recall@50": 0.0, "rr": 0.0}


def make_retriever(engine, rerank=None, expand=None):
    return EngineRetriever(
        engine=engine,
        embed_query=lambda q: (1.0, 0.0) if q == "bail" else (0.0, 1.0),
        fuse=simple_rrf,
        rerank=rerank,
        expand=expand,
        keyword_k=50,
        vector_k=50,
        rrf_k=60,
        rerank_candidates=100,
    )


def engine():
    return FakeEngine(
        CHUNKS,
        keyword={"bail": [f"{B}#judgment#001", f"{A}#editorial#001"]},
        vector={
            ((1.0, 0.0), True): [f"{A}#judgment#009"],
            ((1.0, 0.0), False): [f"{C}#judgment#001", f"{A}#judgment#005"],
        },
        links=LINKS,
    )


def test_engine_retriever_methods():
    eng = engine()
    r = make_retriever(eng, rerank=lambda q, cs: [float(c.judgment_id == A) for c in cs])
    assert r.run("keyword", "bail", None).chunk_ids == [f"{B}#judgment#001", f"{A}#editorial#001"]
    assert r.run("exact", "bail", None).chunk_ids == [f"{A}#judgment#009"]
    assert r.run("hnsw", "bail", None).chunk_ids == [f"{C}#judgment#001", f"{A}#judgment#005"]
    hybrid = r.run("hybrid", "bail", None).chunk_ids
    assert set(hybrid) == {
        f"{B}#judgment#001",
        f"{A}#editorial#001",
        f"{C}#judgment#001",
        f"{A}#judgment#005",
    }
    rr = r.run("hybrid_rerank", "bail", None)
    assert rr.top_score == 1.0
    assert {rr.chunk_ids[0], rr.chunk_ids[1]} == {f"{A}#editorial#001", f"{A}#judgment#005"}
    # the query vector is computed once per question
    assert sum(1 for c in eng.calls if c[0] == "vector") == 4


def test_engine_retriever_expand_and_errors():
    seen = []

    def expand(fused, eng):
        seen.append(len(fused))
        return fused[:1]

    r = make_retriever(engine(), expand=expand)
    assert len(r.run("hybrid", "bail", None).chunk_ids) == 1 and seen == [4]
    with pytest.raises(ValueError, match="reranker"):
        r.run("hybrid_rerank", "bail", None)
    with pytest.raises(ValueError, match="unknown method"):
        r.run("bm42", "bail", None)
    empty = make_retriever(FakeEngine(CHUNKS), rerank=lambda q, cs: [])
    assert empty.run("hybrid_rerank", "nothing", None) == Ranking([], None)


def test_evaluate_report():
    eng = engine()
    r = make_retriever(eng, rerank=lambda q, cs: [float(c.judgment_id == A) for c in cs])
    items = [
        item(id="q1", question="bail"),
        item(id="q2", question="other"),
        DevItem(id="u1", question="none", type="unanswerable"),
    ]
    rep = evaluate(items, r, eng)
    assert rep.item_ids == ["q1", "q2"]
    s = rep.summary()
    # keyword: B (miss), then A's Held chunk -> A#judgment#009 at rank 2 (paragraph hit)
    assert s["keyword"]["judgment"] == {"recall@5": 0.5, "recall@50": 0.5, "mrr": 0.25}
    assert s["keyword"]["paragraph"]["mrr"] == 0.25
    # exact: A#judgment#009 at rank 1
    assert s["exact"]["paragraph"]["recall@5"] == 0.5
    # hnsw: C then A#judgment#005 (judgment hit, not a paragraph hit)
    assert s["hnsw"]["judgment"]["mrr"] == 0.25
    assert s["hnsw"]["paragraph"]["recall@50"] == 0.0
    assert rep.top_scores == {"q1": 1.0, "q2": None}
    assert rep.vector("keyword", "judgment", "mrr") == [0.5, 0.0]
