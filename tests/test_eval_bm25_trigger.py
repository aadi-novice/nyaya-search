import pytest

from eval_fakes import FakeEngine, chunk
from nyaya.eval.bm25_trigger import Bm25Index, TriggerRetriever, load_corpus, run_bm25_trigger
from nyaya.eval.datasets import DevItem, Gold

TEXTS = {
    "A#judgment#001": "anticipatory bail under section 438 needs gravity of accusation weighed",
    "B#judgment#001": "the arbitral award was set aside under section 34 for patent illegality",
    "C#judgment#001": "a dying declaration must inspire confidence before conviction rests on it",
    "D#judgment#001": "land acquisition compensation market value determined from sale deeds",
}


def make():
    chunks = [chunk(cid, labels=["1"], text=t) for cid, t in TEXTS.items()]
    return chunks, Bm25Index(list(TEXTS), list(TEXTS.values()))


def test_bm25_index_search():
    _, idx = make()
    assert idx.search("arbitral award illegality", 2)[0] == "B#judgment#001"
    assert idx.search("the of and", 5) == []  # stopwords only
    assert idx.search("zebra", 5) == []
    assert len(idx.search("section", 50)) <= 4


def test_bm25_index_rejects_empty():
    with pytest.raises(ValueError):
        Bm25Index([], [])


def test_trigger_fires_when_bm25_is_clearly_better():
    chunks, idx = make()
    questions = {
        "pre-arrest bail gravity": "A",
        "arbitral award patent illegality": "B",
        "dying declaration confidence": "C",
        "market value sale deeds": "D",
    }
    eng = FakeEngine(chunks, keyword={})  # Postgres finds nothing
    items = [
        DevItem(id=f"q{i}", question=q, type="answerable", gold=[Gold(path=p, para_labels=["1"])])
        for i, (q, p) in enumerate(questions.items())
    ] * 5
    items = [it.model_copy(update={"id": f"q{n}"}) for n, it in enumerate(items)]
    out = run_bm25_trigger(items, eng, idx, k=50, samples=500, seed=7, min_gap_points=3.0)
    assert out["summary"]["bm25"]["judgment"]["recall@50"] == 1.0
    assert out["summary"]["keyword"]["judgment"]["recall@50"] == 0.0
    assert out["gap_points"]["mean"] == pytest.approx(100.0)
    assert out["fires"] is True and out["questions"] == 20


def test_trigger_does_not_fire_when_equal():
    chunks, idx = make()
    q = "arbitral award patent illegality"
    eng = FakeEngine(chunks, keyword={q: ["B#judgment#001"]})
    items = [DevItem(id="q1", question=q, type="answerable", gold=[Gold(path="B")])]
    out = run_bm25_trigger(items, eng, idx, k=50, samples=100, seed=7, min_gap_points=3.0)
    assert out["fires"] is False


def test_trigger_needs_answerable_questions():
    chunks, idx = make()
    with pytest.raises(ValueError, match="no answerable"):
        run_bm25_trigger(
            [DevItem(id="u", question="x", type="unanswerable")],
            FakeEngine(chunks),
            idx,
            k=50,
            samples=10,
            seed=1,
            min_gap_points=3.0,
        )


def test_trigger_retriever_unknown_method():
    chunks, idx = make()
    with pytest.raises(ValueError):
        TriggerRetriever(FakeEngine(chunks), idx).run("hnsw", "q", None)


def test_load_corpus():
    class Conn:
        def execute(self, q, p):
            assert "header || ' ' || text" in q and p == ("v1",)

            class R:
                def fetchall(self):
                    return [("a#judgment#1", "h t")]

            return R()

    assert load_corpus(Conn(), "v1") == (["a#judgment#1"], ["h t"])
