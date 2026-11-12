from contextlib import contextmanager

import pytest

from nyaya.eval.datasets import DevItem, Gold
from nyaya.eval.retrieval import Ranking
from nyaya.eval.threshold import (
    as_dict,
    candidate_thresholds,
    choose_threshold,
    tune,
    write_thresholds,
)


def test_candidates():
    assert candidate_thresholds([0.2, None, 0.6, 0.2]) == pytest.approx([0.2, 0.4, 0.600001])
    assert candidate_thresholds([None]) == [0.0]


def test_choose_maximises_correct_refusals_within_limit():
    answerable = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.1]
    unanswerable = [0.05, 0.15, 0.2, 0.32, 0.5]
    c = choose_threshold(answerable, unanswerable, max_wrong_share=0.10)
    # refusing below 0.2 (midpoint of 0.1 and 0.3) loses one answerable (10%)
    assert c.threshold == pytest.approx(0.25)
    assert c.wrong_refusal_share == pytest.approx(0.1)
    assert c.correct_refusal_share == pytest.approx(0.6)
    assert c.feasible and c.n_answerable == 10 and c.n_unanswerable == 5


def test_ties_prefer_fewer_wrong_refusals_then_lower_threshold():
    c = choose_threshold([0.9, 0.8], [0.1, 0.2], max_wrong_share=0.5)
    assert c.correct_refusal_share == 1.0 and c.wrong_refusal_share == 0.0
    assert c.threshold == pytest.approx(0.5)


def test_no_scores_are_always_refused():
    c = choose_threshold([0.9, None], [None, None], max_wrong_share=0.1)
    assert not c.feasible
    assert c.wrong_refusal_share == 0.5 and c.correct_refusal_share == 1.0


def test_needs_both_sets():
    with pytest.raises(ValueError):
        choose_threshold([], [0.1], 0.1)


class ScoreRetriever:
    def __init__(self, scores):
        self.scores = scores

    def run(self, method, question, filters):
        assert method == "hybrid_rerank"
        return Ranking(["x#judgment#1"], self.scores[question])


def test_tune_per_reranker():
    items = [
        DevItem(id="a1", question="a1", type="answerable", gold=[Gold(path="p")]),
        DevItem(id="a2", question="a2", type="answerable", gold=[Gold(path="p")]),
        DevItem(id="u1", question="u1", type="unanswerable"),
    ]
    out = tune(
        items,
        {
            "qwen": ScoreRetriever({"a1": 0.9, "a2": 0.7, "u1": 0.2}),
            "cohere": ScoreRetriever({"a1": 0.3, "a2": 0.25, "u1": 0.1}),
        },
        max_wrong_share=0.0,
    )
    assert out["qwen"].threshold == pytest.approx(0.45)
    assert out["cohere"].threshold == pytest.approx(0.175)
    assert as_dict(out)["qwen"]["correct_refusal_share"] == 1.0
    with pytest.raises(ValueError, match="unknown reranker"):
        tune(items, {"bge": ScoreRetriever({})}, 0.1)


class Conn:
    def __init__(self, row):
        self.row = row
        self.sql = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, q, p):
        self.sql.append((q, p))
        row = self.row

        class R:
            def fetchone(self):
                return row

        return R()


def test_write_thresholds_updates_active_row():
    from nyaya.eval.threshold import ThresholdChoice

    c = ThresholdChoice(0.4, 0.9, 0.05, 10, 10, True)
    conn = Conn(("v2025",))
    assert write_thresholds(conn, {"qwen": c, "cohere": c}) == "v2025"
    q, p = conn.sql[0]
    assert q == (
        "UPDATE index_versions SET threshold_qwen = %s, threshold_cohere = %s "
        "WHERE status = 'active' RETURNING id"
    )
    assert p == [0.4, 0.4]
    with pytest.raises(RuntimeError, match="no active"):
        write_thresholds(Conn(None), {"qwen": c})
    with pytest.raises(ValueError):
        write_thresholds(conn, {})
