"""`nyaya-eval tune-threshold`: the refusal threshold for each reranker (G5).

Design: "Refusal gate", Goal G5, file 43.

The gate refuses when the top rerank score is below the threshold. For each
reranker, the chosen threshold maximises correct refusals of unanswerable
questions while keeping wrong refusals of answerable questions at
`refusal.max_wrong_refusal_share` or less. Ties go to fewer wrong refusals,
then to the lower threshold. A question with no candidates (no top score) is
always refused. Thresholds are tuned on the development set only, and written
to the active `index_versions` row, the only place the gate reads.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from nyaya.eval.datasets import DevItem
from nyaya.eval.retrieval import Retriever

COLUMNS = {"qwen": "threshold_qwen", "cohere": "threshold_cohere"}


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    correct_refusal_share: float  # unanswerable questions refused
    wrong_refusal_share: float  # answerable questions refused
    n_answerable: int
    n_unanswerable: int
    feasible: bool  # False: no threshold meets the wrong-refusal limit; 0 refusals chosen


def _refused(score: float | None, t: float) -> bool:
    return score is None or score < t


def _share(scores: Sequence[float | None], t: float) -> float:
    return sum(_refused(s, t) for s in scores) / len(scores) if scores else 0.0


def candidate_thresholds(scores: Sequence[float | None]) -> list[float]:
    """The lowest score (refuses nothing scored), midpoints, and just above the highest."""
    xs = sorted({float(s) for s in scores if s is not None})
    if not xs:
        return [0.0]
    mids = [(a + b) / 2.0 for a, b in itertools.pairwise(xs)]
    return [xs[0], *mids, xs[-1] + 1e-6]


def choose_threshold(
    answerable: Sequence[float | None],
    unanswerable: Sequence[float | None],
    max_wrong_share: float,
) -> ThresholdChoice:
    if not answerable or not unanswerable:
        raise ValueError("need both answerable and unanswerable questions")
    best: tuple[float, float, float] | None = None  # (-correct, wrong, t)
    for t in candidate_thresholds([*answerable, *unanswerable]):
        wrong = _share(answerable, t)
        if wrong > max_wrong_share + 1e-12:
            continue
        key = (-_share(unanswerable, t), wrong, t)
        if best is None or key < best:
            best = key
    feasible = best is not None
    if best is None:  # even refusing nothing breaks the limit (unscored answerables)
        t = candidate_thresholds([*answerable, *unanswerable])[0]
        best = (-_share(unanswerable, t), _share(answerable, t), t)
    return ThresholdChoice(
        threshold=best[2],
        correct_refusal_share=-best[0],
        wrong_refusal_share=best[1],
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        feasible=feasible,
    )


def top_scores(items: Sequence[DevItem], retriever: Retriever) -> list[float | None]:
    return [retriever.run("hybrid_rerank", i.question, i.filters).top_score for i in items]


def tune(
    items: Sequence[DevItem],
    retrievers: Mapping[str, Retriever],
    max_wrong_share: float,
) -> dict[str, ThresholdChoice]:
    """One choice per reranker name (`qwen`, `cohere`)."""
    answerable = [i for i in items if i.type == "answerable"]
    unanswerable = [i for i in items if i.type == "unanswerable"]
    out = {}
    for name, retriever in retrievers.items():
        if name not in COLUMNS:
            raise ValueError(f"unknown reranker {name!r}; expected one of {sorted(COLUMNS)}")
        out[name] = choose_threshold(
            top_scores(answerable, retriever), top_scores(unanswerable, retriever), max_wrong_share
        )
    return out


def write_thresholds(conn: Any, choices: Mapping[str, ThresholdChoice]) -> str:
    """Write each threshold to the active index version; returns its id."""
    if not choices:
        raise ValueError("no thresholds to write")
    sets = ", ".join(f"{COLUMNS[name]} = %s" for name in choices)
    params = [c.threshold for c in choices.values()]
    with conn.transaction():
        row = conn.execute(
            f"UPDATE index_versions SET {sets} WHERE status = 'active' RETURNING id", params
        ).fetchone()
    if row is None:
        raise RuntimeError("no active index version")
    return row[0]


def as_dict(choices: Mapping[str, ThresholdChoice]) -> dict[str, dict[str, Any]]:
    return {k: asdict(v) for k, v in choices.items()}
