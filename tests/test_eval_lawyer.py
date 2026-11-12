"""Lawyer review: export to CSV and HTML, then import the ratings (round trip)."""

from __future__ import annotations

import csv

import pytest

from nyaya.eval import lawyer as LW


def record(i, answered=True):
    cite = {
        "case_name": f"A{i} v. State",
        "scr_citation": f"[2025] 1 S.C.R. {i}",
        "paras": ["9", "10"],
        "quote": f"the quote <{i}> & more words",
    }
    return {
        "id": f"dev-{i:03d}",
        "question": f"Question {i}?",
        "answer": [{"sentence": f"Sentence {i}.", "citations": [cite]}] if answered else [],
        "gaps": [f"Gap {i}."] if i % 2 else [],
    }


RECORDS = [record(i, answered=i % 5 != 0) for i in range(1, 41)]


def test_sample_is_seeded_and_answered_only():
    a = LW.sample_answers(RECORDS, 10, seed=13)
    b = LW.sample_answers(list(reversed(RECORDS)), 10, seed=13)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 10 and all(r["answer"] for r in a)
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)
    assert LW.sample_answers(RECORDS, 10, seed=14) != a
    assert len(LW.sample_answers(RECORDS, 1000, seed=1)) == 32


def test_export_writes_csv_html_and_rubric(tmp_path):
    info = LW.export(RECORDS, tmp_path / "out", run_id="answers-1", size=5, seed=13)
    assert info["exported"] == 5 and info["answered"] == 32
    with open(info["csv"], encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(LW.COLUMNS)
    assert [r["review_id"] for r in rows] == ["r001", "r002", "r003", "r004", "r005"]
    assert all(r[q] == "" for r in rows for q in LW.QUESTIONS)
    assert "[A" in rows[0]["answer"] and "para 9, 10" in rows[0]["answer"]
    page = (tmp_path / "out" / "review.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<title>Lawyer review</title>" in page
    assert "&lt;" in page and "<script" not in page  # escaped, standalone
    assert page.count("<article>") == 5
    assert (tmp_path / "out" / "rubric_v1.md").exists()


def fill(path, ratings):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row, values in zip(rows, ratings, strict=False):
        row.update(values)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LW.COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_export_import_round_trip(tmp_path):
    info = LW.export(RECORDS, tmp_path, run_id="answers-1", size=3, seed=13)
    path = tmp_path / "ratings.csv"
    fill(
        path,
        [
            {
                "right_paragraphs": "3",
                "law_correct": "3",
                "nothing_missing": "2",
                "usable": "3",
                "reviewer": "AB",
                "comment": "good",
            },
            {
                "right_paragraphs": "1",
                "law_correct": "2",
                "nothing_missing": "",
                "usable": "1",
                "reviewer": "AB",
            },
            {"right_paragraphs": "3", "law_correct": "3", "nothing_missing": "3", "usable": "2"},
        ],
    )
    rows = LW.read_ratings(path)
    with open(info["csv"], encoding="utf-8-sig", newline="") as f:
        exported = [r["question_id"] for r in csv.DictReader(f)]
    assert [r["question_id"] for r in rows] == exported
    m = LW.summarize_ratings(rows)
    assert m["answers"] == 3 and m["fully_rated"] == 2 and m["reviewers"] == ["AB"]
    rp = m["questions"]["right_paragraphs"]
    assert rp["mean"] == pytest.approx(7 / 3) and rp["counts"] == {"1": 1, "2": 0, "3": 2}
    assert m["questions"]["nothing_missing"]["rated"] == 2
    assert m["gate"] is False


def test_bad_rating_is_rejected_with_the_row(tmp_path):
    LW.export(RECORDS, tmp_path, run_id="answers-1", size=2, seed=13)
    path = tmp_path / "ratings.csv"
    fill(path, [{}, {"usable": "4"}])
    with pytest.raises(LW.RatingsError, match=r"ratings.csv:3 \(r002\): usable"):
        LW.read_ratings(path)


def test_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("review_id,question_id\nr1,q1\n", encoding="utf-8")
    with pytest.raises(LW.RatingsError, match="missing columns"):
        LW.read_ratings(path)
