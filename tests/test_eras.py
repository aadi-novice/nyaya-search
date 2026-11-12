"""Era classification rules (Technical Plan step 3)."""

import pytest

from corpus_helpers import era_c_pdf, plain_pdf
from nyaya.config import load_settings
from nyaya.contracts import Era
from nyaya.parse.eras import classify, classify_pdf, count_d_labels, count_margin_lines

MARKERS = load_settings(env_file=None).era_markers

ERA_C_TEXT = "\n".join(
    [
        "[2015] 2 S.C.R. 882",
        "A",
        "UNION OF INDIA v. DILEEP KUMAR SINGH",
        "B",
        "Persons with Disabilities Act, 1995 - s.47",
        "C",
        "D",
        "  E  ",
        "HELD: 1. The proviso to s.47 would apply.",
        "F",
    ]
)

ERA_D_TEXT = """[2025] 1 S.C.R. 105 : 2025 INSC 20
Issue for Consideration
Whether the High Court was correct.
Headnotes
Held: Tribunals may order eviction.
Case Law
Cited
Brahmpal v. National Insurance Company
List of Acts
Maintenance and Welfare of Parents Act, 2007.
"""


def test_era_d_needs_three_labels():
    assert count_d_labels(ERA_D_TEXT, MARKERS) == 4  # "Case Law\nCited" wraps
    assert classify(ERA_D_TEXT, 2025, MARKERS) is Era.D


def test_era_d_ignores_decision_year():
    assert classify(ERA_D_TEXT, 2019, MARKERS) is Era.D


def test_two_d_labels_are_not_enough():
    text = "Issue for Consideration\nHeadnotes\nHeld: x"
    assert classify(text, 2025, MARKERS) is Era.OUT_OF_SCOPE


def test_era_c():
    assert count_margin_lines(ERA_C_TEXT, MARKERS) == 6
    assert classify(ERA_C_TEXT, 2015, MARKERS) is Era.C


@pytest.mark.parametrize("year", [2009, 2024, 1995])
def test_era_c_outside_its_years_is_out_of_scope(year):
    assert classify(ERA_C_TEXT, year, MARKERS) is Era.OUT_OF_SCOPE


@pytest.mark.parametrize("year", [2010, 2023])
def test_era_c_year_bounds_are_inclusive(year):
    assert classify(ERA_C_TEXT, year, MARKERS) is Era.C


def test_era_c_needs_held():
    text = ERA_C_TEXT.replace("HELD:", "Held that")
    assert classify(text, 2015, MARKERS) is Era.OUT_OF_SCOPE


def test_era_c_held_is_case_insensitive():
    assert classify(ERA_C_TEXT.replace("HELD:", "Held:"), 2015, MARKERS) is Era.C


def test_era_c_held_allows_space_before_colon():
    assert classify(ERA_C_TEXT.replace("HELD:", "HELD :"), 2015, MARKERS) is Era.C


def test_era_c_needs_five_margin_lines():
    lines = [ln for ln in ERA_C_TEXT.splitlines() if ln.strip() not in {"A", "B"}]
    text = "\n".join(lines)
    assert count_margin_lines(text, MARKERS) == 4
    assert classify(text, 2015, MARKERS) is Era.OUT_OF_SCOPE


def test_letters_inside_lines_are_not_margin_lines():
    text = "A proviso\nB.\nI\nJ\nAB\n" + "HELD: x"
    assert count_margin_lines(text, MARKERS) == 0


def test_d_wins_over_c():
    assert classify(ERA_C_TEXT + "\n" + ERA_D_TEXT, 2015, MARKERS) is Era.D


def test_default_markers_come_from_settings():
    assert classify(ERA_D_TEXT, 2025) is Era.D


def test_classify_pdf_on_synthetic_pages():
    body = ["HELD: 1. The appeal is allowed.", "The proviso applies to the whole Act."]
    pdf = era_c_pdf([body, ["more text"], ["even more"], ["page four"]])
    assert classify_pdf(pdf, 2020, MARKERS) is Era.C
    assert classify_pdf(plain_pdf([ERA_D_TEXT.splitlines()]), 2025, MARKERS) is Era.D
