import pytest

from nyaya.text.normalize import normalize_judge, normalize_text

from conftest import FIXTURES, read_jsonl

CASES = read_jsonl(FIXTURES / "normalize_cases.jsonl")
TEXT_CASES = [c for c in CASES if c["fn"] == "normalize_text"]
JUDGE_CASES = [c for c in CASES if c["fn"] == "normalize_judge"]


def test_fixture_size_and_coverage():
    assert len(TEXT_CASES) >= 40
    assert {"quotes", "dashes", "spacing", "nfkc"} <= {c["category"] for c in TEXT_CASES}
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", TEXT_CASES, ids=[c["id"] for c in TEXT_CASES])
def test_normalize_text(case):
    assert normalize_text(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", JUDGE_CASES, ids=[c["id"] for c in JUDGE_CASES])
def test_normalize_judge(case):
    assert normalize_judge(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", TEXT_CASES, ids=[c["id"] for c in TEXT_CASES])
def test_normalize_text_is_idempotent(case):
    once = normalize_text(case["input"])
    assert normalize_text(once) == once


@pytest.mark.parametrize("case", JUDGE_CASES, ids=[c["id"] for c in JUDGE_CASES])
def test_normalize_judge_is_idempotent(case):
    once = normalize_judge(case["input"])
    assert normalize_judge(once) == once


def test_normalised_quote_is_substring_of_normalised_text():
    text = "The Court held that the “right to life” – under Art. 21 – is wide."
    quote = 'the "RIGHT to life" - under  art. 21'
    assert normalize_text(quote) in normalize_text(text)


def test_judge_initial_j_is_kept_but_suffix_removed():
    # "J." as a first initial is part of the name; ", J." at the end is a title.
    assert normalize_judge("J. Chelameswar, J.") == "J CHELAMESWAR"
    assert normalize_judge("RAJ") == "RAJ"
    assert normalize_judge("ARIJIT PASAYAT") == "ARIJIT PASAYAT"
