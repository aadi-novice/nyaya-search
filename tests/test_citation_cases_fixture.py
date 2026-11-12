"""Checks that citation_cases.jsonl agrees with the design's citation-check rules.

The citation checker itself (answer/citation_check.py) is Phase 1; its test,
test_citation_check.py, runs every case through it. This Phase 0 test keeps the
fixture honest using only Phase 0 code (normalize_text) and the rules as written
in the design ("Citation check (code, not the LLM)").
"""

import pytest
import regex

from nyaya.contracts import UnverifiedReason
from nyaya.text.normalize import normalize_text

from conftest import FIXTURES, read_jsonl

CASES = read_jsonl(FIXTURES / "citation_cases.jsonl")
MIN_WORDS, MAX_WORDS = 8, 40  # quote.min_words, quote.max_words
CASE_NAME_RE = regex.compile(
    r"([A-Z][\w.&'’()-]*(?:\s+[A-Z&][\w.&'’()-]*){0,8})\s+(?:v\.|vs\.?|versus)\s+"
    r"([A-Z][\w.&'’()-]*(?:\s+[A-Z&][\w.&'’()-]*){0,8})"
)


def citation_failure(passages: dict, cit: dict) -> str | None:
    """First failing step (1-3) for one citation, or None if it passes."""
    passage = passages.get(cit["label"])
    if passage is None:
        return "unknown_label"
    if not MIN_WORDS <= len(cit["quote"].split()) <= MAX_WORDS:
        return "quote_length"
    if normalize_text(cit["quote"]) not in normalize_text(passage["text"]):
        return "quote_not_found"
    return None


def has_uncited_case_name(passages: dict, sentence: str) -> bool:
    titles = [normalize_text(p["title"]) for p in passages.values()]
    for m in CASE_NAME_RE.finditer(sentence):
        left, right = normalize_text(m.group(1)), normalize_text(m.group(2))
        if not any(left in t and right in t for t in titles):
            return True
    return False


def test_fixture_size():
    assert len(CASES) >= 40
    assert len({c["id"] for c in CASES}) == len(CASES)


def test_required_categories_present():
    cats = {c["category"] for c in CASES}
    required = {
        "exact_quote",
        "case_difference",
        "quote_length",
        "unknown_label",
        "uncited_case_name",
        "union_of_india",
        "header_quote",
        "no_citation",
    }
    assert required <= cats
    lengths = {
        len(cit["quote"].split())
        for c in CASES
        if c["category"] == "quote_length"
        for cit in c["sentence"]["citations"]
    }
    assert {7, 41} <= lengths


def test_union_of_india_cases_pass():
    for c in CASES:
        if c["category"] == "union_of_india":
            assert c["expected"]["verified"] is True
            assert "Union of India" in c["sentence"]["sentence"] or c["note"]


def test_header_quotes_are_in_header_only():
    for c in CASES:
        if c["category"] != "header_quote":
            continue
        for cit in c["sentence"]["citations"]:
            p = c["passages"][cit["label"]]
            assert normalize_text(cit["quote"]) in normalize_text(p["header"])
            assert normalize_text(cit["quote"]) not in normalize_text(p["text"])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_verdict_follows_design_rules(case):
    passages = case["passages"]
    sent = case["sentence"]
    expected = case["expected"]
    if expected["reason"] is not None:
        UnverifiedReason(expected["reason"])  # a valid wire value
    assert expected["verified"] == (expected["reason"] is None)

    cits = sent["citations"]
    if not cits:
        assert expected == {"verified": False, "reason": "no_citation"}
        return
    failures = [citation_failure(passages, cit) for cit in cits]
    if any(f is None for f in failures):
        if has_uncited_case_name(passages, sent["sentence"]):
            assert expected == {"verified": False, "reason": "uncited_case_name"}
        else:
            assert expected == {"verified": True, "reason": None}
            # property: a verified sentence has a citation whose quote is a
            # normalised substring of its chunk text
            assert any(
                normalize_text(c["quote"]) in normalize_text(passages[c["label"]]["text"])
                for c in cits
                if c["label"] in passages
            )
    else:
        # every citation fails; the cases are written so they agree on the reason
        assert len(set(failures)) == 1
        assert expected == {"verified": False, "reason": failures[0]}
