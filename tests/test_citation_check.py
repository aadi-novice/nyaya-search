"""citation_check.check (safety-critical): every fixture case, plus edge cases.

The fixture passages carry a title, chunk text and header. The citation-pattern
part of step 4 needs `find_citations` (link/aliases.py) and `resolve_alias`
(the search engine); both are faked here with the design's SCR and neutral
patterns.
"""

from __future__ import annotations

import re

import pytest

from nyaya.answer.citation_check import CheckResult, check
from nyaya.config import load_settings
from nyaya.contracts import (
    ChunkRecord,
    PassageRef,
    UnverifiedReason,
    UnverifiedSentence,
    Zone,
)
from nyaya.text.normalize import normalize_text

from conftest import FIXTURES, read_jsonl

CASES = read_jsonl(FIXTURES / "citation_cases.jsonl")
SETTINGS = load_settings(env_file=None)

_SCR = re.compile(r"\[(\d{4})\]\s*(\d+)\s*S\.?\s*C\.?\s*R\.?\s*(\d+)")
_INSC = re.compile(r"(\d{4})\s*INSC\s*(\d+)")
_SCC = re.compile(r"\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)")
_CNR = re.compile(r"\b[A-Z]{4}\d{12}\b")


def fake_find_citations(text: str):
    out = []
    for m in _SCR.finditer(text):
        out.append((m.span(), "scr", f"scr:{m[1]}:{m[2]}:{m[3]}"))
    for m in _INSC.finditer(text):
        out.append((m.span(), "neutral", f"insc:{m[1]}:{int(m[2])}"))
    for m in _SCC.finditer(text):
        out.append((m.span(), "scc", f"scc:{m[1]}:{m[2]}:{m[3]}"))
    for m in _CNR.finditer(text):
        out.append((m.span(), "cnr", f"cnr:{m[0].lower()}"))
    return out


ALIASES = {
    "scr:2025:1:230": "2025_1_230_238",
    "insc:2025:31": "2025_1_230_238",
    "scr:1978:2:621": "1978_2_621_700",
    "scr:2001:1:1": "2001_1_1_9",  # a loaded judgment not among the passages
}


def fake_resolve_alias(canonical: str) -> str | None:
    return ALIASES.get(canonical)


def make_chunk(p: dict) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=p["chunk_id"],
        judgment_id=p["judgment_id"],
        zone=Zone(p["zone"]),
        opinion_type=p.get("opinion_type"),
        para_labels=[],
        para_refs=[],
        text=p["text"],
        header=p["header"],
        char_start=0,
        char_end=len(p["text"]),
        pages=[1],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch1",
    )


def make_ref(p: dict) -> PassageRef:
    return PassageRef(
        chunk_id=p["chunk_id"],
        judgment_id=p["judgment_id"],
        case_name=p["title"],
        scr_citation="",
        pages=[1],
        zone=Zone(p["zone"]),
        opinion_type=p.get("opinion_type"),
    )


def run(passages: dict, sentences: list, gaps=None) -> CheckResult:
    label_map = {label: make_ref(p) for label, p in passages.items()}
    chunks = {p["chunk_id"]: make_chunk(p) for p in passages.values()}
    llm_output = {"sentences": sentences, "gaps": gaps if gaps is not None else []}
    return check(
        llm_output,
        label_map,
        chunks,
        SETTINGS,
        find_citations=fake_find_citations,
        resolve_alias=fake_resolve_alias,
    )


BASE = CASES[0]["passages"]  # P1..P6 in the first case
P1_QUOTE = (
    "The power under Section 482 of the Code of Criminal Procedure must be exercised "
    "sparingly and with circumspection."
)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fixture_case(case):
    result = run(case["passages"], [case["sentence"]])
    expected = case["expected"]
    if expected["verified"]:
        assert len(result.verified) == 1 and not result.unverified
        sent = result.verified[0]
        assert sent.sentence == case["sentence"]["sentence"]
        assert sent.citations
        # property: every kept citation's quote is a normalised substring of its chunk
        for c in sent.citations:
            text = case["passages"][c.passage_label]["text"]
            assert normalize_text(c.quote) in normalize_text(text)
            assert c.chunk_id == case["passages"][c.passage_label]["chunk_id"]
    else:
        assert not result.verified and len(result.unverified) == 1
        u = result.unverified[0]
        assert u.reason == UnverifiedReason(expected["reason"])
        assert u.labels == [c["label"] for c in case["sentence"]["citations"]]
        # property: an unverified sentence never carries citation details
        assert set(UnverifiedSentence.model_fields) == {"sentence", "labels", "reason"}
        assert result.rule_breaks


def test_settings_quote_bounds_are_used():
    assert (SETTINGS.quote.min_words, SETTINGS.quote.max_words) == (8, 40)


def test_failed_citations_on_a_verified_sentence_are_dropped_but_counted():
    sent = {
        "sentence": "Quashing powers are used sparingly.",
        "citations": [{"label": "P9", "quote": P1_QUOTE}, {"label": "P1", "quote": P1_QUOTE}],
    }
    result = run(BASE, [sent])
    assert [c.passage_label for c in result.verified[0].citations] == ["P1"]
    assert [b.kind for b in result.rule_breaks] == ["unknown_label"]
    assert result.rule_breaks[0].label == "P9"


def test_earliest_step_wins_when_all_citations_fail():
    sent = {
        "sentence": "Quashing powers are used sparingly.",
        "citations": [
            {"label": "P1", "quote": "this quote is not anywhere in the passage at all"},
            {"label": "P1", "quote": "too short"},
            {"label": "P7", "quote": P1_QUOTE},
        ],
    }
    result = run(BASE, [sent])
    assert result.unverified[0].reason is UnverifiedReason.UNKNOWN_LABEL
    assert [b.kind for b in result.rule_breaks] == [
        "quote_not_found",
        "quote_length",
        "unknown_label",
    ]


def test_quote_length_beats_quote_not_found():
    sent = {
        "sentence": "x",
        "citations": [
            {"label": "P1", "quote": "this quote is not anywhere in the passage at all"},
            {"label": "P1", "quote": "too short"},
        ],
    }
    assert run(BASE, [sent]).unverified[0].reason is UnverifiedReason.QUOTE_LENGTH


def test_matching_citation_pattern_passes():
    sent = {
        "sentence": "In [2025] 1 S.C.R. 230 (2025 INSC 31) the Court held quashing is rare.",
        "citations": [{"label": "P1", "quote": P1_QUOTE}],
    }
    result = run(BASE, [sent])
    assert len(result.verified) == 1 and not result.rule_breaks


@pytest.mark.parametrize(
    "citation",
    [
        "[2001] 1 S.C.R. 1",  # resolves, but to a judgment not among the passages
        "[1999] 3 S.C.R. 999",  # does not resolve at all
        "(2009) 8 SCC 751",  # SCC citations are never aliases
    ],
)
def test_foreign_citation_pattern_is_a_rule_break(citation):
    sent = {
        "sentence": f"As held in {citation}, quashing is rare.",
        "citations": [{"label": "P1", "quote": P1_QUOTE}],
    }
    result = run(BASE, [sent])
    assert not result.verified
    assert result.unverified[0].reason is UnverifiedReason.UNCITED_CASE_NAME
    assert [b.kind for b in result.rule_breaks] == ["uncited_citation"]
    assert result.rule_breaks[0].detail


def test_cnr_in_sentence_is_not_checked():
    sent = {
        "sentence": "The case ESCR010004242099 shows quashing is rare.",
        "citations": [{"label": "P1", "quote": P1_QUOTE}],
    }
    assert len(run(BASE, [sent]).verified) == 1


def test_case_name_and_citation_breaks_are_both_counted():
    sent = {
        "sentence": "Arnesh Kumar v. State of Bihar, (2014) 8 SCC 273, limits arrest.",
        "citations": [{"label": "P1", "quote": P1_QUOTE}],
    }
    result = run(BASE, [sent])
    assert sorted(b.kind for b in result.rule_breaks) == ["uncited_case_name", "uncited_citation"]
    assert result.unverified[0].reason is UnverifiedReason.UNCITED_CASE_NAME


def test_failing_citations_and_case_name_report_citation_reason():
    sent = {
        "sentence": "Arnesh Kumar v. State of Bihar limits arrest.",
        "citations": [{"label": "P1", "quote": "too short"}],
    }
    result = run(BASE, [sent])
    assert result.unverified[0].reason is UnverifiedReason.QUOTE_LENGTH
    kinds = sorted(b.kind for b in result.rule_breaks)
    assert kinds == ["quote_length", "uncited_case_name"]


def test_gaps_pass_through_and_are_never_rule_breaks():
    gaps = ["The passages do not address limitation.", "  ", 7, "Nor costs."]
    result = run(BASE, [], gaps=gaps)
    assert result.gaps == ["The passages do not address limitation.", "Nor costs."]
    assert result.rule_breaks == []
    assert result.verified == [] and result.unverified == []


def test_multiple_sentences_keep_order_and_indexes():
    good = {"sentence": "Good.", "citations": [{"label": "P1", "quote": P1_QUOTE}]}
    bad = {"sentence": "Bad.", "citations": []}
    result = run(BASE, [bad, good, bad])
    assert [s.sentence for s in result.verified] == ["Good."]
    assert [s.sentence for s in result.unverified] == ["Bad.", "Bad."]
    assert [b.sentence_index for b in result.rule_breaks] == [0, 2]
    assert all(b.kind == "no_citation" for b in result.rule_breaks)


@pytest.mark.parametrize(
    "llm_output",
    [
        {},
        {"sentences": None, "gaps": None},
        {"sentences": "not a list", "gaps": "not a list"},
    ],
)
def test_malformed_top_level_gives_nothing(llm_output):
    label_map = {label: make_ref(p) for label, p in BASE.items()}
    chunks = {p["chunk_id"]: make_chunk(p) for p in BASE.values()}
    result = check(
        llm_output,
        label_map,
        chunks,
        SETTINGS,
        find_citations=fake_find_citations,
        resolve_alias=fake_resolve_alias,
    )
    assert result == CheckResult([], [], [], [])


def test_malformed_sentences_are_unverified_never_verified():
    sentences = [
        "a bare string",
        {"citations": [{"label": "P1", "quote": P1_QUOTE}]},  # no sentence text
        {"sentence": "  ", "citations": [{"label": "P1", "quote": P1_QUOTE}]},
        {"sentence": "Odd citations.", "citations": "P1"},
        {"sentence": "Odd items.", "citations": ["P1", {"label": 3, "quote": None}]},
        {"sentence": "Missing quote.", "citations": [{"label": "P1"}]},
    ]
    result = run(BASE, sentences)
    assert result.verified == []
    reasons = [(u.sentence, u.reason.value, u.labels) for u in result.unverified]
    assert reasons == [
        ("Odd citations.", "no_citation", []),
        ("Odd items.", "unknown_label", ["3"]),
        ("Missing quote.", "quote_length", ["P1"]),
    ]
    assert [b.kind for b in result.rule_breaks].count("malformed_sentence") == 3


def test_label_whose_chunk_is_missing_is_unknown():
    label_map = {"P1": make_ref(BASE["P1"])}
    sent = {"sentence": "x", "citations": [{"label": "P1", "quote": P1_QUOTE}]}
    result = check(
        {"sentences": [sent], "gaps": []},
        label_map,
        {},
        SETTINGS,
        find_citations=fake_find_citations,
        resolve_alias=fake_resolve_alias,
    )
    assert result.unverified[0].reason is UnverifiedReason.UNKNOWN_LABEL


def test_label_whitespace_is_tolerated():
    sent = {"sentence": "x", "citations": [{"label": " P1 ", "quote": P1_QUOTE}]}
    result = run(BASE, [sent])
    assert result.verified[0].citations[0].passage_label == "P1"


def test_verified_citation_carries_passage_ref_fields():
    sent = {"sentence": "x", "citations": [{"label": "P1", "quote": P1_QUOTE}]}
    cit = run(BASE, [sent]).verified[0].citations[0]
    ref = make_ref(BASE["P1"])
    assert cit.model_dump(exclude={"passage_label", "quote"}) == ref.model_dump()
    assert cit.quote == P1_QUOTE


def test_default_finder_requires_link_module(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nyaya.link.aliases":
            raise ImportError("not built yet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from nyaya.answer import citation_check

    with pytest.raises(RuntimeError, match="find_citations"):
        citation_check.default_find_citations()
