"""Router (safety-critical): rules only, no LLM.

A question with an SCR, neutral or CNR citation takes the citation_lookup
route; anything else, including SCC and AIR citations (which have no
aliases), takes the hybrid route.
"""

import pytest

from nyaya.contracts import Route
from nyaya.search.router import (
    CitationMatch,
    find_lookup_citations,
    lookup_canonicals,
    route,
)

L = Route.CITATION_LOOKUP
H = Route.HYBRID

LABELLED = [
    # --- citation lookup: neutral citations ---
    ("2025 INSC 31", L),
    ("2025INSC31", L),
    ("What did 2025 INSC 719 decide?", L),
    ("summarise 2024 insc 5 please", L),
    ("2023  INSC  1002", L),
    # --- citation lookup: SCR ---
    ("[2025] 1 S.C.R. 230", L),
    ("[2025] 1 SCR 230", L),
    ("Explain [2019] 3 S. C. R. 101 on bail", L),
    ("2025 (1) SCR 230", L),
    ("2025(1)SCR230", L),
    ("[2006] SUPP. 6 S.C.R. 72", L),
    ("[2006] Supp 6 SCR 72", L),
    ("[2010] 12 S.C.R 400 and its ratio", L),
    # --- citation lookup: CNR ---
    ("ESCR010004242025", L),
    ("case ESCR010004242025 details", L),
    ("escr010004242025", L),
    # --- hybrid: plain questions ---
    ("anticipatory bail section 438", H),
    ("What is the test for anticipatory bail under section 438 CrPC?", H),
    ("When can a High Court quash an FIR under Section 482?", H),
    ("Article 21 right to privacy", H),
    ("dowry death presumption under section 113B Evidence Act", H),
    ("judgments from 2019 on arbitration", H),
    ("cases decided in 2025 about the NDPS Act 1985", H),
    ("limitation period of 3 years for suits", H),
    ("Order 7 Rule 11 CPC rejection of plaint", H),
    ("section 302 IPC (murder) sentencing in 2018", H),
    ("", H),
    ("   ", H),
    # --- hybrid: citations that have no alias table ---
    ("(2009) 8 SCC 751", H),
    ("what did AIR 1999 SC 3750 say", H),
    ("[2025] 1 SCC 230", H),
    ("2025 (1) SCC 230", H),
    # --- hybrid: near misses ---
    ("2025 INSC", H),
    ("INSC 31", H),
    ("[2025] S.C.R. 230", H),
    ("2025 (1) SCRA 230", H),
    ("ESCR01000424202", H),  # 11 digits
    ("ESCR0100042420251", H),  # 13 digits
    ("ESC010004242025", H),  # 3 letters
    ("12345 INSC 3", H),  # five-digit year is not a year
]


def test_at_least_30_labelled_questions():
    assert len(LABELLED) >= 30


@pytest.mark.parametrize(("question", "expected"), LABELLED)
def test_route(question, expected):
    assert route(question) is expected


def test_anticipatory_bail_routes_hybrid():
    assert route("anticipatory bail section 438") is Route.HYBRID


@pytest.mark.parametrize(
    ("text", "canonical", "kind"),
    [
        ("2025 INSC 31", "insc:2025:31", "neutral"),
        ("2025INSC31", "insc:2025:31", "neutral"),
        ("2025 INSC 031", "insc:2025:31", "neutral"),
        ("[2025] 1 S.C.R. 230", "scr:2025:1:230", "scr"),
        ("2025 (1) SCR 230", "scr:2025:1:230", "scr"),
        ("[2006] SUPP. 6 S.C.R. 72", "scr:2006:s6:72", "scr"),
        ("ESCR010004242025", "cnr:escr010004242025", "cnr"),
        ("escr010004242025", "cnr:escr010004242025", "cnr"),
    ],
)
def test_canonical_forms(text, canonical, kind):
    matches = find_lookup_citations(text)
    assert len(matches) == 1
    assert matches[0].canonical == canonical
    assert matches[0].kind == kind
    start, end = matches[0].span
    assert text[start:end].strip() == text.strip()


def test_multiple_citations_in_text_order_and_deduplicated():
    q = "Compare ESCR010004242025 with 2025 INSC 31 and [2025] 1 SCR 230 and 2025INSC31"
    canon = lookup_canonicals(q)
    assert canon == ["cnr:escr010004242025", "insc:2025:31", "scr:2025:1:230"]
    spans = [m.span for m in find_lookup_citations(q)]
    assert spans == sorted(spans)


def test_no_citations_gives_empty_list():
    assert find_lookup_citations("anticipatory bail section 438") == []
    assert lookup_canonicals("") == []


def test_route_uses_injected_finder():
    def finder(_q):
        return [CitationMatch(span=(0, 1), kind="neutral", canonical="insc:1:1")]

    assert route("anything", finder=finder) is Route.CITATION_LOOKUP
    assert route("2025 INSC 31", finder=lambda _q: []) is Route.HYBRID


def test_injected_finder_kinds_outside_lookup_are_ignored():
    def finder(_q):
        return [
            CitationMatch(span=(0, 1), kind="scc", canonical="scc:2009:8:751"),
            CitationMatch(span=(2, 3), kind="air", canonical="air:1999:3750"),
        ]

    assert route("x", finder=finder) is Route.HYBRID


def test_route_rejects_non_string():
    with pytest.raises(TypeError):
        route(None)  # type: ignore[arg-type]
