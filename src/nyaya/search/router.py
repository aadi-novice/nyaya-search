"""Query router (safety-critical): rules only, no LLM.

A question containing an SCR, neutral-citation or CNR pattern takes the
`citation_lookup` route; anything else takes the `hybrid` route. SCC and AIR
citations are recognised by the linker but have no alias rows (the metadata
holds no SCC or AIR fields), so they never trigger a lookup.

The patterns are the ones in design file 24 (`link/aliases.py`). Until that
module lands, this file carries its own copy for the three lookup kinds; the
`finder` argument of `route()` lets the pipeline pass
`nyaya.link.aliases.find_citations` instead, so both agree on one pattern set.
Matching is case-insensitive (people type "2025 insc 31"); canonical strings
are lower case.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nyaya.contracts import Route

LOOKUP_KINDS = frozenset({"scr", "neutral", "cnr"})


@dataclass(frozen=True)
class CitationMatch:
    span: tuple[int, int]
    kind: str  # scr | neutral | cnr (| scc | air from the linker's finder)
    canonical: str


Finder = Callable[[str], Sequence[CitationMatch]]

# [2025] 1 S.C.R. 230   |   [2006] SUPP. 6 S.C.R. 72
_SCR_BRACKET = re.compile(
    r"\[([0-9]{4})\]\s*(SUPP\.?\s*)?([0-9]+)\s*S\.?\s*C\.?\s*R\.?\s*([0-9]+)", re.IGNORECASE
)
# 2025 (1) SCR 230
_SCR_PAREN = re.compile(r"(?<![0-9])([0-9]{4})\s*\(([0-9]+)\)\s*SCR\s*([0-9]+)", re.IGNORECASE)
# 2025 INSC 31   |   2025INSC31
_NEUTRAL = re.compile(r"(?<![0-9])([0-9]{4})\s*INSC\s*([0-9]+)", re.IGNORECASE)
# ESCR010004242025
_CNR = re.compile(r"\b[A-Z]{4}[0-9]{12}\b", re.IGNORECASE)


def find_lookup_citations(text: str) -> list[CitationMatch]:
    """SCR, neutral and CNR citations in `text`, in text order, one per canonical form."""
    found: list[CitationMatch] = []
    for m in _SCR_BRACKET.finditer(text):
        year, supp, vol, page = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        volume = f"s{vol}" if supp else str(vol)
        found.append(CitationMatch(m.span(), "scr", f"scr:{year}:{volume}:{page}"))
    for m in _SCR_PAREN.finditer(text):
        year, vol, page = m.group(1), int(m.group(2)), int(m.group(3))
        found.append(CitationMatch(m.span(), "scr", f"scr:{year}:{vol}:{page}"))
    for m in _NEUTRAL.finditer(text):
        found.append(CitationMatch(m.span(), "neutral", f"insc:{m.group(1)}:{int(m.group(2))}"))
    for m in _CNR.finditer(text):
        found.append(CitationMatch(m.span(), "cnr", f"cnr:{m.group(0).lower()}"))
    found.sort(key=lambda c: c.span)
    seen: set[str] = set()
    out: list[CitationMatch] = []
    for c in found:
        if c.canonical not in seen:
            seen.add(c.canonical)
            out.append(c)
    return out


def lookup_canonicals(text: str, finder: Finder = find_lookup_citations) -> list[str]:
    """Canonical strings of the lookup-kind citations in `text`, for `resolve_alias`."""
    return [c.canonical for c in finder(text) if c.kind in LOOKUP_KINDS]


def route(question: str, finder: Finder = find_lookup_citations) -> Route:
    """CITATION_LOOKUP if the question holds an SCR, neutral or CNR citation, else HYBRID."""
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")
    if lookup_canonicals(question, finder):
        return Route.CITATION_LOOKUP
    return Route.HYBRID
