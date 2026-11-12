"""Query router (safety-critical): rules only, no LLM.

A question containing an SCR, neutral-citation or CNR pattern takes the
`citation_lookup` route; anything else takes the `hybrid` route. SCC and AIR
citations are recognised by the linker but have no alias rows (the metadata
holds no SCC or AIR fields), so they never trigger a lookup.

There is one pattern set: `nyaya.link.aliases.find_citations` (design file
24), which is the default finder here and the one the pipeline passes. Its
matching is case-insensitive (people type "2025 insc 31") and its canonical
strings are lower case, so the router, the alias table and the citation check
always agree. The `finder` argument exists so tests can inject one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from nyaya.contracts import Route
from nyaya.link.aliases import find_citations

LOOKUP_KINDS = frozenset({"scr", "neutral", "cnr"})


class CitationMatch(NamedTuple):
    """One citation, in the `(span, kind, canonical)` shape of `find_citations`."""

    span: tuple[int, int]
    kind: str  # scr | neutral | cnr | scc | air
    canonical: str


Finder = Callable[[str], Sequence[tuple[tuple[int, int], str, str]]]


def find_lookup_citations(text: str, finder: Finder = find_citations) -> list[CitationMatch]:
    """SCR, neutral and CNR citations in `text`, in text order, one per canonical form."""
    out: list[CitationMatch] = []
    seen: set[str] = set()
    for span, kind, canonical in sorted(finder(text), key=lambda f: tuple(f[0])):
        if kind in LOOKUP_KINDS and canonical not in seen:
            seen.add(canonical)
            out.append(CitationMatch(tuple(span), kind, canonical))  # type: ignore[arg-type]
    return out


def lookup_canonicals(text: str, finder: Finder = find_citations) -> list[str]:
    """Canonical strings of the lookup-kind citations in `text`, for `resolve_alias`."""
    return [c.canonical for c in find_lookup_citations(text, finder)]


def route(question: str, finder: Finder = find_citations) -> Route:
    """CITATION_LOOKUP if the question holds an SCR, neutral or CNR citation, else HYBRID."""
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")
    if lookup_canonicals(question, finder):
        return Route.CITATION_LOOKUP
    return Route.HYBRID
