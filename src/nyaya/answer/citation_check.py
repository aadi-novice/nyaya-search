"""Citation check: code, not the LLM, decides which sentences are shown (safety-critical).

Design, "Citation check". For each citation on a sentence, in order:

1. the label exists (and names a passage whose chunk we hold);
2. the quote is `quote.min_words`-`quote.max_words` words;
3. after `normalize_text()` on both, the quote is a substring of the chunk's
   `text` (never its header).

Separately (step 4), the sentence is scanned:

* for SCR, INSC, SCC and AIR citations (`find_citations`, link/aliases.py). A
  citation whose canonical form does not resolve to one of the passages'
  judgments is a rule break. CNRs are not checked (the design names only the
  four reporter forms);
* for "X v. Y" case names. A name matches a passage when both sides, after
  `normalize_text()`, are contained in that passage's normalised title
  (`PassageRef.case_name`). A name matching no passage is a rule break.
  Party names alone are not checked.

Quote snapping (`quote.snap`, decision 2026-11-18): when step 3 fails, the
quote may be replaced by the exact span of the same chunk's text that it copies
with small edits (answer/quote_snap.py). A snapped citation passes and carries
the span, never the LLM's quote, so every quote shown is still a substring of
the cited chunk. The failure is still a `quote_not_found` rule break, with
`snap` set in the trace.

Each `quote_length` or `quote_not_found` rule break keeps the LLM's quote in
`detail`, so a saved run can be re-checked offline (nyaya-eval replay-check).

A sentence is verified when at least one citation passes steps 1-3 and step 4
passes. Only passing citations are kept on it. Otherwise it is unverified,
with the earliest failing step as its reason. A step-4 citation-pattern break
has no reason of its own on the wire, so it reports `uncited_case_name` (the
sentence names a judgment that is not among the passages).

Every failure is recorded as a rule break before filtering, including the
failed citations of a sentence that is verified through another citation.
`gaps` are passed through and are never rule breaks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import regex

from nyaya.answer.quote_snap import Snap, snap_quote
from nyaya.config import Settings
from nyaya.contracts import (
    AnswerSentence,
    ChunkRecord,
    Citation,
    PassageRef,
    UnverifiedReason,
    UnverifiedSentence,
)
from nyaya.text.normalize import normalize_text

# (span, kind, canonical) for each citation found in a text (link/aliases.py).
FindCitations = Callable[[str], Iterable[tuple[Any, str, str]]]
# canonical citation -> judgment path, or None (SearchEngine.resolve_alias).
ResolveAlias = Callable[[str], str | None]

# The design's regex. `_NAME_CHARS` holds a curly apostrophe (U+2019).
_NAME_CHARS = r"[\w.&'" + "\u2019" + r"()-]*"
_NAME = rf"([A-Z]{_NAME_CHARS}(?:\s+[A-Z&]{_NAME_CHARS}){{0,8}})"
CASE_NAME_RE = regex.compile(rf"{_NAME}\s+(?:v\.|vs\.?|versus)\s+{_NAME}")

# Step order: a lower number is an earlier step and wins as the reason.
_STEP = {
    UnverifiedReason.UNKNOWN_LABEL: 1,
    UnverifiedReason.QUOTE_LENGTH: 2,
    UnverifiedReason.QUOTE_NOT_FOUND: 3,
}
_UNCHECKED_KINDS = frozenset({"cnr"})


@dataclass(frozen=True)
class RuleBreak:
    """One failure, counted in the trace before filtering."""

    sentence_index: int
    # unknown_label | quote_length | quote_not_found | no_citation |
    # uncited_case_name | uncited_citation | malformed_sentence
    kind: str
    label: str | None = None
    detail: str | None = None  # the quote, for quote_length and quote_not_found
    snap: dict[str, Any] | None = None  # set when the quote was snapped

    def to_dict(self) -> dict[str, Any]:
        """The trace form; `snap` appears only on a snapped citation."""
        d = asdict(self)
        if d["snap"] is None:
            del d["snap"]
        return d


class CheckResult(NamedTuple):
    verified: list[AnswerSentence]
    unverified: list[UnverifiedSentence]
    gaps: list[str]
    rule_breaks: list[RuleBreak]


def default_find_citations() -> FindCitations:
    """`find_citations` from link/aliases.py (file 24)."""
    try:
        from nyaya.link.aliases import find_citations
    except ImportError as exc:  # built in parallel; must exist before use
        raise RuntimeError("citation check needs find_citations from nyaya.link.aliases") from exc
    return find_citations


def _word_count(quote: str) -> int:
    return len(normalize_text(quote).split())


def _length_ok(quote: str, settings: Settings) -> bool:
    return settings.quote.min_words <= _word_count(quote) <= settings.quote.max_words


def _snap(quote: str, text: str, settings: Settings) -> Snap | None:
    """A span of `text` to show instead of `quote`, if snapping is on and one qualifies."""
    if not settings.quote.snap:
        return None
    snap = snap_quote(
        quote,
        text,
        min_similarity=settings.quote.snap_min_similarity,
        max_edits=settings.quote.snap_max_edits,
    )
    # Re-check the span with the same rules as any quote: belt and braces.
    if snap is None or not _length_ok(snap.span, settings):
        return None
    if normalize_text(snap.span) not in normalize_text(text):  # pragma: no cover - by construction
        return None
    return snap


def _citation_failure(
    label: str,
    quote: str,
    label_map: Mapping[str, PassageRef],
    chunks: Mapping[str, ChunkRecord],
    settings: Settings,
) -> tuple[UnverifiedReason | None, Snap | None]:
    """The first failing step (1-3) for one citation (None if it passes), and any snap.

    A snap is returned only with QUOTE_NOT_FOUND: the citation then passes with
    the snapped span as its quote.
    """
    ref = label_map.get(label)
    if ref is None or ref.chunk_id not in chunks:
        return UnverifiedReason.UNKNOWN_LABEL, None
    if not _length_ok(quote, settings):
        return UnverifiedReason.QUOTE_LENGTH, None
    text = chunks[ref.chunk_id].text
    if normalize_text(quote) not in normalize_text(text):
        return UnverifiedReason.QUOTE_NOT_FOUND, _snap(quote, text, settings)
    return None, None


def _sentence_breaks(
    index: int,
    sentence: str,
    label_map: Mapping[str, PassageRef],
    find_citations: FindCitations,
    resolve_alias: ResolveAlias,
) -> list[RuleBreak]:
    """Step 4: citation patterns and case names that match no passage."""
    breaks: list[RuleBreak] = []
    judgment_ids = {ref.judgment_id for ref in label_map.values()}
    for _span, kind, canonical in find_citations(sentence):
        if kind in _UNCHECKED_KINDS or canonical.startswith("cnr:"):
            continue
        if resolve_alias(canonical) not in judgment_ids:
            breaks.append(RuleBreak(index, "uncited_citation", detail=canonical))
    titles = [normalize_text(ref.case_name) for ref in label_map.values()]
    for m in CASE_NAME_RE.finditer(sentence):
        left, right = normalize_text(m.group(1)), normalize_text(m.group(2))
        if not any(left in t and right in t for t in titles):
            breaks.append(RuleBreak(index, "uncited_case_name", detail=m.group(0)))
    return breaks


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def check(
    llm_output: Mapping[str, Any],
    label_map: Mapping[str, PassageRef],
    chunks: Mapping[str, ChunkRecord],
    settings: Settings,
    *,
    resolve_alias: ResolveAlias,
    find_citations: FindCitations | None = None,
) -> CheckResult:
    """Split the LLM's sentences into verified and unverified.

    `llm_output` is `{"sentences": [{sentence, citations: [{label, quote}]}],
    "gaps": [str]}`; it is read defensively, and anything malformed can only
    make a sentence unverified. `label_map` maps "P1".. to the passage shown
    under that label, and `chunks` maps chunk_id to the chunk (the quote check
    runs on its `text`).
    """
    finder = find_citations if find_citations is not None else default_find_citations()
    verified: list[AnswerSentence] = []
    unverified: list[UnverifiedSentence] = []
    breaks: list[RuleBreak] = []

    for index, item in enumerate(_as_list(llm_output.get("sentences"))):
        sentence = item.get("sentence") if isinstance(item, dict) else None
        if not isinstance(sentence, str) or not sentence.strip():
            breaks.append(RuleBreak(index, "malformed_sentence"))
            continue
        raw_cits = [c for c in _as_list(item.get("citations")) if isinstance(c, dict)]
        labels = [str(c.get("label", "")).strip() for c in raw_cits]

        passed: list[Citation] = []
        failures: list[UnverifiedReason] = []
        for label, raw in zip(labels, raw_cits, strict=True):
            quote = raw.get("quote")
            quote = quote if isinstance(quote, str) else ""
            failure, snap = _citation_failure(label, quote, label_map, chunks, settings)
            if failure is None or snap is not None:
                ref = label_map[label]
                shown = quote if snap is None else snap.span
                passed.append(Citation(**ref.model_dump(), passage_label=label, quote=shown))
            if failure is None:
                continue
            if snap is None:
                failures.append(failure)
            detail = quote if failure is not UnverifiedReason.UNKNOWN_LABEL else None
            breaks.append(
                RuleBreak(
                    index,
                    failure.value,
                    label=label,
                    detail=detail,
                    snap=snap.to_dict() if snap is not None else None,
                )
            )

        step4 = _sentence_breaks(index, sentence, label_map, finder, resolve_alias)
        breaks.extend(step4)

        if passed and not step4:
            verified.append(AnswerSentence(sentence=sentence, citations=passed))
            continue
        if not raw_cits:
            reason = UnverifiedReason.NO_CITATION
            breaks.append(RuleBreak(index, reason.value))
        elif not passed:
            reason = min(failures, key=_STEP.__getitem__)
        else:
            reason = UnverifiedReason.UNCITED_CASE_NAME
        unverified.append(UnverifiedSentence(sentence=sentence, labels=labels, reason=reason))

    gaps = [g.strip() for g in _as_list(llm_output.get("gaps")) if isinstance(g, str) and g.strip()]
    return CheckResult(verified, unverified, gaps, breaks)
