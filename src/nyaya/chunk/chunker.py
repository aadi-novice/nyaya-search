"""Chunk a zoned judgment (design, Technical Plan step 5; file 22).

`chunk(meta, parsed, zones, opinions, settings) -> [ChunkRecord]`:

*   one `issue` chunk (Era D), then one `editorial` chunk per Held point;
*   `judgment` chunks: numbered paragraphs within one opinion, merged up to
    `chunk.target_words`; a paragraph over `chunk.max_words` is split at
    sentence boundaries with one sentence of overlap (`recursive_split`); a
    chunk under `chunk.min_words` is merged into a neighbour in the same
    opinion when the result stays within `chunk.max_words`;
*   an opinion with no paragraph numbers falls back to blank-line paragraphs,
    then to sentence windows (`unnumbered`).

Chunk text is always the exact slice clean_text[char_start:char_end], so the
quote check and page lookups work on offsets. The header is built from
metadata only.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nyaya.config import Settings
from nyaya.contracts import (
    ChunkRecord,
    Opinion,
    OpinionType,
    ParsedDoc,
    Zone,
    Zones,
)
from nyaya.zone.zoner import canonical_act, default_settings, load_act_aliases

ZONE_LABELS = {
    Zone.JUDGMENT: "Judgment",
    Zone.EDITORIAL: "Headnote (editorial)",
    Zone.ISSUE: "Issue",
}
OPINION_SUFFIX = {OpinionType.DISSENTING: " (dissenting)", OpinionType.CONCURRING: " (concurring)"}
MAX_HEADER_ACTS = 3

# A numbered paragraph starts a line: "12. ", "12.1 ", "1.\n". On the first line
# of an opinion the author may come first: "DEEPAK GUPTA, J. 1. ...".
_PARA = re.compile(
    r"^[ \t]*(?:[A-Z][A-Za-z. \t]{2,60}?,[ \t]*(?:CJI|C\.J|J)\.?[ \t]*:?[ \t]+"
    r"(?:[-\u2013\u2014][ \t]+)?)?"
    r"(\d{1,3})(?:\.(\d{1,2}))?(\.?)(?=\s)",
    re.MULTILINE,
)
_MAX_JUMP = 10
# A sub-heading in capitals on its own line ("FACTUAL ASPECT", "ANALYSIS").
_HEADING = re.compile(r"^[ \t]*([A-Z][A-Z0-9 .,:;'&()/\-]{2,80}?)[ \t]*$", re.MULTILINE)
_NOT_HEADING = re.compile(r",\s*(?:J|JJ|CJI|C\.J)\.?\s*:?$|^[\dIVXL.\s]+$")


def count_words(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Paragraph:
    label: str | None
    start: int
    end: int


# --- sentences -------------------------------------------------------------------

_SENT_END = re.compile(r"[.?!][\"'\u201d\u2019)\]]*\s+(?=[\"'\u201c\u2018(\[]?[A-Z0-9])")


def split_sentences(
    text: str, start: int, end: int, abbreviations: list[str]
) -> list[tuple[int, int]]:
    """Sentence spans in text[start:end], ignoring `chunk.abbreviations` and initials."""
    abbr = {a.lower() for a in abbreviations}
    spans: list[tuple[int, int]] = []
    s = start
    for m in _SENT_END.finditer(text, start, end):
        word = text[s : m.start() + 1].split()
        last = word[-1].lstrip("(\"'\u201c\u2018[").split("/")[-1] if word else ""
        if last.lower() in abbr or re.fullmatch(r"[A-Z]\.(?:[A-Z]\.)*", last):
            continue
        spans.append((s, m.end()))
        s = m.end()
    if s < end and text[s:end].strip():
        spans.append((s, end))
    elif spans:
        spans[-1] = (spans[-1][0], end)
    return [_strip(text, a, b) for a, b in spans if text[a:b].strip()]


# --- paragraphs ------------------------------------------------------------------


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def find_paragraphs(text: str, start: int, end: int) -> list[Paragraph]:
    """Numbered paragraphs in text[start:end], in order; [] if none.

    Labels must run in order ("12" -> "13" or "12.1"); a whole-number label
    needs its full stop (a wrapped "13 of the Act" is not a paragraph), and
    may skip ahead by at most 10 (a paragraph the parser lost). Text before
    the first paragraph joins it.
    """
    found: list[tuple[int, str]] = []
    last: tuple[int, int] | None = None
    for m in _PARA.finditer(text, start, end):
        ma, mi, dot = int(m.group(1)), int(m.group(2) or 0), bool(m.group(3))
        if not mi and not dot:
            continue
        if last is None:
            ok = mi in (0, 1)
        else:
            lma, lmi = last
            ok = (
                (ma == lma + 1 and mi in (0, 1))
                or (ma == lma and mi == lmi + 1)
                or (dot and lma + 1 < ma <= lma + _MAX_JUMP and mi in (0, 1))
            )
        if ok:
            found.append((m.start(), f"{ma}.{mi}" if mi else str(ma)))
            last = (ma, mi)
    if not found:
        return []
    paras = []
    for i, (s, label) in enumerate(found):
        e = found[i + 1][0] if i + 1 < len(found) else end
        if i == 0:
            s = start
        a, b = _strip(text, s, e)
        if a < b:
            paras.append(Paragraph(label, a, b))
    return paras


def _unnumbered(text: str, start: int, end: int, settings: Settings) -> list[Paragraph]:
    blocks = []
    pos = start
    for m in re.finditer(r"\n[ \t]*\n", text[start:end]):
        blocks.append((pos, start + m.start()))
        pos = start + m.end()
    blocks.append((pos, end))
    blocks = [_strip(text, a, b) for a, b in blocks]
    blocks = [(a, b) for a, b in blocks if a < b]
    if len(blocks) > 1:
        return [Paragraph(None, a, b) for a, b in blocks]
    if not blocks:
        return []
    # one block: sentence windows of about target_words
    out = []
    cur: list[tuple[int, int]] = []
    words = 0
    for a, b in split_sentences(text, *blocks[0], settings.chunk.abbreviations):
        w = count_words(text[a:b])
        if cur and words + w > settings.chunk.target_words:
            out.append(Paragraph(None, cur[0][0], cur[-1][1]))
            cur, words = [], 0
        cur.append((a, b))
        words += w
    if cur:
        out.append(Paragraph(None, cur[0][0], cur[-1][1]))
    return out


# --- packing ---------------------------------------------------------------------


@dataclass
class _Unit:
    labels: list[str]
    start: int
    end: int
    words: int
    split_of: int | None = None  # index of the paragraph this piece was split from
    flags: list[str] = field(default_factory=list)


def _split_long(text: str, para: Paragraph, idx: int, settings: Settings) -> list[_Unit]:
    target, max_w = settings.chunk.target_words, settings.chunk.max_words
    sents = split_sentences(text, para.start, para.end, settings.chunk.abbreviations)
    # hard-split any single sentence over the maximum at word boundaries
    pieces: list[tuple[int, int]] = []
    for a, b in sents:
        if count_words(text[a:b]) <= max_w:
            pieces.append((a, b))
            continue
        words = list(re.finditer(r"\S+", text[a:b]))
        for k in range(0, len(words), target):
            grp = words[k : k + target]
            pieces.append((a + grp[0].start(), a + grp[-1].end()))
    labels = [para.label] if para.label else []
    units: list[_Unit] = []
    i = 0
    while i < len(pieces):
        cur = [pieces[i]]
        if units and i > 0:
            cur = [pieces[i - 1], pieces[i]]  # one sentence of overlap
            if count_words(text[cur[0][0] : cur[-1][1]]) > max_w:
                cur = [pieces[i]]
        i += 1
        while i < len(pieces) and count_words(text[cur[0][0] : pieces[i][1]]) <= target:
            cur.append(pieces[i])
            i += 1
        s, e = cur[0][0], cur[-1][1]
        units.append(_Unit(list(labels), s, e, count_words(text[s:e]), idx, ["recursive_split"]))
    return units


def _pack(text: str, paras: list[Paragraph], settings: Settings, flags: list[str]) -> list[_Unit]:
    target, min_w, max_w = (
        settings.chunk.target_words,
        settings.chunk.min_words,
        settings.chunk.max_words,
    )
    units: list[_Unit] = []
    for idx, p in enumerate(paras):
        w = count_words(text[p.start : p.end])
        if w > max_w:
            units.extend(_split_long(text, p, idx, settings))
        else:
            units.append(_Unit([p.label] if p.label else [], p.start, p.end, w, None, []))

    chunks: list[_Unit] = []
    for u in units:
        cur = chunks[-1] if chunks else None
        same_split = cur is not None and u.split_of is not None and cur.split_of == u.split_of
        if cur is None or same_split or cur.words + u.words > target:
            chunks.append(_Unit(list(u.labels), u.start, u.end, u.words, u.split_of, list(u.flags)))
        else:
            _absorb(text, cur, u)

    changed = True
    while changed and len(chunks) > 1:
        changed = False
        for i, c in enumerate(chunks):
            if c.words >= min_w:
                continue
            prev = chunks[i - 1] if i > 0 else None
            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            if prev and count_words(text[prev.start : c.end]) <= max_w:
                _absorb(text, prev, c)
                del chunks[i]
            elif nxt and count_words(text[c.start : nxt.end]) <= max_w:
                _absorb(text, c, nxt)
                del chunks[i + 1]
            else:
                continue
            changed = True
            break
    for c in chunks:
        for f in flags:
            if f not in c.flags:
                c.flags.append(f)
    return chunks


def _absorb(text: str, into: _Unit, other: _Unit) -> None:
    into.start = min(into.start, other.start)
    into.end = max(into.end, other.end)
    for lab in other.labels:
        if lab not in into.labels:
            into.labels.append(lab)
    for f in other.flags:
        if f not in into.flags:
            into.flags.append(f)
    into.words = count_words(text[into.start : into.end])
    into.split_of = other.split_of


# --- fields ----------------------------------------------------------------------


def _meta(meta: Any, key: str) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key)
    return getattr(meta, key, None)


def _decision_year(meta: Any) -> int | None:
    year = _meta(meta, "decision_year")
    if year:
        return int(year)
    d = _meta(meta, "decision_date")
    if d is None:
        return None
    if hasattr(d, "year"):
        return int(d.year)
    m = re.search(r"(\d{4})", str(d))
    return int(m.group(1)) if m else None


def display_title(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(re.sub(r"\s+versus\s+", " v. ", title, flags=re.IGNORECASE).split())


def make_header(
    title: str,
    year: int | None,
    zone: Zone,
    author: str | None,
    opinion_type: OpinionType | None,
    para_labels: list[str],
    acts: list[str],
) -> str:
    """`{title} ({year}) · {zone_label} · {author}{suffix} · paras {a}\u2013{b} · {acts}`."""
    parts = []
    head = title
    if year:
        head = f"{title} ({year})" if title else f"({year})"
    if head:
        parts.append(head)
    parts.append(ZONE_LABELS[zone])
    if author:
        parts.append(author + OPINION_SUFFIX.get(opinion_type, ""))  # type: ignore[arg-type]
    if para_labels:
        parts.append(f"paras {para_labels[0]}\u2013{para_labels[-1]}")
    if acts:
        parts.append("; ".join(acts[:MAX_HEADER_ACTS]))
    return " · ".join(parts)


def pages_for(position_map: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Printed pages of the lines that text[start:end] touches."""
    if not position_map:
        return []
    offsets = [o for o, _ in position_map]
    i = max(0, bisect.bisect_right(offsets, start) - 1)
    j = max(i, bisect.bisect_right(offsets, max(start, end - 1)) - 1)
    out: list[int] = []
    for _, page in position_map[i : j + 1]:
        if page not in out:
            out.append(page)
    return out


def _headings(text: str, start: int, end: int) -> list[tuple[int, str]]:
    out = []
    for m in _HEADING.finditer(text, start, end):
        label = m.group(1).strip()
        if sum(c.isalpha() for c in label) < 3 or _NOT_HEADING.search(label):
            continue
        out.append((m.start(), label))
    return out


def _section_label(headings: list[tuple[int, str]], first_para_line: int) -> str | None:
    label = None
    for pos, h in headings:
        if pos < first_para_line:
            label = h
        else:
            break
    return label


def _first_para_line(text: str, start: int, end: int) -> int:
    m = _PARA.search(text, start, end)
    return m.start() if m else start


# --- chunk -----------------------------------------------------------------------


def chunk(
    meta: Any,
    parsed: ParsedDoc,
    zones: Zones,
    opinions: list[Opinion],
    settings: Settings | None = None,
    aliases: Mapping[str, str] | None = None,
) -> list[ChunkRecord]:
    """Chunks of one judgment: issue, editorial, then judgment chunks per opinion.

    `meta` is a mapping or object with `title` and `decision_year` (or
    `decision_date`), such as a manifest row or a JudgmentRecord.
    """
    settings = settings or default_settings()
    if aliases is None:
        aliases = load_act_aliases(settings)
    text = parsed.clean_text
    title = display_title(_meta(meta, "title"))
    year = _decision_year(meta)
    acts: list[str] = []
    for a in zones.acts:
        c = canonical_act(a, aliases)
        if c not in acts:
            acts.append(c)
    doc_flags = ["zone_missing"] if Zone.JUDGMENT in zones.missing else []
    seq = {Zone.ISSUE: 0, Zone.EDITORIAL: 0, Zone.JUDGMENT: 0}
    out: list[ChunkRecord] = []

    def emit(
        zone: Zone,
        start: int,
        end: int,
        *,
        author: str | None = None,
        opinion_type: OpinionType | None = None,
        para_labels: list[str] | None = None,
        para_refs: list[str] | None = None,
        section_label: str | None = None,
        flags: list[str] | None = None,
    ) -> None:
        start, end = _strip(text, start, end)
        if start >= end:
            return
        seq[zone] += 1
        labels = para_labels or []
        all_flags = list(dict.fromkeys([*(flags or []), *doc_flags]))
        out.append(
            ChunkRecord(
                chunk_id=f"{parsed.path}#{zone.value}#{seq[zone]:03d}",
                judgment_id=parsed.path,
                zone=zone,
                opinion_author=author,
                opinion_type=opinion_type,
                para_labels=labels,
                para_refs=para_refs or [],
                section_label=section_label,
                text=text[start:end],
                header=make_header(title, year, zone, author, opinion_type, labels, acts),
                char_start=start,
                char_end=end,
                pages=pages_for(parsed.position_map, start, end),
                flags=all_flags,  # type: ignore[arg-type]
                clean_text_version=parsed.clean_text_version,
                chunker_version=settings.versions.chunker,
            )
        )

    if zones.issue_span is not None:
        emit(Zone.ISSUE, *zones.issue_span)
    for hp in zones.editorial:
        emit(Zone.EDITORIAL, hp.char_start, hp.char_end, para_refs=list(hp.para_refs))

    spans = [(o.char_start, o.char_end, o.author, o.type) for o in opinions] or [
        (zones.judgment_span[0], zones.judgment_span[1], None, None)
    ]
    for s, e, author, kind in spans:
        paras = find_paragraphs(text, s, e)
        flags: list[str] = []
        if not paras:
            paras = _unnumbered(text, s, e, settings)
            flags = ["unnumbered"]
        headings = _headings(text, s, e)
        for u in _pack(text, paras, settings, flags):
            emit(
                Zone.JUDGMENT,
                u.start,
                u.end,
                author=author,
                opinion_type=kind,
                para_labels=u.labels,
                section_label=_section_label(headings, _first_para_line(text, u.start, u.end)),
                flags=u.flags,
            )
    return out
