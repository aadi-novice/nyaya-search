"""Coordinate-aware PDF parsing into a ParsedDoc (design file 19, step 3).

`parse(pdf_bytes, path, era, settings)` extracts words with their boxes
(PyMuPDF `get_text("words")`) and removes furniture by position before any
text is joined:

* **Margin letters** (Era C only): a one-character token that is an upper-case
  A-H, or one of its common OCR misreadings (`c`, `o`, `O`, `0`, `8`), whose
  box lies inside the calibrated left or right band (or which stands in a
  column of such tokens near the page edge) and which sits alone: no other
  word on its line is closer than `MIN_MARGIN_GAP` points.
* **Running heads and footers:** a line whose centre lies in the top or bottom
  band and that matches `running_head_patterns`, or whose letters repeat (with
  small OCR differences) on at least half of the judgment's pages after the
  first.
* **Page numbers:** inside those bands, a digits-only token equal to the
  page's printed number, or a line made only of digits.

Band units. Every band is a pair of fractions of the page: `bands.era_c.*_x`
of the page width, `bands.top_y`/`bottom_y` of the page height. The 2010-2023
volumes come as scans about 425 points wide and as born-digital 612-point
pages, so fractions keep one band usable for both. Era C needs calibrated
margin bands (`nyaya-build calibrate`); Era D has no margin letters and uses
the fixed head and foot bands `ERA_D_TOP_Y` and `ERA_D_BOTTOM_Y`, as does Era
C when `bands.top_y`/`bottom_y` are empty.

The remaining words are grouped into lines (PyMuPDF lines, merged when they
share a baseline, so table cells join into one line), ordered top to bottom.
A word split by a hyphen at a line end is rejoined when the next line starts
with a lower-case letter: the hyphen is kept when the joined form with a
hyphen appears elsewhere in the judgment ("Respondent-complainant"), and
dropped otherwise.

`clean_text` is the lines joined with "\\n". `position_map` holds
`(char_offset, printed_page)` for the start of every line, where the printed
page is the SCR start page in `path` plus the PDF page index.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pymupdf

from nyaya.config import ConfigError, Settings
from nyaya.contracts import Era, ParsedDoc

Band = tuple[float, float]

MARGIN_LETTERS = frozenset("ABCDEFGH")
# OCR misreadings of margin letters seen in the 2010-2017 scans (C, D, B).
MARGIN_OCR_VARIANTS = frozenset({"c", "o", "O", "0", "8"})
MARGIN_TOKENS = MARGIN_LETTERS | MARGIN_OCR_VARIANTS
MIN_MARGIN_GAP = 6.0  # points; body words sit 0-5 points from their neighbours
COLUMN_MIN = 4  # lone tokens in one column, counting itself
COLUMN_TOL = 0.012  # page-width fraction
OUTER_ZONE = 0.3  # page-width fraction from either edge

# 0.17 (was 0.115), 2026-11-13: the late-2023 "CASE DETAILS" volumes, classed
# Era D, print the running head at about 0.16 of the page height (as Era C
# does), and it leaked into 2023 Held text. Only head-like lines in the band
# are removed (patterns, repeats, page numbers), so body lines stay.
ERA_D_TOP_Y: Band = (0.0, 0.17)
ERA_D_BOTTOM_Y: Band = (0.94, 1.0)
DEFAULT_TOP_Y: Band = (0.0, 0.17)
DEFAULT_BOTTOM_Y: Band = (0.94, 1.0)

REPEAT_SIMILARITY = 0.85
# Letters a line needs before repetition can mark it as a head: shorter keys
# ("scr", "supra", "scc") repeat in citation lists and footnotes.
MIN_REPEAT_KEY = 8
# The volume head "[2015] 2 S.C.R." (with or without the page number), which
# has too few letters for the repetition rule. Body citations print "SCR".
SCR_HEAD_RE = re.compile(
    r"^[\[(]\d{4}[\])]\s*(?:SUPP\.?\s*)?\d+\s*S\.\s?C\.\s?R\.(?:\s*\d+)?$", re.I
)

_PATH_PAGES_RE = re.compile(r"_(\d+)_(\d+)$")
DIGITS_RE = re.compile(r"^\d+$")
PARA_START_RE = re.compile(r"^\s*\d{1,3}(?:\.\d{1,2})*\.\s")
CONTINUATION_MAX_WIDTH = 0.6  # page-width fraction; full body lines are wider on scans
CONTINUATION_X_SLACK = 0.03  # page-width fraction a continuation may overhang its head line
PAGE_NUMBER_SLACK = 2  # a lone number this close to the printed page is a page number


def is_page_number_line(line: Line, printed: int) -> bool:
    """A line made only of one number within PAGE_NUMBER_SLACK of the printed page.

    Lone footnote markers ("1", "2") sit in the same bands on born-digital
    pages, so a bare number elsewhere in the band is kept.
    """
    return (
        len(line.words) == 1
        and DIGITS_RE.match(line.words[0].text) is not None
        and abs(int(line.words[0].text) - printed) <= PAGE_NUMBER_SLACK
    )


@dataclass(slots=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block: int
    line: int


@dataclass(slots=True)
class Line:
    words: list[Word]
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.words.sort(key=lambda w: w.x0)
        self.x0 = min(w.x0 for w in self.words)
        self.y0 = min(w.y0 for w in self.words)
        self.x1 = max(w.x1 for w in self.words)
        self.y1 = max(w.y1 for w in self.words)

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass(slots=True)
class PageWords:
    index: int
    width: float
    height: float
    words: list[Word] = field(default_factory=list)


# --- path and bands --------------------------------------------------------------


def start_page(path: str) -> int:
    """SCR start page from a judgment path: 230 for `2025_1_230_238`."""
    m = _PATH_PAGES_RE.search(path)
    if not m:
        raise ValueError(f"cannot read the SCR start page from path {path!r}")
    return int(m.group(1))


def margin_bands(settings: Settings) -> tuple[Band, Band]:
    b = settings.bands.era_c
    if b.left_x is None or b.right_x is None:
        raise ConfigError(
            "bands.era_c.left_x/right_x: not calibrated; run `nyaya-build calibrate` "
            "to write build/bands.json"
        )
    return b.left_x, b.right_x


def head_bands(settings: Settings, era: Era) -> tuple[Band, Band]:
    if era is Era.D:
        return ERA_D_TOP_Y, ERA_D_BOTTOM_Y
    return (settings.bands.top_y or DEFAULT_TOP_Y, settings.bands.bottom_y or DEFAULT_BOTTOM_Y)


# --- words -----------------------------------------------------------------------


def page_words(doc: pymupdf.Document) -> list[PageWords]:
    out = []
    for i, page in enumerate(doc):
        pw = PageWords(i, float(page.rect.width), float(page.rect.height))
        for x0, y0, x1, y1, text, block, line, _ in page.get_text("words"):
            if text.strip():
                pw.words.append(Word(x0, y0, x1, y1, text, block, line))
        out.append(pw)
    return out


def _v_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def neighbour_gap(w: Word, words: Sequence[Word]) -> float:
    """Horizontal distance to the nearest other word on the same line (inf if none)."""
    best = float("inf")
    h = w.y1 - w.y0
    for o in words:
        if o is w:
            continue
        if _v_overlap(w.y0, w.y1, o.y0, o.y1) < 0.4 * min(h, o.y1 - o.y0):
            continue
        best = min(best, max(o.x0 - w.x1, w.x0 - o.x1))
    return best


def in_band(lo_hi: Band, a: float, b: float) -> bool:
    lo, hi = lo_hi
    return lo <= a and b <= hi


def _lone_token(w: Word, pw: PageWords) -> bool:
    return (
        len(w.text) == 1
        and w.text in MARGIN_TOKENS
        and neighbour_gap(w, pw.words) >= MIN_MARGIN_GAP
    )


def margin_letter_ids(pw: PageWords, left: Band, right: Band) -> set[int]:
    """ids of the margin letters on a page.

    A lone margin token is a margin letter when its box lies inside a band, or
    when it stands in a column of at least COLUMN_MIN lone tokens (centres
    within COLUMN_TOL of the page width) in the outer OUTER_ZONE of the page:
    a scan shifted further than any in the calibration sample still prints
    its letters A to H in one column.
    """
    lone = [w for w in pw.words if _lone_token(w, pw)]
    out: set[int] = set()
    centres = [(w.x0 + w.x1) / 2 / pw.width for w in lone]
    for w, c in zip(lone, centres, strict=True):
        fx0, fx1 = w.x0 / pw.width, w.x1 / pw.width
        if (
            in_band(left, fx0, fx1)
            or in_band(right, fx0, fx1)
            or (
                (c < OUTER_ZONE or c > 1 - OUTER_ZONE)
                and sum(abs(c - o) <= COLUMN_TOL for o in centres) >= COLUMN_MIN
            )
        ):
            out.add(id(w))
    return out


def drop_margin_letters(pages: list[PageWords], left: Band, right: Band) -> int:
    """Remove margin letters in place; return how many were removed."""
    removed = 0
    for pw in pages:
        ids = margin_letter_ids(pw, left, right)
        removed += len(ids)
        pw.words = [w for w in pw.words if id(w) not in ids]
    return removed


def group_lines(words: Sequence[Word]) -> list[Line]:
    """PyMuPDF (block, line) groups, ordered by vertical centre."""
    groups: dict[tuple[int, int], list[Word]] = {}
    for w in words:
        groups.setdefault((w.block, w.line), []).append(w)
    lines = [Line(ws) for ws in groups.values()]
    lines.sort(key=lambda ln: (ln.yc, ln.x0))
    return lines


# --- furniture -------------------------------------------------------------------


def letters_key(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _band_of(line: Line, height: float, top: Band, bottom: Band) -> bool:
    f = line.yc / height
    return top[0] <= f <= top[1] or bottom[0] <= f <= bottom[1]


def matches_head(text: str, patterns: Sequence[str]) -> bool:
    """A configured running-head pattern (substring, any case) or the SCR volume head."""
    flat = " ".join(text.split())
    if SCR_HEAD_RE.match(flat):
        return True
    return any(" ".join(p.split()).casefold() in flat.casefold() for p in patterns)


def _similar(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > max(2, len(a) // 5):
        return False
    m = SequenceMatcher(None, a, b, autojunk=False)
    return m.real_quick_ratio() >= REPEAT_SIMILARITY and m.ratio() >= REPEAT_SIMILARITY


def repeated_keys(keys_by_page: Sequence[set[str]], min_pages: int) -> set[str]:
    """Keys found on at least `min_pages` pages, allowing small OCR differences.

    A key seen exactly on two or more pages is a candidate; the pages of keys
    similar to it (OCR variants) count towards it, and those variants are
    returned with it. Comparing only against candidates keeps this linear in
    the number of distinct keys for long judgments.
    """
    pages_of: dict[str, set[int]] = {}
    for i, ks in enumerate(keys_by_page):
        for k in ks:
            pages_of.setdefault(k, set()).add(i)
    candidates = sorted(k for k, ps in pages_of.items() if len(ps) >= 2)
    out: set[str] = set()
    for c in candidates:
        variants = [k for k in pages_of if k == c or _similar(c, k)]
        pages = set().union(*(pages_of[k] for k in variants))
        if len(pages) >= min_pages:
            out.update(variants)
    return out


def remove_furniture(
    lines_by_page: list[list[Line]],
    sizes: Sequence[tuple[float, float]],
    first_printed: int,
    top: Band,
    bottom: Band,
    patterns: Sequence[str],
) -> list[list[Line]]:
    """Drop running heads, footers and page numbers that sit in the bands.

    `sizes` holds each page's (width, height).
    """
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    n = len(lines_by_page)
    banded: list[list[Line]] = [
        [ln for ln in lines if _band_of(ln, heights[i], top, bottom)]
        for i, lines in enumerate(lines_by_page)
    ]
    keys_by_page = [
        {k for ln in lines if len(k := letters_key(ln.text)) >= MIN_REPEAT_KEY}
        for lines in banded[1:]
    ]
    repeats = repeated_keys(keys_by_page, max(2, (n - 1) // 2)) if n > 2 else set()

    out: list[list[Line]] = []
    for i, lines in enumerate(lines_by_page):
        page_no = str(first_printed + i)
        drop: set[int] = set()
        trimmed: dict[int, Line] = {}
        for ln in banded[i]:
            if (
                matches_head(ln.text, patterns)
                or letters_key(ln.text) in repeats
                or is_page_number_line(ln, first_printed + i)
            ):
                drop.add(id(ln))
                continue
            words = [w for w in ln.words if w.text != page_no]
            if not words:
                drop.add(id(ln))
            elif len(words) != len(ln.words):
                trimmed[id(ln)] = Line(words)
        _drop_head_continuations(banded[i], drop, widths[i], heights[i], top)
        out.append([trimmed.get(id(ln), ln) for ln in lines if id(ln) not in drop])
    return out


def _drop_head_continuations(
    banded: Sequence[Line], drop: set[int], width: float, height: float, top: Band
) -> None:
    """Drop the wrapped second line of a running head.

    Running heads that carry a long case title or the author's name wrap onto
    a second line whose wording changes from page to page, so it neither
    repeats nor matches a pattern. A top-band line is that second line when it
    starts within about half a line height of the bottom of a head line
    dropped by the rules above (OCR boxes may overlap slightly), is narrower
    than CONTINUATION_MAX_WIDTH of the page or lies within that head line's
    width, and does not start with a paragraph number. Only one level is
    followed, so body text under the head is never reached by chaining.
    """
    heads = [ln for ln in banded if top[0] <= ln.yc / height <= top[1]]
    dropped = [ln for ln in heads if id(ln) in drop and ln.y1 > ln.y0]
    for ln in heads:
        if id(ln) in drop or PARA_START_RE.match(ln.text):
            continue
        narrow = (ln.x1 - ln.x0) / width < CONTINUATION_MAX_WIDTH
        for d in dropped:
            under = -0.5 <= (ln.y0 - d.y1) / (d.y1 - d.y0) <= 0.6
            slack = CONTINUATION_X_SLACK * width
            inside = ln.x0 >= d.x0 - slack and ln.x1 <= d.x1 + slack
            if under and (narrow or inside):
                drop.add(id(ln))
                break


# --- rows and text ---------------------------------------------------------------


def merge_rows(lines: Sequence[Line]) -> list[Line]:
    """Join lines that share a baseline and do not overlap horizontally."""
    rows: list[Line] = []
    for ln in sorted(lines, key=lambda x: (x.yc, x.x0)):
        if rows:
            r = rows[-1]
            ov = _v_overlap(r.y0, r.y1, ln.y0, ln.y1)
            h = min(r.y1 - r.y0, ln.y1 - ln.y0)
            if h > 0 and ov >= 0.5 * h and all(ln.x1 <= w.x0 or ln.x0 >= w.x1 for w in r.words):
                r.words.extend(ln.words)
                r.refresh()
                continue
        rows.append(Line(list(ln.words)))
    return rows


def _hyphen_split(last: str, first: str) -> bool:
    return len(last) >= 2 and last.endswith("-") and last[-2].isalpha() and first[:1].islower()


def rejoin_hyphens(rows: list[tuple[int, list[str]]]) -> list[tuple[int, list[str]]]:
    """Rejoin words split by a hyphen at a line end (rows are (page, tokens))."""
    vocab = {t.lower().strip(".,;:()[]\"'") for _, toks in rows for t in toks[1:-1]}
    out: list[tuple[int, list[str]]] = []
    pending: list[tuple[int, list[str]]] = [(p, list(t)) for p, t in rows]
    for i, (page, toks) in enumerate(pending):
        if not toks:
            continue
        nxt = pending[i + 1][1] if i + 1 < len(pending) else None
        if nxt and _hyphen_split(toks[-1], nxt[0]):
            head = toks[-1][:-1]
            tail = nxt[0]
            bare_tail = tail.rstrip(".,;:)]\"'")
            hyphenated = f"{head}-{bare_tail}".lower()
            toks[-1] = f"{head}-{tail}" if hyphenated in vocab else head + tail
            del nxt[0]
        out.append((page, toks))
    return out


def build_text(rows: Sequence[tuple[int, list[str]]], first_printed: int):
    parts: list[str] = []
    pmap: list[tuple[int, int]] = []
    offset = 0
    for page, toks in rows:
        if not toks:
            continue
        line = " ".join(toks)
        if parts:
            offset += 1  # the "\n" before this line
        pmap.append((offset, first_printed + page))
        parts.append(line)
        offset += len(line)
    return "\n".join(parts), pmap


# --- entry point -----------------------------------------------------------------


def parse(pdf_bytes: bytes, path: str, era: Era, settings: Settings) -> ParsedDoc:
    """Parse one judgment PDF into clean text with a position map."""
    if era is Era.OUT_OF_SCOPE:
        raise ValueError(f"{path}: out-of-scope PDFs are not parsed")
    first_printed = start_page(path)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        n_pages = doc.page_count
        pages = page_words(doc)
    if era is Era.C:
        left, right = margin_bands(settings)
        drop_margin_letters(pages, left, right)
    top, bottom = head_bands(settings, era)
    lines_by_page = [group_lines(pw.words) for pw in pages]
    lines_by_page = remove_furniture(
        lines_by_page,
        [(pw.width, pw.height) for pw in pages],
        first_printed,
        top,
        bottom,
        settings.running_head_patterns,
    )
    rows = [
        (i, [w.text for w in row.words])
        for i, lines in enumerate(lines_by_page)
        for row in merge_rows(lines)
    ]
    rows = rejoin_hyphens(rows)
    text, pmap = build_text(rows, first_printed)
    return ParsedDoc(
        path=path,
        era=era,
        clean_text=text,
        position_map=pmap,
        pages=n_pages,
        clean_text_version=settings.versions.clean_text,
    )
