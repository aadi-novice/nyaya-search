"""Era classification from the text of a judgment's first 3 pages (design file 18).

Rules (Technical Plan step 3, `era_markers.*` in config):

* Era D: at least `era_markers.d.min_count` of the `era_markers.d.labels`
  appear (each label counts once). The labels follow `MARKERS` in
  scripts/layout_study.py.
* Era C: not Era D, at least `era_markers.c.min_margin_lines` lines match
  `era_markers.c.margin_line_regex`, `era_markers.c.held_regex` matches, and
  the decision year is within `era_markers.c.year_from`..`year_to`.
* Otherwise out of scope.

The labels are matched case-insensitively, with any run of whitespace between
their words (a label can wrap across lines). The margin-line pattern is used
as configured (multi-line). The HELD pattern is matched case-insensitively,
like the zoning patterns of step 4: many 2017-2023 headnotes print "Held:"
rather than "HELD:" (11 of the 50 judgments in the calibration sample).
"""

from __future__ import annotations

import re
from functools import lru_cache

import pymupdf

from nyaya.config import EraMarkers, load_settings
from nyaya.contracts import Era

FIRST_PAGES = 3


@lru_cache(maxsize=8)
def _label_res(labels: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(r"\b" + r"\s+".join(map(re.escape, label.split())) + r"\b", re.IGNORECASE)
        for label in labels
    )


def count_d_labels(text: str, markers: EraMarkers) -> int:
    """How many distinct Era D labels appear in `text`."""
    return sum(1 for rx in _label_res(tuple(markers.d.labels)) if rx.search(text))


def count_margin_lines(text: str, markers: EraMarkers) -> int:
    """Lines that hold nothing but a margin letter."""
    return len(re.findall(markers.c.margin_line_regex, text, flags=re.MULTILINE))


def classify(first3_text: str, decision_year: int, markers: EraMarkers | None = None) -> Era:
    """Return Era.D, Era.C or Era.OUT_OF_SCOPE for a judgment.

    `markers` defaults to `era_markers` from load_settings().
    """
    if markers is None:
        markers = load_settings().era_markers
    if count_d_labels(first3_text, markers) >= markers.d.min_count:
        return Era.D
    c = markers.c
    if (
        c.year_from <= decision_year <= c.year_to
        and count_margin_lines(first3_text, markers) >= c.min_margin_lines
        and re.search(c.held_regex, first3_text, flags=re.MULTILINE | re.IGNORECASE)
    ):
        return Era.C
    return Era.OUT_OF_SCOPE


def first_pages_text(doc: pymupdf.Document, n: int = FIRST_PAGES) -> str:
    """PyMuPDF `get_text("text")` of the first `n` pages, joined."""
    return "".join(doc[i].get_text("text") for i in range(min(n, doc.page_count)))


def classify_pdf(pdf_bytes: bytes, decision_year: int, markers: EraMarkers | None = None) -> Era:
    """`classify` on the first 3 pages of a PDF."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return classify(first_pages_text(doc), decision_year, markers)
