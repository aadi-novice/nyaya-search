"""Calibrate the Era C margin and furniture bands (design file 17, step 2).

`calibrate(settings)`:

* samples `calibrate.sample_size` manifest rows with a PDF and a decision year
  from 2010 to 2023 (`calibrate.seed`);
* classifies each on its first 3 pages and keeps Era C only; fewer than
  `calibrate.min_era_c` stops the step with exit code 2;
* clusters the x-positions of lone upper-case A-H letters (no other word
  within `MIN_MARGIN_GAP` points on their line) into a left and a right band;
* measures the top and bottom furniture bands from the lines that are running
  heads (a configured pattern, or text repeated on half the pages) or page
  numbers;
* writes `build/bands.json` with the standard `json` module. No left or right
  cluster stops the step with exit code 2 and "no margin band found for era C".

Bands are fractions of the page width (x) or height (y); see pdf_layout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import pymupdf

from nyaya.config import REPO_ROOT, Settings
from nyaya.contracts import Era
from nyaya.corpus.manifest import load_manifest, resolve
from nyaya.corpus.tarstore import read_pdf
from nyaya.parse.eras import classify, first_pages_text
from nyaya.parse.pdf_layout import (
    MARGIN_LETTERS,
    MIN_MARGIN_GAP,
    MIN_REPEAT_KEY,
    Band,
    Line,
    group_lines,
    is_page_number_line,
    letters_key,
    matches_head,
    neighbour_gap,
    page_words,
    repeated_keys,
    start_page,
)

YEAR_FROM, YEAR_TO = 2010, 2023
EXIT_CODE = 2
NO_BAND_MESSAGE = "no margin band found for era C"

BIN = 0.01  # histogram bin, as a fraction of the page width
MIN_BIN_COUNT = 3  # lone letters in a bin for it to count as part of a cluster
MIN_CLUSTER = 20  # lone letters a side needs to have a band at all
MARGIN_TRIM = 0.005  # share of letters trimmed from each end of a margin band
MARGIN_PAD = 0.02  # padding on each side of a margin band (scans shift between volumes)
TRIM = 0.01  # share of lines trimmed from each end of a furniture band
PAD = 0.005  # padding on each side of a furniture band
SEARCH_ZONE = 0.25  # furniture is looked for in the top and bottom quarter of a page
MIN_FURNITURE_LINES = 5


class CalibrationError(SystemExit):
    """Stops `nyaya-build calibrate` with exit code 2; `message` says why."""

    def __init__(self, message: str) -> None:
        super().__init__(EXIT_CODE)
        self.message = message

    def __str__(self) -> str:
        return self.message


# --- margin letters ----------------------------------------------------------------


def lone_letters(doc: pymupdf.Document) -> list[tuple[float, float, float]]:
    """(x0, x1, centre) as page-width fractions for each lone A-H letter."""
    out = []
    for pw in page_words(doc):
        for w in pw.words:
            if (
                len(w.text) == 1
                and w.text in MARGIN_LETTERS
                and neighbour_gap(w, pw.words) >= MIN_MARGIN_GAP
            ):
                x0, x1 = w.x0 / pw.width, w.x1 / pw.width
                out.append((x0, x1, (x0 + x1) / 2))
    return out


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def cluster_band(letters: Sequence[tuple[float, float, float]]) -> Band | None:
    """The band holding the dense part of one side's letters, or None.

    Letters in histogram bins with fewer than MIN_BIN_COUNT members are noise.
    The band spans the remaining boxes, trimmed by MARGIN_TRIM at each end and
    padded by MARGIN_PAD. A side with fewer than MIN_CLUSTER clustered letters has no band.
    """
    bins = Counter(int(c / BIN) for _, _, c in letters)
    dense = [lt for lt in letters if bins[int(lt[2] / BIN)] >= MIN_BIN_COUNT]
    if len(dense) < MIN_CLUSTER:
        return None
    x0s = sorted(lt[0] for lt in dense)
    x1s = sorted(lt[1] for lt in dense)
    lo = max(0.0, _quantile(x0s, MARGIN_TRIM) - MARGIN_PAD)
    hi = min(1.0, _quantile(x1s, 1 - MARGIN_TRIM) + MARGIN_PAD)
    return (round(lo, 4), round(hi, 4))


def margin_bands(letters: Iterable[tuple[float, float, float]]) -> tuple[Band | None, Band | None]:
    left, right = [], []
    for lt in letters:
        (left if lt[2] < 0.5 else right).append(lt)
    return cluster_band(left), cluster_band(right)


# --- furniture -----------------------------------------------------------------------


def furniture_spans(doc: pymupdf.Document, path: str, patterns: Sequence[str]):
    """(top, bottom) lists of (y0, y1) page-height fractions of furniture lines."""
    first = start_page(path)
    pages = page_words(doc)
    zones: list[list[Line]] = []
    for pw in pages:
        zones.append(
            [
                ln
                for ln in group_lines(pw.words)
                if ln.yc / pw.height <= SEARCH_ZONE or ln.yc / pw.height >= 1 - SEARCH_ZONE
            ]
        )
    n = len(pages)
    keys = [{k for ln in z if len(k := letters_key(ln.text)) >= MIN_REPEAT_KEY} for z in zones[1:]]
    repeats = repeated_keys(keys, max(2, (n - 1) // 2)) if n > 2 else set()
    top, bottom = [], []
    for pw, z in zip(pages, zones, strict=True):
        page_no = str(first + pw.index)
        for ln in z:
            furniture = (
                matches_head(ln.text, patterns)
                or letters_key(ln.text) in repeats
                or (len(ln.words) <= 2 and any(w.text == page_no for w in ln.words))
                or is_page_number_line(ln, first + pw.index)
            )
            if furniture:
                span = (ln.y0 / pw.height, ln.y1 / pw.height)
                (top if ln.yc / pw.height <= 0.5 else bottom).append(span)
    return top, bottom


def furniture_band(spans: Sequence[tuple[float, float]]) -> Band | None:
    if len(spans) < MIN_FURNITURE_LINES:
        return None
    y0s = sorted(s[0] for s in spans)
    y1s = sorted(s[1] for s in spans)
    lo = max(0.0, _quantile(y0s, TRIM) - PAD)
    hi = min(1.0, _quantile(y1s, 1 - TRIM) + PAD)
    return (round(lo, 4), round(hi, 4))


# --- the step ------------------------------------------------------------------------


def measure_bands(docs: Iterable[tuple[str, bytes]], patterns: Sequence[str]) -> dict:
    """Bands from Era C PDFs given as (path, bytes); raises CalibrationError if none."""
    letters: list[tuple[float, float, float]] = []
    top: list[tuple[float, float]] = []
    bottom: list[tuple[float, float]] = []
    for path, data in docs:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            letters.extend(lone_letters(doc))
            t, b = furniture_spans(doc, path, patterns)
        top.extend(t)
        bottom.extend(b)
    left, right = margin_bands(letters)
    if left is None or right is None:
        raise CalibrationError(NO_BAND_MESSAGE)
    tb, bb = furniture_band(top), furniture_band(bottom)
    return {
        "era_c": {"left_x": list(left), "right_x": list(right)},
        "top_y": list(tb) if tb else [],
        "bottom_y": list(bb) if bb else [],
    }


def calibration_sample(manifest: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The seeded calibration sample (every era): `nyaya-eval g6-holdout` excludes it."""
    m = manifest
    pool = m[m["has_pdf"] & m["decision_year"].between(YEAR_FROM, YEAR_TO)]
    pool = pool.sort_values("path", kind="mergesort").reset_index(drop=True)
    n = min(settings.calibrate.sample_size, len(pool))
    return pool.sample(n=n, random_state=settings.calibrate.seed) if n else pool


def sample_era_c(settings: Settings, root: Path) -> list[tuple[str, bytes]]:
    """The seeded sample's Era C PDFs, as (path, bytes)."""
    sample = calibration_sample(load_manifest(settings, root), settings)
    markers = settings.era_markers
    out = []
    for r in sample.itertuples(index=False):
        data = read_pdf(r.tar_path, int(r.offset_data), int(r.size))
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            era = classify(first_pages_text(doc), int(r.decision_year), markers)
        if era is Era.C:
            out.append((r.path, data))
    return out


def calibrate(settings: Settings, root: Path | str | None = None) -> Path:
    """Write `build/bands.json`; return its path. Raises CalibrationError (exit 2)."""
    root = Path(root) if root is not None else REPO_ROOT
    docs = sample_era_c(settings, root)
    if len(docs) < settings.calibrate.min_era_c:
        raise _fail(
            f"only {len(docs)} era C judgments in the calibration sample; "
            f"need {settings.calibrate.min_era_c}"
        )
    try:
        bands = measure_bands(docs, settings.running_head_patterns)
    except CalibrationError as exc:
        raise _fail(exc.message) from None
    out = resolve(root, settings.paths.build) / "bands.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bands, indent=2) + "\n", encoding="utf-8")
    return out


def _fail(message: str) -> CalibrationError:
    print(message, file=sys.stderr)
    return CalibrationError(message)
