"""`nyaya-eval g6-holdout`: the held-out PDFs for the G6 hand check (design file 39).

G6 asks for zero margin letters inside sentences and zero running heads in a
hand check of `g6.holdout_size` held-out PDFs. This module picks them and lays
each one out for the author to read:

* **Sample.** Manifest rows with a PDF and a decision year in `--years`
  (default 2010-2026), minus the calibration sample (`calibrate.seed`, the
  rows `nyaya-build calibrate` measured), in path order, shuffled with
  `g6.holdout_seed`. Rows are taken in that order until `g6.holdout_size`
  in-scope PDFs (era C or D) are kept; out-of-scope PDFs are listed as skipped.
* **Per PDF**, under `<out>/<path>/`: `page-NNN.png` for every PDF page,
  `clean_text.txt` (what the build indexes), and `review.html`, which puts each
  page image beside the clean text of that page (split by the position map).
* **Hints.** Each PDF gets two automatic counts, as a starting point only: the
  margin-letter pattern `(?<=[a-z,]) [A-H] (?=[a-z])` in the clean text, and
  lines matching `running_head_patterns`. The hand check is the gate.
* **Index.** `<out>/index.html` links every review page; `handcheck.csv` has
  one row per PDF with blank `margin_letters_found`, `running_heads_found` and
  `notes` columns for the author; `sample.json` records the seed, the years,
  the excluded calibration paths and the skipped rows.

Reads the manifest and the tars only; never the database or the models.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nyaya.config import REPO_ROOT, ConfigError, Settings
from nyaya.contracts import Era, ParsedDoc

DEFAULT_OUT = REPO_ROOT / "eval" / "runs" / "g6_holdout"
DEFAULT_DPI = 110
MARGIN_RESIDUE = re.compile(r"(?<=[a-z,]) [A-H] (?=[a-z])")
HANDCHECK_COLUMNS = (
    "path",
    "decision_year",
    "era",
    "pdf_pages",
    "auto_margin_hits",
    "auto_head_hits",
    "margin_letters_found",
    "running_heads_found",
    "notes",
)

ReadPdf = Callable[[str, int, int], bytes]
ClassifyPdf = Callable[[bytes, int], Era]
ParsePdf = Callable[[bytes, str, Era, Settings], ParsedDoc]
RenderPages = Callable[[bytes, Path, int], int]


# --- sampling ----------------------------------------------------------------------------


def holdout_order(
    manifest: pd.DataFrame,
    years: Sequence[int],
    exclude: Iterable[str],
    seed: int,
) -> pd.DataFrame:
    """Every eligible row (has a PDF, year in `years`, not excluded), seeded order."""
    m = manifest
    excluded = set(exclude)
    pool = m[m["has_pdf"].astype(bool) & m["decision_year"].isin(list(years))]
    pool = pool[~pool["path"].isin(excluded)]
    pool = pool.sort_values("path", kind="mergesort").reset_index(drop=True)
    if pool.empty:
        return pool
    return pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --- one PDF -----------------------------------------------------------------------------


@dataclass
class Reviewed:
    path: str
    decision_year: int
    era: str
    pdf_pages: int
    margin_hits: list[str] = field(default_factory=list)  # snippets around each hit
    head_hits: list[str] = field(default_factory=list)  # whole lines

    def row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "decision_year": self.decision_year,
            "era": self.era,
            "pdf_pages": self.pdf_pages,
            "auto_margin_hits": len(self.margin_hits),
            "auto_head_hits": len(self.head_hits),
            "margin_letters_found": "",
            "running_heads_found": "",
            "notes": "",
        }


def margin_hits(text: str, context: int = 30) -> list[str]:
    out = []
    for m in MARGIN_RESIDUE.finditer(text):
        a, b = max(0, m.start() - context), min(len(text), m.end() + context)
        out.append(text[a:b].replace("\n", " "))
    return out


def head_hits(text: str, patterns: Sequence[str]) -> list[str]:
    pats = [p.lower() for p in patterns if p]
    return [line for line in text.splitlines() if any(p in line.lower() for p in pats)]


def text_by_page(doc: ParsedDoc, first_printed: int) -> dict[int, str]:
    """Clean text per PDF page index, from the position map (printed page - first)."""
    pages: dict[int, list[str]] = {}
    pmap = doc.position_map
    text = doc.clean_text
    for i, (offset, printed) in enumerate(pmap):
        end = pmap[i + 1][0] - 1 if i + 1 < len(pmap) else len(text)
        pages.setdefault(printed - first_printed, []).append(text[offset:end])
    return {k: "\n".join(v) for k, v in pages.items()}


def render_pages(pdf: bytes, out_dir: Path, dpi: int) -> int:
    """Write `page-NNN.png` (1-based) for every page; return the page count."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for i, page in enumerate(doc, 1):
            page.get_pixmap(dpi=dpi).save(out_dir / f"page-{i:03d}.png")
        return doc.page_count


def _first_printed(path: str) -> int:
    from nyaya.parse.pdf_layout import start_page

    return start_page(path)


def review_html(r: Reviewed, doc: ParsedDoc, n_pages: int) -> str:
    e = html.escape
    by_page = text_by_page(doc, _first_printed(r.path))
    rows = []
    for i in range(n_pages):
        rows.append(
            f'<tr><td class="img"><img loading="lazy" src="page-{i + 1:03d}.png" '
            f'alt="PDF page {i + 1}"></td><td><div class="pno">PDF page {i + 1}</div>'
            f"<pre>{e(by_page.pop(i, ''))}</pre></td></tr>"
        )
    leftover = "\n".join(by_page[k] for k in sorted(by_page))
    if leftover:
        rows.append(
            '<tr><td class="img"></td><td><div class="pno">text outside the PDF pages'
            f"</div><pre>{e(leftover)}</pre></td></tr>"
        )
    hints = "".join(f"<li>margin letter? <code>{e(h)}</code></li>" for h in r.margin_hits)
    hints += "".join(f"<li>running head? <code>{e(h)}</code></li>" for h in r.head_hits)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{e(r.path)} - G6 hand check</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
td {{ vertical-align: top; border-top: 1px solid #ccc; padding: 8px; }}
td.img {{ width: 50%; }} img {{ max-width: 100%; border: 1px solid #ddd; }}
pre {{ white-space: pre-wrap; font-family: Georgia, serif; font-size: 15px; margin: 0; }}
.pno {{ color: #777; font-size: 12px; }}
</style></head><body>
<p><a href="../index.html">All held-out PDFs</a></p>
<h1>{e(r.path)}</h1>
<p>Decision year {r.decision_year} &middot; era {e(r.era)} &middot; {n_pages} PDF pages.
Check the clean text (right) for margin letters inside sentences and for running heads.</p>
<h2>Automatic hints ({len(r.margin_hits) + len(r.head_hits)})</h2>
<ul>{hints or "<li>none</li>"}</ul>
<table>{"".join(rows)}</table>
</body></html>
"""


def review_one(
    row: dict[str, Any],
    settings: Settings,
    out: Path,
    *,
    read_pdf: ReadPdf,
    classify_pdf: ClassifyPdf,
    parse_pdf: ParsePdf,
    render: RenderPages,
    dpi: int,
) -> Reviewed | None:
    """Lay out one PDF for review; None if it is out of scope."""
    path = str(row["path"])
    year = int(row["decision_year"])
    pdf = read_pdf(str(row["tar_path"]), int(row["offset_data"]), int(row["size"]))
    era = classify_pdf(pdf, year)
    if era is Era.OUT_OF_SCOPE:
        return None
    doc = parse_pdf(pdf, path, era, settings)
    d = out / path
    d.mkdir(parents=True, exist_ok=True)
    n_pages = render(pdf, d, dpi)
    (d / "clean_text.txt").write_text(doc.clean_text + "\n", encoding="utf-8")
    r = Reviewed(
        path=path,
        decision_year=year,
        era=era.value,
        pdf_pages=n_pages,
        margin_hits=margin_hits(doc.clean_text),
        head_hits=head_hits(doc.clean_text, settings.running_head_patterns),
    )
    (d / "review.html").write_text(review_html(r, doc, n_pages), encoding="utf-8")
    return r


# --- the run -----------------------------------------------------------------------------


@dataclass
class HoldoutResult:
    out: Path
    reviewed: list[Reviewed]
    skipped: list[dict[str, Any]]
    excluded: int
    eligible: int

    def summary(self) -> dict[str, Any]:
        return {
            "out": str(self.out),
            "reviewed": len(self.reviewed),
            "skipped": len(self.skipped),
            "eligible": self.eligible,
            "excluded_calibration": self.excluded,
            "auto_margin_hits": sum(len(r.margin_hits) for r in self.reviewed),
            "auto_head_hits": sum(len(r.head_hits) for r in self.reviewed),
            "index": str(self.out / "index.html"),
            "handcheck": str(self.out / "handcheck.csv"),
        }


def _index_html(result: HoldoutResult, seed: int) -> str:
    e = html.escape
    rows = "".join(
        f'<tr><td><a href="{e(r.path)}/review.html">{e(r.path)}</a></td>'
        f"<td>{r.decision_year}</td><td>{e(r.era)}</td><td>{r.pdf_pages}</td>"
        f"<td>{len(r.margin_hits)}</td><td>{len(r.head_hits)}</td></tr>"
        for r in result.reviewed
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>G6 hand check</title>
<style>body {{ font-family: system-ui, sans-serif; margin: 16px; }}
td, th {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #ddd; }}</style>
</head><body>
<h1>G6 hand check: {len(result.reviewed)} held-out PDFs</h1>
<p>Seed {seed}; the calibration sample ({result.excluded} rows) is excluded.
G6 passes with zero margin letters inside sentences and zero running heads.
Record findings in <code>handcheck.csv</code>. The hint counts are automatic and
only a starting point.</p>
<table><tr><th>path</th><th>year</th><th>era</th><th>pages</th>
<th>margin hints</th><th>head hints</th></tr>{rows}</table>
</body></html>
"""


def run_holdout(
    manifest: pd.DataFrame,
    settings: Settings,
    years: Sequence[int],
    out: Path,
    *,
    size: int | None = None,
    seed: int | None = None,
    dpi: int = DEFAULT_DPI,
    read_pdf: ReadPdf | None = None,
    classify_pdf: ClassifyPdf | None = None,
    parse_pdf: ParsePdf | None = None,
    render: RenderPages | None = None,
) -> HoldoutResult:
    from nyaya.parse.calibrate import calibration_sample

    if read_pdf is None:
        from nyaya.corpus.tarstore import read_pdf as tar_read_pdf

        read_pdf = tar_read_pdf
    if classify_pdf is None:
        from nyaya.parse.eras import classify_pdf as real_classify

        markers = settings.era_markers

        def classify_pdf(pdf: bytes, year: int) -> Era:
            return real_classify(pdf, year, markers)

    if parse_pdf is None:
        from nyaya.parse.pdf_layout import parse as parse_pdf
    render = render or render_pages
    size = settings.g6.holdout_size if size is None else size
    seed = settings.g6.holdout_seed if seed is None else seed

    excluded = list(calibration_sample(manifest, settings)["path"])
    order = holdout_order(manifest, years, excluded, seed)
    out.mkdir(parents=True, exist_ok=True)
    result = HoldoutResult(out, [], [], len(excluded), len(order))
    for row in order.to_dict("records"):
        if len(result.reviewed) >= size:
            break
        try:
            r = review_one(
                row,
                settings,
                out,
                read_pdf=read_pdf,
                classify_pdf=classify_pdf,
                parse_pdf=parse_pdf,
                render=render,
                dpi=dpi,
            )
        except ConfigError:
            raise  # e.g. Era C bands not calibrated: every PDF would fail
        except Exception as exc:
            result.skipped.append(
                {"path": row["path"], "reason": "error", "detail": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if r is None:
            result.skipped.append({"path": row["path"], "reason": "era_out_of_scope"})
        else:
            result.reviewed.append(r)

    with (out / "handcheck.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HANDCHECK_COLUMNS)
        w.writeheader()
        for r in result.reviewed:
            w.writerow(r.row())
    (out / "index.html").write_text(_index_html(result, seed), encoding="utf-8")
    sample = {
        "seed": seed,
        "size": size,
        "years": list(years),
        "calibration_seed": settings.calibrate.seed,
        "excluded_calibration_paths": sorted(excluded),
        "paths": [r.path for r in result.reviewed],
        "skipped": result.skipped,
    }
    (out / "sample.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    return result
