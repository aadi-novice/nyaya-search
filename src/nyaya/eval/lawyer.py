"""`nyaya-eval lawyer-export` and `lawyer-import` (design file 46).

*   **Export:** from a finished (or partly finished) `answers` run, sample
    `lawyer.sample_size` answered questions with `lawyer.seed` (from the
    answered questions sorted by id), and write `ratings.csv` (to fill in)
    and `review.html` (a standalone page to read), plus `rubric_v1.md`.
*   **Import:** read the filled `ratings.csv`, validate every rating (1, 2, 3
    or blank), and record the averages in `eval_runs` as a `lawyer-review`
    run. The review is qualitative evidence, never a gate: import exits 0
    whatever the ratings are.
"""

from __future__ import annotations

import csv
import html
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nyaya.eval.datasets import REPO_ROOT, RUNS_DIR

RUBRIC = REPO_ROOT / "eval" / "lawyer_review" / "rubric_v1.md"
RUBRIC_VERSION = "rubric_v1"
LAWYER_DIR = RUNS_DIR / "lawyer"
QUESTIONS = ("right_paragraphs", "law_correct", "nothing_missing", "usable")
QUESTION_TEXT = {
    "right_paragraphs": "Right paragraphs cited",
    "law_correct": "Law stated correctly",
    "nothing_missing": "Nothing important missing",
    "usable": "Usable as a research starting point",
}
INFO_COLUMNS = ("review_id", "question_id", "question", "answer")
COLUMNS = (*INFO_COLUMNS, *QUESTIONS, "comment", "reviewer")


def sample_answers(records: Sequence[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """`size` answered records, chosen with `seed` from those sorted by id, in id order."""
    answered = sorted((r for r in records if r.get("answer")), key=lambda r: r["id"])
    if len(answered) <= size:
        return answered
    chosen = random.Random(seed).sample(range(len(answered)), size)
    return [answered[i] for i in sorted(chosen)]


def _cite_text(c: dict[str, Any]) -> str:
    paras = ", ".join(c.get("paras") or []) or "-"
    return f"{c['case_name']}, {c['scr_citation']}, para {paras}"


def answer_text(rec: dict[str, Any]) -> str:
    """The answer as plain text, one sentence per line with its citations in brackets."""
    lines = []
    for s in rec["answer"]:
        cites = "; ".join(_cite_text(c) for c in s["citations"])
        lines.append(f"{s['sentence']} [{cites}]")
    for g in rec.get("gaps", []):
        lines.append(f"(gap) {g}")
    return "\n".join(lines)


def export(
    records: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    run_id: str,
    size: int,
    seed: int,
    rubric: Path = RUBRIC,
) -> dict[str, Any]:
    chosen = sample_answers(records, size, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ratings.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for n, rec in enumerate(chosen, start=1):
            w.writerow(
                {
                    "review_id": f"r{n:03d}",
                    "question_id": rec["id"],
                    "question": rec["question"],
                    "answer": answer_text(rec),
                }
            )
    html_path = out_dir / "review.html"
    html_path.write_text(render_html(chosen, run_id), encoding="utf-8")
    if rubric.exists():
        shutil.copyfile(rubric, out_dir / rubric.name)
    return {
        "run_id": run_id,
        "answered": sum(1 for r in records if r.get("answer")),
        "exported": len(chosen),
        "csv": str(csv_path),
        "html": str(html_path),
    }


_CSS = """
:root { --bg: #fbfaf7; --fg: #1d1d1b; --muted: #5f5d57; --line: #dedbd2; --accent: #7a4b16;
        --quote: #f1ede3; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1b1a18; --fg: #ecebe7; --muted: #a9a69e; --line: #3a3833; --accent: #e0a960;
          --quote: #26241f; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 16px/1.55 Georgia, "Times New Roman", serif; }
main { max-width: 52rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
.meta, .muted { color: var(--muted); font-size: .9rem; }
.rubric { border: 1px solid var(--line); padding: .75rem 1rem; margin: 1.5rem 0; }
.rubric ol { margin: .25rem 0 0; padding-left: 1.25rem; }
article { border-top: 2px solid var(--line); padding-top: 1.25rem; margin-top: 2rem; }
h2 { font-size: 1.15rem; margin: 0 0 .5rem; }
h2 .id { color: var(--accent); font-family: ui-monospace, Consolas, monospace;
         font-size: .9rem; margin-right: .5rem; }
.sentence { margin: .9rem 0; }
.cite { font-size: .9rem; color: var(--muted); margin: .25rem 0 0 1rem; }
blockquote { margin: .25rem 0 0 1rem; padding: .4rem .75rem; background: var(--quote);
             border-left: 3px solid var(--accent); font-size: .95rem; }
.gaps { color: var(--muted); font-style: italic; }
"""


def render_html(chosen: Sequence[dict[str, Any]], run_id: str) -> str:
    e = html.escape
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Lawyer review</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>Nyaya Search: answers for review</h1>",
        f'<p class="meta">Answers run {e(run_id)} &middot; {len(chosen)} answers &middot; '
        f"rubric {RUBRIC_VERSION}. Enter ratings in ratings.csv, by review id.</p>",
        '<section class="rubric"><strong>Rate each answer 1 (no), 2 (partly) or 3 (yes):'
        "</strong><ol>",
        *(f"<li>{e(QUESTION_TEXT[q])} (<code>{q}</code>)</li>" for q in QUESTIONS),
        "</ol></section>",
    ]
    for n, rec in enumerate(chosen, start=1):
        parts.append(
            f'<article><h2><span class="id">r{n:03d}</span>{e(rec["question"])}</h2>'
            f'<p class="muted">Question {e(rec["id"])}</p>'
        )
        for s in rec["answer"]:
            parts.append(f'<div class="sentence">{e(s["sentence"])}')
            for c in s["citations"]:
                parts.append(f'<div class="cite">{e(_cite_text(c))}</div>')
                parts.append(f"<blockquote>{e(c['quote'])}</blockquote>")
            parts.append("</div>")
        gaps = rec.get("gaps") or []
        if gaps:
            parts.append('<div class="gaps"><p>Not covered by the passages:</p><ul>')
            parts.extend(f"<li>{e(g)}</li>" for g in gaps)
            parts.append("</ul></div>")
        parts.append("</article>")
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"


# --- import ------------------------------------------------------------------------------


class RatingsError(ValueError):
    pass


def read_ratings(path: Path) -> list[dict[str, Any]]:
    """Validated rows: `{review_id, question_id, ratings: {q: 1-3 | None}, comment, reviewer}`."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [
            c
            for c in ("review_id", "question_id", *QUESTIONS)
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise RatingsError(f"{path}: missing columns {missing}")
        rows = []
        for line, row in enumerate(reader, start=2):
            ratings: dict[str, int | None] = {}
            for q in QUESTIONS:
                raw = (row.get(q) or "").strip()
                if not raw:
                    ratings[q] = None
                elif raw in ("1", "2", "3"):
                    ratings[q] = int(raw)
                else:
                    raise RatingsError(
                        f"{path}:{line} ({row.get('review_id')}): {q} must be 1, 2, 3 or blank, "
                        f"got {raw!r}"
                    )
            rows.append(
                {
                    "review_id": row["review_id"],
                    "question_id": row["question_id"],
                    "ratings": ratings,
                    "comment": (row.get("comment") or "").strip(),
                    "reviewer": (row.get("reviewer") or "").strip(),
                }
            )
    return rows


def summarize_ratings(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_question: dict[str, Any] = {}
    for q in QUESTIONS:
        vals = [r["ratings"][q] for r in rows if r["ratings"][q] is not None]
        per_question[q] = {
            "rated": len(vals),
            "mean": sum(vals) / len(vals) if vals else None,
            "counts": {str(k): vals.count(k) for k in (1, 2, 3)},
            "share_yes": vals.count(3) / len(vals) if vals else None,
        }
    return {
        "rubric": RUBRIC_VERSION,
        "answers": len(rows),
        "fully_rated": sum(1 for r in rows if all(v is not None for v in r["ratings"].values())),
        "reviewers": sorted({r["reviewer"] for r in rows if r["reviewer"]}),
        "questions": per_question,
        "rows": list(rows),
        "gate": False,  # qualitative evidence only (design file 46)
    }
