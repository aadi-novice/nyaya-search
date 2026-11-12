"""Measure parse -> zone -> chunk on seeded samples of judgments (Era B work).

For each group (a decade, or one year) a seeded sample of judgments with a PDF
is drawn from build/manifest.parquet and run through the same steps as
`nyaya-build parse-chunk` (classify, parse, zone, split opinions, chunk,
doc_metrics). Nothing is embedded and no model is loaded.

Per group it reports: the era classification, errors by stage, missing zones,
the G6 metrics (paragraph integrity, size compliance), the paragraph-gap rate
(numbered judgments with fewer distinct whole-number labels than the largest
label), the share of judgments with no paragraph numbers, chunks per judgment,
and a proxy for margin letters left in the text (a lone A-H between two
lower-case words, or a line holding only A-H, per 1,000 words).

Writes, under --out: docs.jsonl (one row per judgment), chunks.jsonl (every
chunk record), texts/<path>.txt (clean text), summary.json and summary.md.

Usage:
  uv run python scripts/era_b_measure.py --out build/era_b/before \
      --data-root D:/Coding_me/rag/docs --decades 1970 1980 1990 2000 --per-group 40 --seed 42
  uv run python scripts/era_b_measure.py --out build/era_b/cd_before --years 2025 --per-group 30

`--force-era c` parses judgments classified out of scope as Era C instead of
skipping them (the "before" baseline: what the Era C rules do on old PDFs).
The script also runs against the code before Era B (PYTHONPATH pointing at
an export of that commit's src/), for the before/after comparison.
`--data-root` points at the folder holding data/tar and metadata/parquet;
the manifest is built under build/ when it is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd

from nyaya.build import opinion_judges
from nyaya.chunk.chunker import count_words, find_paragraphs

try:
    from nyaya.chunk.chunker import paragraph_rules
except ImportError:  # the code before Era B (for "before" runs from an older checkout)

    def paragraph_rules(parsed, settings):
        return {}

from nyaya.chunk.metrics import doc_metrics, summarize
from nyaya.config import REPO_ROOT, load_settings
from nyaya.contracts import Era, Zone
from nyaya.corpus.manifest import build_manifest
from nyaya.corpus.tarstore import read_pdf
from nyaya.parse.eras import classify_pdf
from nyaya.parse.pdf_layout import parse
from nyaya.zone.opinions import split_opinions
from nyaya.zone.zoner import load_act_aliases, zone

INLINE_MARGIN = re.compile(r"(?<=[a-z,;]) [A-H] (?=[a-z])")
LINE_MARGIN = re.compile(r"^[ \t]*[A-H][ \t]*$", re.MULTILINE)
# Running-head residue in the judgment zone: the reports' name, a line that is
# only a volume head ("[1985] 2 S.C.R."), a line of 1-4 digits that is not a
# year (a page number), or a case title in capitals ("STATE OF X v. Y [SEN,
# J.]"). A proxy, checked by hand in docs/era_b_findings.md.
HEAD_RESIDUE = re.compile(
    r"^.*COURT\s+REPORTS.*$"
    r"|^[ \t]*(?:\d{1,4}[ \t]+)?[\[(]\s*\d{4}\s*[\])J]\s*(?:SUPP\.?\s*)?\d*\s*S\s*\.?\s*C\s*\.?\s*R"
    r"[ .]*\d*\W*$"
    r"|^\W*(?!(?:19|20)\d\d\b)\d{1,4}\W*$"
    r"|^[^a-z\n]{3,}\sv[.,]?\s[^a-z\n]{3,}$",
    re.MULTILINE,
)


def sample_rows(manifest: pd.DataFrame, lo: int, hi: int, n: int, seed: int) -> pd.DataFrame:
    pool = manifest[manifest["has_pdf"] & manifest["decision_year"].between(lo, hi)]
    pool = pool.sort_values("path", kind="mergesort").reset_index(drop=True)
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def run_one(row: dict, settings, aliases, force: Era | None, out: Path) -> tuple[dict, list]:
    rec: dict = {"path": row["path"], "year": int(row["decision_year"]), "error": None}
    stage = "read"
    t0 = time.perf_counter()
    try:
        data = read_pdf(row["tar_path"], int(row["offset_data"]), int(row["size"]))
        stage = "classify"
        era = classify_pdf(data, int(row["decision_year"]), settings.era_markers)
        rec["era_detected"] = era.value
        if era is Era.OUT_OF_SCOPE and force is not None:
            era = force
        rec["era_used"] = era.value
        if era is Era.OUT_OF_SCOPE:
            return rec, []
        stage = "parse"
        parsed = parse(data, row["path"], era, settings)
        (out / "texts" / f"{row['path']}.txt").write_text(parsed.clean_text, encoding="utf-8")
        stage = "zone"
        zones = zone(parsed, settings, aliases, [])
        stage = "opinions"
        opinions = split_opinions(parsed, zones, opinion_judges(row), settings)
        stage = "chunk"
        from nyaya.chunk.chunker import chunk

        meta = {"title": row.get("title"), "decision_year": row.get("decision_year")}
        chunks = chunk(meta, parsed, zones, opinions, settings, aliases)
        m = doc_metrics(parsed, zones, opinions, chunks, settings)
    except Exception as exc:  # noqa: BLE001 - measured, not raised
        rec["error"] = {"stage": stage, "error": f"{type(exc).__name__}: {exc}"}
        rec["traceback"] = traceback.format_exc(limit=6)
        return rec, []
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 2)

    text = parsed.clean_text
    js, je = zones.judgment_span
    spans = [(o.char_start, o.char_end) for o in opinions] or [zones.judgment_span]
    labels: list[int] = []
    for s, e in spans:
        paras = find_paragraphs(text, s, e, **paragraph_rules(parsed, settings))
        labels += [int(p.label.split(".")[0]) for p in paras if p.label]
    words = count_words(text)
    jtext = text[js:je]
    rec.update(
        pages=parsed.pages,
        words=words,
        missing=[z.value for z in zones.missing],
        held_points=len(zones.editorial),
        acts=len(zones.acts),
        opinions=len(opinions),
        opinion_authors=[o.author for o in opinions],
        numbered=bool(labels),
        distinct_labels=len(set(labels)),
        max_label=max(labels) if labels else 0,
        gap=bool(labels) and len(set(labels)) < max(labels),
        # the same, not counting an unnumbered first paragraph (common in print)
        gap_mid=bool(labels) and len(set(labels) | {1}) < max(labels),
        chunks=len(chunks),
        judgment_chunks=sum(c.zone is Zone.JUDGMENT for c in chunks),
        editorial_chunks=sum(c.zone is Zone.EDITORIAL for c in chunks),
        judgment_start=js,
        judgment_head=jtext[:160],
        inline_margin=len(INLINE_MARGIN.findall(text)),
        head_residue=[m.group(0)[:80] for m in HEAD_RESIDUE.finditer(jtext)],
        inline_examples=[
            text[max(0, m.start() - 30) : m.end() + 30].replace("\n", " ")
            for m in INLINE_MARGIN.finditer(text)
        ][:10],
        line_margin=len(LINE_MARGIN.findall(text)),
        metrics={k: v for k, v in m.items() if k not in ("judgment_chunk_words",)},
        _m=m,
    )
    return rec, [c.model_dump(mode="json") for c in chunks]


def group_summary(recs: list[dict], settings) -> dict:
    eras = Counter(r.get("era_detected", "unread") for r in recs)
    used = [r for r in recs if r.get("era_used") not in (None, Era.OUT_OF_SCOPE.value)]
    ok = [r for r in used if not r["error"]]
    errs = Counter(r["error"]["stage"] for r in recs if r["error"])
    s = summarize([r["_m"] for r in ok], settings) if ok else None
    numbered = [r for r in ok if r["numbered"]]
    words = sum(r["words"] for r in ok) or 1
    missing = Counter(z for r in ok for z in r["missing"])
    return {
        "sampled": len(recs),
        "era_detected": dict(eras),
        "processed": len(used),
        "errors": dict(errs),
        "error_share": (sum(errs.values()) / len(used)) if used else 0.0,
        "paragraph_integrity": s["paragraph_integrity"] if s else None,
        "size_compliance": s["size_compliance"] if s else None,
        "size_compliance_strict": s["size_compliance_strict"] if s else None,
        "counts": s["counts"] if s else None,
        "judgment_chunk_words": s["judgment_chunk_words"] if s else None,
        "numbered_share": len(numbered) / len(ok) if ok else 0.0,
        "gap_rate": (sum(r["gap"] for r in numbered) / len(numbered)) if numbered else 0.0,
        "gap_rate_mid": (sum(r["gap_mid"] for r in numbered) / len(numbered))
        if numbered
        else 0.0,
        "chunks_per_judgment": (sum(r["chunks"] for r in ok) / len(ok)) if ok else 0.0,
        "judgment_chunks_per_judgment": (sum(r["judgment_chunks"] for r in ok) / len(ok))
        if ok
        else 0.0,
        "held_points_per_judgment": (sum(r["held_points"] for r in ok) / len(ok)) if ok else 0.0,
        "missing_zones": dict(missing),
        "inline_margin_per_1k_words": 1000 * sum(r["inline_margin"] for r in ok) / words,
        "line_margin_per_1k_words": 1000 * sum(r["line_margin"] for r in ok) / words,
        "head_residue_per_judgment": (sum(len(r["head_residue"]) for r in ok) / len(ok))
        if ok
        else 0.0,
        "words_per_judgment": words / len(ok) if ok else 0,
        "seconds_per_judgment": (sum(r["seconds"] for r in recs) / len(recs)) if recs else 0,
    }


def fmt(v, pct=False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.1f}%"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def summary_md(groups: dict[str, dict]) -> str:
    rows = [
        ("sampled", "sampled", False),
        ("processed (not out of scope)", "processed", False),
        ("errors", "error_share", True),
        ("paragraph integrity", "paragraph_integrity", True),
        ("size compliance", "size_compliance", True),
        ("size compliance (strict)", "size_compliance_strict", True),
        ("numbered judgments", "numbered_share", True),
        ("paragraph-gap rate (numbered)", "gap_rate", True),
        ("gap rate, first paragraph aside", "gap_rate_mid", True),
        ("chunks per judgment", "chunks_per_judgment", False),
        ("Held points per judgment", "held_points_per_judgment", False),
        ("inline margin letters / 1k words", "inline_margin_per_1k_words", False),
        ("margin-letter lines / 1k words", "line_margin_per_1k_words", False),
        ("running-head residue lines / judgment", "head_residue_per_judgment", False),
        ("seconds per judgment", "seconds_per_judgment", False),
    ]
    names = list(groups)
    out = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, key, pct in rows:
        out.append(f"| {label} | " + " | ".join(fmt(groups[n][key], pct) for n in names) + " |")
    out.append("| era detected | " + " | ".join(str(groups[n]["era_detected"]) for n in names))
    out.append("| missing zones | " + " | ".join(str(groups[n]["missing_zones"]) for n in names))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None, help="folder with data/tar and metadata/parquet")
    ap.add_argument("--decades", type=int, nargs="*", default=[])
    ap.add_argument("--years", type=int, nargs="*", default=[])
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-era", choices=["b", "c", "d"], default=None)
    args = ap.parse_args(argv)

    settings = load_settings(env_file=None)
    if args.data_root:
        root = Path(args.data_root)
        settings = settings.model_copy(
            update={
                "paths": settings.paths.model_copy(
                    update={
                        "tar_root": (root / "data" / "tar").as_posix(),
                        "metadata_root": (root / "metadata" / "parquet").as_posix(),
                    }
                )
            }
        )
    manifest_path = REPO_ROOT / settings.paths.build / "manifest.parquet"
    if not manifest_path.exists():
        build_manifest(settings)
    manifest = pd.read_parquet(manifest_path)
    aliases = load_act_aliases(settings)
    force = Era(args.force_era) if args.force_era else None
    out = Path(args.out)
    (out / "texts").mkdir(parents=True, exist_ok=True)

    groups = [(f"{d}s", d, d + 9) for d in args.decades] + [(str(y), y, y) for y in args.years]
    summaries: dict[str, dict] = {}
    with (
        open(out / "docs.jsonl", "w", encoding="utf-8") as fd,
        open(out / "chunks.jsonl", "w", encoding="utf-8") as fc,
    ):
        for name, lo, hi in groups:
            recs = []
            rows = sample_rows(manifest, lo, hi, args.per_group, args.seed)
            for row in rows.to_dict(orient="records"):
                j = row.get("judges")
                row["judges"] = list(j) if j is not None else []
                rec, chunks = run_one(row, settings, aliases, force, out)
                rec["group"] = name
                recs.append(rec)
                fd.write(json.dumps({k: v for k, v in rec.items() if k != "_m"}) + "\n")
                for c in chunks:
                    fc.write(json.dumps(c, sort_keys=True) + "\n")
            summaries[name] = group_summary(recs, settings)
            print(
                name,
                json.dumps(
                    {
                        k: summaries[name][k]
                        for k in (
                            "processed",
                            "error_share",
                            "paragraph_integrity",
                            "size_compliance",
                            "gap_rate",
                        )
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_md(summaries), encoding="utf-8")
    print(summary_md(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
