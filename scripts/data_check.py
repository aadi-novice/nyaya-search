"""Data check on the downloaded Supreme Court judgments (metadata + English tars).

Answers the brief's "decide first" questions with measurements:
  1. Canonical judgment ID: which field is unique / complete, and where fields conflict.
  2. Tar access: does each tar match its index.json, and how fast can we index member offsets.
  3. Text layer vs scanned pages, on a seeded sample of PDFs.
  4. Chunk count estimate (paragraph-aware ~400-word chunks), extrapolated to the corpus.

Read-only. Writes docs/data_check_report.md and docs/data_check.json.
Usage: python scripts/data_check.py [--root docs] [--sample 200] [--seed 42]
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import statistics
import tarfile
import time
from collections import Counter

import fitz  # PyMuPDF
import pandas as pd

MIN_CHARS_TEXT_PAGE = 200  # below this, a page is treated as having no usable text layer
CHUNK_WORDS = 400
PARA_RE = re.compile(r"\n\s*\n|\n(?=\s*\d{1,3}\.\s)")  # blank line, or a numbered paragraph start


def blank(s):
    return s.isna() | (s.astype("string").str.strip() == "")


def id_analysis(df):
    out = {"rows": int(len(df))}
    for col in ["path", "cnr", "nc_display", "case_id", "citation"]:
        s = df[col].astype("string")
        b = blank(s)
        nonblank = s[~b]
        out[col] = {
            "blank": int(b.sum()),
            "unique_nonblank": int(nonblank.nunique()),
            "duplicated_nonblank_rows": int(nonblank.duplicated(keep=False).sum()),
        }
    # same path in more than one year folder
    g = df.groupby("path")["year"].nunique()
    out["paths_in_multiple_years"] = int((g > 1).sum())
    # conflicts: one path mapping to several different non-blank CNRs / nc_display values
    for col in ["cnr", "nc_display"]:
        sub = df[~blank(df[col].astype("string"))]
        c = sub.groupby("path")[col].nunique()
        out[f"paths_with_conflicting_{col}"] = int((c > 1).sum())
    # how many distinct judgments if we key on path
    out["distinct_paths"] = int(df["path"].nunique())
    # year folder vs year in decision_date
    dd_year = df["decision_date"].astype("string").str[-4:]
    out["year_folder_ne_decision_year"] = int((dd_year != df["year"].astype("string")).sum())
    langs = df["available_languages"].astype("string").fillna("")
    out["rows_with_non_english_versions"] = int((langs.str.contains(",")).sum())
    return out


def tar_analysis(root):
    rows = []
    members_by_name = {}
    for tpath in sorted(glob.glob(os.path.join(root, "data", "tar", "year=*", "english", "english.tar"))):
        year = tpath.split("year=")[1].split(os.sep)[0].split("/")[0]
        idx_path = tpath.replace("english.tar", "english.index.json")
        t0 = time.perf_counter()
        try:
            with tarfile.open(tpath, "r:") as tf:
                mems = [m for m in tf.getmembers() if m.isfile()]
        except Exception as e:  # noqa: BLE001
            rows.append({"year": year, "error": str(e)})
            continue
        scan_s = time.perf_counter() - t0
        names = {os.path.basename(m.name) for m in mems}
        idx_names = set()
        if os.path.exists(idx_path):
            with open(idx_path, encoding="utf-8") as f:
                idx = json.load(f)
            for p in idx.get("parts", []):
                idx_names.update(p.get("files", []))
        for m in mems:
            members_by_name[(year, os.path.basename(m.name))] = (tpath, m.offset_data, m.size)
        rows.append({
            "year": year, "members": len(mems), "index_files": len(idx_names),
            "in_tar_not_index": len(names - idx_names), "in_index_not_tar": len(idx_names - names),
            "s_prefixed": sum(n.startswith("S_") for n in names),
            "scan_seconds": round(scan_s, 3), "tar_mb": round(os.path.getsize(tpath) / 1e6, 1),
        })
    return rows, members_by_name


def read_member(tpath, offset, size):
    with open(tpath, "rb") as f:
        f.seek(offset)
        return f.read(size)


def chunk_count(text):
    paras = [p.strip() for p in PARA_RE.split(text) if p.strip()]
    chunks, cur = 0, 0
    for p in paras:
        w = len(p.split())
        if w >= CHUNK_WORDS:  # long paragraph: split on its own
            if cur:
                chunks += 1
                cur = 0
            chunks += -(-w // CHUNK_WORDS)
        elif cur + w > CHUNK_WORDS:
            chunks += 1
            cur = w
        else:
            cur += w
    return chunks + (1 if cur else 0)


def pdf_sample(members_by_name, n, seed):
    keys = sorted(members_by_name)
    rng = random.Random(seed)
    sample = rng.sample(keys, min(n, len(keys)))
    res, hashes = [], Counter()
    for year, name in sample:
        tpath, off, size = members_by_name[(year, name)]
        data = read_member(tpath, off, size)
        hashes[hashlib.sha256(data).hexdigest()] += 1
        r = {"year": year, "name": name, "bytes": size}
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            if doc.needs_pass:
                r["error"] = "encrypted"
            else:
                chars = [len(pg.get_text("text").strip()) for pg in doc]
                text = "\n".join(pg.get_text("text") for pg in doc)
                r.update(pages=len(chars), low_text_pages=sum(c < MIN_CHARS_TEXT_PAGE for c in chars),
                         words=len(text.split()), chunks=chunk_count(text))
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {e}"[:200]
        res.append(r)
    return res, sum(c - 1 for c in hashes.values() if c > 1)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.root, "metadata", "parquet", "year=*", "metadata.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ids = id_analysis(df)
    tars, members = tar_analysis(a.root)
    sample, dup_in_sample = pdf_sample(members, a.sample, a.seed)

    ok = [s for s in sample if "error" not in s]
    errs = [s for s in sample if "error" in s]
    pages = sum(s["pages"] for s in ok)
    low = sum(s["low_text_pages"] for s in ok)
    docs_any_low = sum(s["low_text_pages"] > 0 for s in ok)
    chunks = [s["chunks"] for s in ok]
    words = [s["words"] for s in ok]
    n_corpus = ids["distinct_paths"]
    mean_chunks = statistics.mean(chunks) if chunks else 0
    mean_pages = pages / len(ok) if ok else 0

    # metadata path -> tar member coverage for downloaded years
    years_dl = {t["year"] for t in tars if "error" not in t}
    have = {(y, n) for (y, n) in members}
    meta_dl = df[df["year"].astype(str).isin(years_dl)]
    matched = sum(((str(r.year), f"{r.path}_EN.pdf") in have) for r in meta_dl.itertuples())

    summary = {
        "ids": ids, "tars": tars, "sample_size": len(sample), "sample_errors": errs,
        "sample_pages": pages, "sample_low_text_pages": low, "sample_docs_with_low_text": docs_any_low,
        "sample_identical_pdf_bytes_extra_copies": dup_in_sample,
        "mean_pages_per_judgment": mean_pages, "mean_words": statistics.mean(words) if words else 0,
        "median_words": statistics.median(words) if words else 0,
        "mean_chunks_per_judgment": mean_chunks, "est_corpus_chunks": round(mean_chunks * n_corpus),
        "est_corpus_pages": round(mean_pages * n_corpus),
        "meta_rows_downloaded_years": int(len(meta_dl)), "meta_rows_matched_to_pdf": int(matched),
        "params": {"min_chars_text_page": MIN_CHARS_TEXT_PAGE, "chunk_words": CHUNK_WORDS, "seed": a.seed},
    }
    with open(os.path.join(a.root, "data_check.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "sample": sample}, f, indent=2)

    tar_ok = [t for t in tars if "error" not in t]
    L = [
        "# Data check report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/data_check.py` (seed {a.seed}). "
        f"Covers all metadata and the {len(tar_ok)} English year tars on disk at run time.",
        "",
        "## 1. Canonical judgment ID",
        "",
        f"Metadata rows: {ids['rows']}. Distinct `path` values: {ids['distinct_paths']}.",
        "",
        "| Field | Blank | Unique (non-blank) | Rows sharing a value |",
        "| --- | --- | --- | --- |",
    ]
    for col in ["path", "cnr", "nc_display", "case_id", "citation"]:
        c = ids[col]
        L.append(f"| `{col}` | {c['blank']} | {c['unique_nonblank']} | {c['duplicated_nonblank_rows']} |")
    L += [
        "",
        f"- Same `path` in more than one year folder: {ids['paths_in_multiple_years']}",
        f"- `path` values with conflicting CNRs: {ids['paths_with_conflicting_cnr']}; "
        f"with conflicting `nc_display`: {ids['paths_with_conflicting_nc_display']}",
        f"- Rows whose year folder differs from the decision-date year: {ids['year_folder_ne_decision_year']}",
        f"- Rows with non-English versions available: {ids['rows_with_non_english_versions']}",
        "",
        "## 2. Tar access",
        "",
        f"English tars scanned: {len(tar_ok)}; errors: {len(tars) - len(tar_ok)}. "
        f"Total members: {sum(t['members'] for t in tar_ok)}. "
        f"Header scan time, all tars: {sum(t['scan_seconds'] for t in tar_ok):.1f} s.",
        f"Mismatches with index.json: {sum(t['in_tar_not_index'] + t['in_index_not_tar'] for t in tar_ok)} files. "
        f"`S_`-prefixed members: {sum(t['s_prefixed'] for t in tar_ok)}.",
        f"Metadata rows in downloaded years: {len(meta_dl)}; matched to `<path>_EN.pdf`: {matched} "
        f"({pct(matched, len(meta_dl))}).",
        "",
        "`index.json` lists file names only, not byte offsets. One header scan per tar gives the offsets; "
        "each PDF is then read with a single seek, without unpacking.",
        "",
        "## 3. Text layer vs scanned pages",
        "",
        f"Sample: {len(sample)} PDFs (seeded), {len(errs)} unreadable. Pages: {pages}.",
        f"Pages with under {MIN_CHARS_TEXT_PAGE} characters of text: {low} ({pct(low, pages)}). "
        f"Judgments with at least one such page: {docs_any_low} ({pct(docs_any_low, len(ok))}).",
        f"Extra byte-identical copies within the sample: {dup_in_sample}.",
        "",
        "## 4. Size estimate",
        "",
        f"Mean pages per judgment: {mean_pages:.1f}. Words: mean {summary['mean_words']:.0f}, "
        f"median {summary['median_words']:.0f}.",
        f"Mean chunks per judgment (paragraph-aware, ~{CHUNK_WORDS} words): {mean_chunks:.1f}.",
        f"Estimated corpus: about {summary['est_corpus_pages']:,} pages and "
        f"{summary['est_corpus_chunks']:,} chunks for {n_corpus:,} judgments.",
        "",
        "## Caveats",
        "",
        "- The sample covers only years downloaded at run time; recent years may differ (longer judgments).",
        f"- A page under {MIN_CHARS_TEXT_PAGE} characters may be a short last page, not a scan. "
        "Treat the low-text share as an upper bound on OCR need.",
        "- Chunk counts use a simple paragraph split, not the final chunker.",
    ]
    with open(os.path.join(a.root, "data_check_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
