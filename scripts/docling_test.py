"""Run Docling on one PDF per layout era and compare with PyMuPDF raw text.

For each sample writes, under docs/docling_test/:
  <name>.docling.md     Docling markdown export (body only; furniture excluded)
  <name>.labels.txt     every text item with its Docling label and page, in reading order
  <name>.pymupdf.txt    raw PyMuPDF text for comparison
and summary.csv with timings, label counts and artefact checks.

Usage: .venv/Scripts/python scripts/docling_test.py [--ocr-era-a]
"""
import argparse
import csv
import io
import os
import re
import sys
import tarfile
import time
from collections import Counter

import fitz

SAMPLES = [  # (era, year, member name)
    ("A", "1953", "1953_1_1184_1188_EN.pdf"),
    ("B", "1975", "1975_2_873_880_EN.pdf"),
    ("B", "2005", "S_2005_5_75_82_EN.pdf"),
    ("C", "2015", "2015_15_147_155_EN.pdf"),
    ("D", "2025", "2025_1_230_238_EN.pdf"),
]
ROOT = "docs"
OUT = os.path.join(ROOT, "docling_test")
MARGIN_LETTER_LINE = re.compile(r"(?m)^\s*[A-H]\s*$")
MARGIN_LETTER_INLINE = re.compile(r"(?<=[a-z,]) [A-H] (?=[a-z])")


def load_pdf(year, name):
    with tarfile.open(os.path.join(ROOT, "data", "tar", f"year={year}", "english", "english.tar")) as tf:
        return tf.extractfile(tf.getmember(name)).read()


def build_converter(do_ocr):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-era-a", action="store_true", help="also run era A with OCR forced on")
    a = ap.parse_args()
    from docling.datamodel.base_models import DocumentStream

    os.makedirs(OUT, exist_ok=True)
    conv = {False: build_converter(False)}
    runs = [(e, y, n, False) for e, y, n in SAMPLES]
    if a.ocr_era_a:
        conv[True] = build_converter(True)
        runs += [(e, y, n, True) for e, y, n in SAMPLES if e == "A"]

    # warm-up so model loading is not counted in the first timing
    t0 = time.perf_counter()
    conv[False].convert(DocumentStream(name="warm.pdf", stream=io.BytesIO(load_pdf("2025", SAMPLES[-1][2]))), page_range=(1, 1))
    print(f"model load + warm-up: {time.perf_counter() - t0:.1f}s", flush=True)

    rows = []
    for era, year, name, ocr in runs:
        data = load_pdf(year, name)
        stem = name[:-4] + (".ocr" if ocr else "")
        raw = "\n".join(p.get_text() for p in fitz.open(stream=data, filetype="pdf"))
        with open(os.path.join(OUT, stem + ".pymupdf.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        t0 = time.perf_counter()
        res = conv[ocr].convert(DocumentStream(name=name, stream=io.BytesIO(data)))
        secs = time.perf_counter() - t0
        doc = res.document
        md = doc.export_to_markdown()
        with open(os.path.join(OUT, stem + ".docling.md"), "w", encoding="utf-8") as f:
            f.write(md)
        labels = Counter()
        lines = []
        for item, _lvl in doc.iterate_items(included_content_layers=None):
            lab = str(getattr(item, "label", "?")).split(".")[-1]
            labels[lab] += 1
            page = item.prov[0].page_no if getattr(item, "prov", None) else "-"
            txt = (getattr(item, "text", "") or "").replace("\n", " ")
            layer = str(getattr(item, "content_layer", "")).split(".")[-1]
            lines.append(f"[p{page}] {lab:16} {layer:9} | {txt[:160]}")
        with open(os.path.join(OUT, stem + ".labels.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        pages = doc.num_pages()
        row = {
            "era": era, "year": year, "file": name, "ocr": ocr, "pages": pages,
            "seconds": round(secs, 1), "sec_per_page": round(secs / max(pages, 1), 2),
            "md_chars": len(md), "raw_chars": len(raw),
            "margin_letter_lines_md": len(MARGIN_LETTER_LINE.findall(md)),
            "margin_letter_lines_raw": len(MARGIN_LETTER_LINE.findall(raw)),
            "inline_letters_md": len(MARGIN_LETTER_INLINE.findall(md)),
            "inline_letters_raw": len(MARGIN_LETTER_INLINE.findall(raw)),
            "running_head_in_md": md.count("SUPREME COURT REPORTS"),
            "running_head_in_raw": raw.count("SUPREME COURT REPORTS"),
            **{f"lab_{k}": v for k, v in sorted(labels.items())},
        }
        rows.append(row)
        print(row, flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k[0].islower() or k.startswith("lab_"), k))
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
