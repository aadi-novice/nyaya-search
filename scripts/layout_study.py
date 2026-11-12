"""Layout study: dump the first N pages of the first PDF in each year's English tar,
and tabulate which structural markers appear, so we can see how the layout changes over time.

Writes docs/layout_study/<year>_<name>.txt and docs/layout_study/markers.csv.
Usage: python scripts/layout_study.py [--root docs] [--pages 3]
"""
import argparse
import csv
import glob
import os
import re
import tarfile

import fitz

# (column name, regex) – markers seen in Supreme Court Reports layouts
MARKERS = [
    ("headnote", r"\bHEAD\s*NOTE|\bHeadnotes?\b"),
    ("issue_for_consideration", r"Issue for Consideration"),
    ("case_law_cited", r"Case Law Cited|Cases? (referred|cited)|Referred to\b|Relied on\b"),
    ("list_of_acts", r"List of Acts"),
    ("keywords", r"List of Keywords|Keywords"),
    ("case_arising_from", r"Case Arising From|APPELLATE JURISDICTION|ORIGINAL JURISDICTION|CIVIL APPELLATE|CRIMINAL APPELLATE"),
    ("appearances", r"Appearances for Parties|Advocates? for the|for the [Aa]ppellants?\.?$|for the [Rr]espondents?"),
    ("judgment_delivered_by", r"Judgment of the Court was delivered by|delivered the Judgment|J U D G M E N T|\bJUDGMENT\b"),
    ("reportable", r"\bREPORTABLE\b|NON-REPORTABLE"),
    ("neutral_citation", r"\d{4}\s*INSC\s*\d+"),
    ("scr_citation", r"\[\d{4}\]\s*\d+\s*S\.?C\.?R\.?"),
    ("digitally_signed", r"Digitally signed|Signature Not Verified|Reason:"),
    ("disclaimer", r"Disclaimer|vernacular|translated"),
    ("numbered_paras", r"(?m)^\s*\d{1,3}\.\s+[A-Z]"),
    ("held_marker", r"\bHeld\s*[:,]|\bHELD\b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs")
    ap.add_argument("--pages", type=int, default=3)
    a = ap.parse_args()
    out_dir = os.path.join(a.root, "layout_study")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for tpath in sorted(glob.glob(os.path.join(a.root, "data", "tar", "year=*", "english", "english.tar"))):
        year = re.search(r"year=(\d{4})", tpath).group(1)
        with tarfile.open(tpath, "r:") as tf:
            first = next(m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".pdf"))
            data = tf.extractfile(first).read()
        name = os.path.basename(first.name)
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [doc[i].get_text("text") for i in range(min(a.pages, len(doc)))]
        text = "\n".join(f"===== page {i + 1} =====\n{p}" for i, p in enumerate(pages))
        with open(os.path.join(out_dir, f"{year}_{name[:-4]}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        row = {"year": year, "file": name, "total_pages": len(doc),
               "chars_first_pages": sum(len(p) for p in pages),
               "lines_first_pages": sum(p.count("\n") for p in pages)}
        for col, rx in MARKERS:
            row[col] = len(re.findall(rx, text))
        rows.append(row)
    with open(os.path.join(out_dir, "markers.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} files to {out_dir}")


if __name__ == "__main__":
    main()
