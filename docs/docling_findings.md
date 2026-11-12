# Docling Test: One PDF per Layout Era

* **Author:** aadi-novice
* **Date:** 2026-11-12
* **Method:** `scripts/docling_test.py` ran Docling 2.150 on the CPU in
  `.venv` (Python 3.12, table structure on). It compared Docling's output with
  PyMuPDF raw text on 5 PDFs, one per era, plus the 1953 PDF a second time
  with OCR forced on. The outputs are in `docs/docling_test/`.
* **Caveat:** one PDF per era, and CPU only. The speed figures are for the
  CPU, not a GPU.

## Results

| Era | File | Pages | Seconds per page | Margin letters inside sentences (Docling) | Running heads removed |
| --- | --- | --- | --- | --- | --- |
| A | 1953_1_1184_1188 | 5 | 1.0 | 0 | yes |
| A, OCR on | 1953_1_1184_1188 | 5 | **6.5** | 1 | yes |
| B | 1975_2_873_880 | 8 | 1.1 | 2 (plus 46 letter-only lines) | yes |
| B | S_2005_5_75_82 | 8 | 0.9 | **31** | yes |
| C | 2015_15_147_155 | 9 | 1.1 | **25** | yes |
| D | 2025_1_230_238 | 9 | 1.0 | 0 | no: shown as headings |

The first run also spent about 17 minutes downloading and loading models,
which happens once per machine.

## Findings

1. **Docling makes the margin-letter problem worse in Eras B and C.**
   - PyMuPDF keeps the margin letters A to H on lines of their own, where
     they are easy to strip.
   - Docling's reading-order step merges them into sentences: "local **C**
     bullies", "offence of **C** murder", "the short **H** issue".
   - Stripping a lone capital letter inside a sentence afterwards is risky,
     because real text contains "Schedule C", "Part B" and "Form A".
   - The reliable fix uses position: drop the letters by their page
     coordinates (they sit in a fixed margin band) **before** text is
     merged. PyMuPDF gives these coordinates directly.
2. **Era D sections are detected, with two flaws.**
   - All labelled sections came out as headings: Issue for Consideration,
     Headnotes, Case Law Cited, List of Acts, List of Keywords, Case Arising
     From, Appearances, Judgment.
   - **Flaw 1:** running heads ("Digital Supreme Court Reports" and the case
     title) are also marked as headings in the body. They appear every
     page, so they must be filtered out.
   - **Flaw 2:** the reading order swapped "List of Acts" and "List of
     Keywords". The Act text "Penal Code, 1860." landed under the Keywords
     heading. Section contents cannot be trusted without a check.
3. **Running heads are removed in Eras A to C.** "SUPREME COURT REPORTS"
   and the page numbers do not appear in Docling's markdown. This is a gain
   over raw PyMuPDF.
4. **OCR helps Era A, and the earlier data check underrated the problem.**
   - The data check found a text layer on 99.8% of pages, but in Era A that
     layer is itself old, poor OCR.
   - Re-running OCR on the 1953 PDF raised the share of word-like tokens from
     0.79 to 0.85. The text became readable:

     | Text layer | OCR re-run |
     | --- | --- |
     | "comin,ina1 electorates", "Mciintaincib-ility" | "communal electorates", "Maintainability" |

   - The cost: 6.5 seconds per page on the CPU, against 1.0.
5. **Speed.**
   - Without OCR, about 1 second per page on the CPU. The full corpus is
     about 527,000 pages, which comes to about 146 CPU-hours in one process.
   - PyMuPDF extracts text about two orders of magnitude faster.
   - GPU speed on Kaggle has not been measured.

## What this changes

- **Docling earns its cost in two places:** Era D section detection, and OCR
  of Era A. In Eras B and C its main output is cleaner text without running
  heads, but it makes margin letters worse.
- **Option to test: a mixed parser.**
  - **PyMuPDF with coordinates** for all eras: strip margin letters, running
    heads and page numbers by their position on the page, and keep the
    page-and-line map for citations.
  - **Docling with OCR** only for Era A, run on Kaggle GPUs.
  - Era D sections found by their fixed labels in the PyMuPDF text, or by
    Docling headings checked against those labels.
- **Before deciding,** measure Docling on a Kaggle GPU (pages per second,
  with and without OCR) on 20 to 50 PDFs per era.
