# Era B Findings: Parsing the 1964–2009 Printed SCR Volumes

* **Author:** aadi-novice
* **Date:** 2026-11-13
* **Method:** `scripts/era_b_measure.py` draws a seeded sample of judgments
  with a PDF from the manifest, per decade of decision year, and runs the
  same steps as `nyaya-build parse-chunk` (classify, parse, zone, opinions,
  chunk, G6 `doc_metrics`). Nothing is embedded and no model is loaded.
  Samples: 40 per decade for the 1970s–2000s with seed 42 (the measurement
  sample, used while building) and seed 43 (a fresh sample, to check for
  overfitting); 13 per decade with seed 4242 (the held-out hand check); 8 per
  year for 1964–1969. "Before" is the code at `c29ad78` (the base of this
  branch), run on the same samples. Coordinates and page texts were read
  with `get_text("words")` and `get_text("dict")` on about 20 PDFs.
* **Caveat:** the samples are small (40 per decade). The text layer is the
  publisher's OCR; nothing here re-OCRs a page.

## Summary

* The current pipeline skips every Era B judgment: all 160 of the seed-42
  sample were `out_of_scope` (Era C needs a 2010–2023 decision year). So
  1970–2009 had no chunks at all.
* Forcing the Era C rules on these PDFs gives G6 metrics that look perfect
  (100% integrity and size compliance) but text that is not: 1.4–3.5 running
  heads per judgment are left in the judgment text, every "Case Law
  Reference" zone is missing, judgment starts are missed in 7 of 40 1970s
  judgments, and numbered paragraphs are mostly false (a lone "752." at a
  line start numbers a judgment).
* With the Era B rules, both samples pass the G6 automatic gates in every
  decade (integrity 100%, size compliance 99.5–100%), running-head residue
  falls to 0.13–0.30 lines per judgment (all checked residue is body text,
  not heads), and lone margin letters B–H fall to 0.06–0.29 per 1,000 words,
  the level of real text ("Part B", "R & D") in the 2016/2025 output
  (0.16–0.19).
* Era C and Era D output is byte-identical before and after (182 judgments,
  2010–2026, clean text and chunks).

## The layout, as measured

* **Margin letters A–H** stand in a column beside the text on every page:
  on the right of even printed pages and the left of odd ones in most
  volumes. PyMuPDF puts them on a line of their own (1970s–80s) or, in most
  1990–2009 scans, on the body line itself ("D  communicated to the person",
  "would cross over  D"), 6–20 points from the nearest word. OCR misreads
  them as "c", "8", "e", "B'", "J1", "0", or adds specks ("~-", "'", ".").
* **Positions** (fractions of the page width): letters at 0.04–0.13 (left)
  and 0.88–0.96 (right) on the usual 400–450 point scans; body text at
  0.09–0.15 to 0.82–0.93. Scans are skewed: one 1979 page moves its left
  body edge from 0.088 at the top to 0.060 at the bottom, and its letters
  from 0.040 to 0.006. 3–6% of 1970–1985 PDFs place the scanned page in the
  middle of a 612 × 792 sheet (text at 0.20–0.80 of the width, 0.15–0.85 of
  the height), so fixed page-fraction bands miss their letters and heads.
* **Running heads:** "938 SUPREME COURT REPORTS [1985] 2 S.C.R." on even
  pages; the case title on odd pages, in capitals, often with the author
  ("STATE OF KERALA v. A. LAKSHMIKUTTY [SEN, J.] 143"; 1990s–2000s: "MIS.
  SETHI AUTO S. STATION & ANR. v. DELHI" wrapping to a second line). The
  page number is at the head's start or end; the first page prints
  "[2008] 14 S.C.R. 598" at the top or the page number at the foot. OCR
  varies the title head from page to page ("LAKSHMIKUTTY", "LAKSHMJKUTTY",
  "LAKSHMIKUTIY"), so the "repeats on half the pages" rule misses most.
* **Zones:** parties, date, bench "[A. N. RAY, C.J. AND JASWANT SINGH,
  JJ.]"; catchwords; a summary of the facts (absent in Era C); "HELD:"
  (OCR: "HELB :", "HBLD:", "J{ELO") with points "1.", "1.1", "(1)", "(i)"
  or "(I)", and page-and-line pinpoints "[798A G-H]" rather than paragraph
  links; cases cited; "CIVIL APPELLATE JURISDICTION: ..." (OCR: "CtVIL",
  "C1v1L APPHLATE", "CRntrNAL", "JuR1smcnoN"); the order appealed from;
  counsel; "The Judgment of the Court was delivered by" (or "The following
  Order of the Court was delivered:", "JUDGMENT", or nothing); then the
  author and the first sentence on one line ("JASWANT SINGH, J.-These
  appeals ...", "Shah, J. One Sashi ..."). No "Case Law Reference" table.
* **Paragraph numbers:** numbered judgments are 15–18% of the 1970s, 23–28%
  of the 1980s, 25–35% of the 1990s and 50% of the 2000s. The first
  paragraph is usually unprinted as "1." (the number appears from "2."), and
  OCR damages numbers: "4, The", "8: Shri", "11.Aggrieved", "l. This",
  "O 11. Mr.". Unnumbered judgments mark paragraphs only by a first-line
  indent of 0.03–0.05 of the page width.
* **Era A and B:** on the first 3 pages, 149 of 150 judgments decided
  1950–1963 show 0–2 lines with a margin letter (one 1957 judgment shows 4);
  from 1964 on, nearly all show 10–24, and 1–2 page orders 4–5. 1964 is the
  changeover (3 of 8 sampled are Era B). Era A has side notes and
  neighbouring cases and needs OCR (Future).
* **Footnotes:** "(1) [1967] 1 S.C.R. 120." at the foot of a page, after the
  body text; they stay in the text (see Remaining failure modes).

## What was built

* **Detection** (`eras.py`, `[era_markers.b]`): not D or C, a decision year
  up to 2009, and at least 5 lines with a margin letter alone or at one end
  of a line. No HELD is required. `out_of_scope_reason` names Era A
  (`era_a_scan`: a year up to 2009 and too few margin lines); the build
  writes it to the `detail` column of `skipped.csv`.
* **Parser** (`pdf_layout.py`, `[era_b]`), Era B only:
  * margin letters by the body column: a token of up to 3 characters holding
    one letter A–H or OCR variant, or only punctuation, inside a wide band
    (left −0.05–0.35, right 0.65–1.05) and at least 0.008 of the width
    outside the body column. The column is the 10th/90th percentile of the
    starts/ends of the page's 4+-word lines after removing the scan's skew,
    which is the median slope between lone letters on the same side. A
    bracketed token ("(c)", "(1)") is never a margin letter;
  * running heads: of the first two rows of a page (top down, stopping at
    the first that is not a head, within 0.15 of the page height below the
    first text line), a row that starts or ends with the printed page number
    (OCR "l43" allowed), names the reports, or is a case title in capitals
    with " v. ". Their wrapped second lines go by the existing continuation
    rule; a lone page number as the last row of a page goes too. Lines of
    specks only are dropped;
  * paragraph breaks from indentation, written as a blank line in the clean
    text (Era C and D text never has blank lines): a first-line indent
    (relative to the next line), an indented line after a full one, or the
    text after a block quotation; the last line of a quotation (indented on
    the right too) is not a start.
* **Zoner** (`zoner.py`): the Era C bench line, or a damaged one ("JJ,]",
  "JJ.J", no opening bracket); OCR forms of HELD and of the jurisdiction
  line; the judgment starts at the "delivered by" line, a "JUDGMENT" /
  "ORDER" line or the author's "NAME, J." line within 3,000 characters after
  the jurisdiction line (so case details and counsel are in no zone), else
  at the jurisdiction line; without a jurisdiction line the editorial ends
  at "Appeal by special leave ..." / "From the Judgment ...". Held points
  split on Era C numbers or on "(1)", "(i)", "(I)". No references zone.
  Act names are cut to their run of capitalised words, because the facts
  summary names an Act mid-sentence.
* **Opinions** (`opinions.py`): an author line followed by the first
  sentence; OCR forms ("SHARMA, 'J.", "PATHAK, J,", "AHMADI, CJ."); names
  matched to the bench or metadata with OCR tolerance (letters of the
  printed name against the end of the known name, ratio ≥ 0.75).
* **Chunker** (`chunker.py`, `metrics.py`): an OCR-tolerant paragraph-number
  pattern, and a numbered sequence needs 3 paragraphs
  (`era_b.min_numbered_paragraphs`); fewer, and the judgment is chunked on
  its blank-line paragraphs. `paragraph_integrity` uses the same rules.

## Before and after

"Before" forces the Era C rules (the old pipeline itself processed none of
these: every one was out of scope). Integrity and size compliance are the
G6 gates (98% and 99%). The gap rate counts numbered judgments with fewer
distinct whole-number labels than the largest; "first paragraph aside" does
not count an unnumbered first paragraph. Running-head residue counts lines
in the judgment zone that are only the reports' name, a volume head, a
number that is not a year, or a capitalised "X v. Y"; after the change the
residue is footnote numbers and list items, not heads (checked by hand).

### Measurement sample (seed 42, 40 per decade)

| metric | 1970s before | after | 1980s before | after | 1990s before | after | 2000s before | after |
|---|---|---|---|---|---|---|---|---|
| era detected (old / new) | 40 out of scope | 40 b | 40 out of scope | 39 b, 1 out | 40 out of scope | 40 b | 40 out of scope | 40 b |
| errors | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| paragraph integrity | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| size compliance | 100% | 100% | 100% | 100% | 100% | 99.5% | 100% | 100% |
| numbered judgments | 45.0% | 15.0% | 62.5% | 28.2% | 55.0% | 35.0% | 77.5% | 50.0% |
| paragraph-gap rate | 72.2% | 33.3% | 80.0% | 63.6% | 86.4% | 64.3% | 83.9% | 35.0% |
| gap rate, first paragraph aside | 55.6% | 33.3% | 60.0% | 36.4% | 59.1% | 21.4% | 71.0% | 35.0% |
| chunks per judgment | 14.28 | 13.68 | 17.45 | 17.05 | 11.78 | 11.68 | 12.78 | 12.22 |
| Held points per judgment | 1.12 | 2.15 | 1.15 | 1.51 | 2.20 | 2.20 | 2.62 | 2.85 |
| running-head residue lines / judgment | 1.82 | 0.20 | 2.80 | 0.13 | 1.70 | 0.15 | 1.62 | 0.30 |
| lone B–H per 1,000 words | 0.23 | 0.13 | 0.37 | 0.14 | 0.18 | 0.09 | 0.53 | 0.06 |
| missing zones | header 8, judgment 7, references 40 | none | header 7, judgment 4, references 40 | header 1 | header 1, judgment 1, references 40 | none | header 2, judgment 1, references 39 | none |
| seconds per judgment | 0.15 | 0.16 | 0.21 | 0.19 | 0.13 | 0.13 | 0.12 | 0.13 |

### Fresh sample (seed 43, 40 per decade)

| metric | 1970s before | after | 1980s before | after | 1990s before | after | 2000s before | after |
|---|---|---|---|---|---|---|---|---|
| era detected (old / new) | 40 out of scope | 40 b | 40 out of scope | 40 b | 40 out of scope | 40 b | 40 out of scope | 40 b |
| errors | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| paragraph integrity | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| size compliance | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| numbered judgments | 52.5% | 17.5% | 55.0% | 22.5% | 42.5% | 25.0% | 77.5% | 50.0% |
| paragraph-gap rate | 81.0% | 28.6% | 81.8% | 66.7% | 82.4% | 70.0% | 80.6% | 40.0% |
| gap rate, first paragraph aside | 52.4% | 0.0% | 63.6% | 44.4% | 47.1% | 30.0% | 67.7% | 40.0% |
| chunks per judgment | 17.70 | 14.18 | 14.50 | 14.80 | 14.15 | 14.53 | 17.62 | 17.68 |
| Held points per judgment | 0.88 | 1.82 | 1.45 | 2.02 | 2.88 | 3.08 | 2.70 | 2.88 |
| running-head residue lines / judgment | 1.38 | 0.17 | 2.40 | 0.17 | 3.05 | 0.17 | 3.50 | 0.17 |
| lone B–H per 1,000 words | 0.29 | 0.12 | 0.56 | 0.29 | 0.31 | 0.08 | 0.26 | 0.09 |
| missing zones | header 14, judgment 7, references 40 | header 5, judgment 1 | header 9, judgment 4, references 40 | header 4, judgment 1 | header 3, judgment 2, references 40 | header 2 | header 3, judgment 1, references 40 | header 1 |

The fresh sample's 1980s B–H rate (0.29) is one judgment about the "R & D
Organisation"; every listed occurrence was read and none is a margin letter.

### Other checks

* **Held-out sample (seed 4242, 13 per decade, 52 judgments):** all Era B;
  integrity and size compliance 100% in every decade; no missing zones.
  Of the 48 lone letters A–H between two lower-case words, 47 were read
  (the listing stops at 10 per judgment): all are text ("Part A", "Item
  B", "Form C", "Wings A and B") or OCR splitting a word ("B ihar"). The
  first line of every page of 10 judgments (99 pages) was read: no running
  head is left.
* **Unnumbered judgments:** 99.9–100% of their blank-line paragraphs (up to
  700 words) lie whole inside one chunk in all three samples.
* **Model limit:** the longest embedded text (header, blank line, chunk) is
  733 words / 4,472 characters, far under the 8,000-token limit. Tokens
  were not counted: no tokenizer was loaded while the GPU build ran.
* **1964–1969 (8 per year):** 1965–1969 all Era B, 1964 3 of 8; integrity
  and size compliance 100%.
* **Era C and D unchanged:** 2025 and 2016 (25 each, seed 5) and 12 per
  year for 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2023, 2024, 2025 and
  2026 (seed 77, 132 judgments): the clean texts and `chunks.jsonl` are
  byte-identical before and after (SHA-256 `c64924df…` for the first,
  `820b24e9…` for the second, 1,356 and 2,630 chunks).

## G6 on Era B

* **Paragraph integrity** (≥ 98%): 100% in every decade of all three
  samples. It counts numbered paragraphs only, which exist in 15–50% of
  Era B judgments; the blank-line paragraphs of the others are 99.9–100%
  intact.
* **Size compliance** (≥ 99%): 99.5–100%.
* **Within the model's limit:** yes by a wide margin (see above), not
  counted with the tokenizer.
* **No furniture left** (hand check): no margin letter inside a sentence
  and no running head in the held-out sample's reviewed text.

So Era B passes the G6 gates on these samples. What G6 does not measure is
the OCR itself (below).

## Remaining failure modes

1. **OCR text.** Words are misread ("cheniical", "1nentioned",
   "LAKSHMJKUTTY"): keyword search and the quote check suffer. Not fixable
   without re-OCR.
2. **Footnotes inline.** "(1) [1967] 1 S.C.R. 120." sits after the last
   body line of a page and so interrupts a sentence that runs on to the next
   page.
3. **No paragraph pinpoints for most judgments.** Era B is cited by page and
   margin letter ("[79-G-H]"). The parser drops the letters and keeps pages
   in the position map; recording each line's margin letter would allow
   page-and-letter pinpoints (Future).
4. **Held points have no paragraph links.** Their pinpoints are
   page-and-letter, so `para_refs` is empty and no editorial → judgment
   links are made.
5. **Paragraph labels:** an unprinted "1." puts the first paragraph under
   label 2's chunk; numbers OCR loses entirely (21–44% of numbered
   judgments still have a gap besides the first paragraph).
6. **Bench line too damaged** in 0–13% per decade (header zone missing; opinions then
   match authors against metadata judges only), and the judgment start is
   missed in up to 1 in 40.
7. **Opinion authors:** 12–20% of judgments that are not orders get no
   author (17 of 139 and 29 of 142 in the two samples; orders have none by
   nature), when the printed name is badly damaged ("f AZAL -ALI. J.,") or
   the line has another form; multi-opinion judgments split only at
   recognised author lines.
8. **Act names:** OCR forms ("Land Refonns Act") and phrases such as "First
   Schedule to the Central Excises and Salt Act" reach the refs.
9. **Short orders** with fewer than 5 margin lines on their pages are out of
   scope (1 of 160 in the seed-42 sample, a 2-page 1984 order).
10. **Indentation paragraphs** are not checked against a ground truth; a run
    of one-line indented paragraphs can merge, and quotations not indented
    on the right can be split.

## Decisions

* **Markers, not years, separate Era A and B.** `era_markers.b.year_from`
  is 1950: margin letters decide, and 1964–1969 judgments, which have the
  Era B layout, are Era B. Era A stays out of scope with reason
  `era_a_scan`.
* **Margin letters by the body column, not fixed bands.** The task asked
  for bands as page fractions; the bands exist (`era_b.left_x/right_x`) but
  are wide, because the measured letter positions overlap the body of other
  scans (letter-size sheets, shifted and skewed pages). The column, measured
  per page and corrected for skew, decides. All tunables are in
  `[era_b]`.
* **Blank lines mark Era B paragraphs** in the clean text, so the existing
  unnumbered fallback (blank-line paragraphs) chunks on real paragraphs
  instead of 400-word sentence windows.
* **The case details and counsel lines between the jurisdiction line and
  the "delivered by" line are in no zone** for Era B (not chunked), like the
  header. Era C keeps them in its judgment zone; unchanged.
* **No version bump.** `clean_text` (ct1) and `chunker` (ch4) stay: Era C
  and Era D output is byte-identical (above), and there is no earlier Era B
  output for a version to tell apart. Era B records carry `era = "b"`. The
  next change to Era B output after an Era B build must bump them. After
  the merge into `p2-p4-integration` the versions are `phase-1`'s ct2 (Era D
  head band) and ch4: on the seed-42 sample the Era B clean texts and
  chunks are identical to this branch's except for that version string,
  and Era C/D output is byte-identical to `phase-1` (132 judgments,
  2010–2026).
* **`Era.B = "b"`** is a new contract value; `contracts/schema.json` is
  regenerated and `db/migrations/003_era_b.sql` widens the `judgments.era`
  check.

## What a full 1970–2009 build needs

* **Judgments:** 21,213 with a PDF (1970s 2,928; 1980s 2,700; 1990s 6,494;
  2000s 9,091). About 1,900 more for 1964–1969 if included.
* **Chunks:** 13.9, 15.9, 13.1 and 15.0 per judgment by decade (mean of the
  two samples), so about **305,000 chunks** (roughly 260,000–350,000 given
  the sample sizes). That is below the data check's corpus-wide estimate
  of 19 per judgment: Era B judgments are shorter.
* **Time:** parse-chunk about 0.16 s per judgment in one process on this
  machine (measured while the GPU build ran): about 1 hour, or 15–20
  minutes with 4 workers. Embedding at the 17 chunks per second measured
  for 2025 on the RTX 2060: about **5 hours**. Load and the HNSW rebuild
  come on top (not measured for this size).
* **Disk:** vectors 305,000 × 4 KB = **1.25 GB** (`build/vectors`, float32);
  chunk records about 2.4 KB each, **0.75 GB**; judgments with clean text
  about 0.3 GB as Parquet. In Postgres, `vector(1024)` rows and the HNSW
  index add roughly 2.5–3.5 GB, and chunk text with its `tsvector` about
  1 GB.
