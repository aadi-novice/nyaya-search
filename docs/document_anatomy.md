# Document Anatomy: Supreme Court Judgments, 1950–2026

* **Author:** aadi-novice
* **Date:** 2026-11-12
* **Method:** `scripts/layout_study.py` extracted the first 3 pages of the
  first PDF in each year's English tar, 77 PDFs in all. The text is in
  `docs/layout_study/<year>_<file>.txt`, and marker counts per year are in
  `docs/layout_study/markers.csv`. I also read samples closely from 1950,
  1962, 1975, 1990, 2005, 2015, 2018, 2023 and 2025, including the last page
  of several.
* **Caveat:** the study used one PDF per year. The era boundaries below are
  approximate and need checking on a larger sample. The text was extracted
  with PyMuPDF, not Docling. Docling may handle margins and columns
  differently.

## Summary

Every PDF is a set of pages cut from the printed *Supreme Court Reports*
(SCR) or, from 2024, the *Digital Supreme Court Reports*. Each PDF holds more
than the court's own words. It also contains:

- editorial material written by the reporter, not the court;
- printing artefacts, such as margin letters, running heads and page numbers;
- in early years, parts of neighbouring cases.

The layout falls into four eras. A chunker that ignores this will:

- index other cases' text under the wrong judgment;
- split sentences on stray margin letters;
- mix the reporter's summary with the court's reasoning.

## Facts that hold for every era

- **`S_` prefix = Supplement volume.** The 2005 `S_` file's running head reads
  "[2005] SUPP. 5 S.C.R.". There are 7,901 such files. They are separate
  judgments, not duplicates.
- **The file name is the SCR citation.** `2018_3_177_195` means year 2018,
  volume 3, pages 177–195, so page numbers inside the PDF match it.
- **Two kinds of text sit in one PDF.** The *editorial* part (headnote,
  "Held" points, list of cases cited) is written by the law reporter and is
  not binding. The *judgment* is the court's own text. Lawyers cite the
  judgment, and quote the headnote only as a summary.

## The four eras

### Era A: about 1950 to the late 1960s. Scanned SCR pages with OCR text

Example: `1950_1_459_519` (*Bharat Bank Ltd. v. Employees*).

- **The PDF does not start or end at this case.** It covers whole printed
  pages, so:
  - page 1 opens with the end of the previous case ("…the appeal must be
    dismissed. Agent for the appellant: S. P. Varma.");
  - the last page holds the title block and headnote of the next case
    ("…THE STATE OF DELHI… Constitution of India, Art 19…").
- **Margin notes are mixed into the text.** Side headings with the case
  name, year and judge ("1950 / Pritam Singh v. The State / May 26",
  "Kania C.J.") land in the middle of sentences.
- **OCR errors are visible.** Examples: "Conititution of In&ia", "Siiprc~
  Court", "Ig6I" for 1961. The data check measured overall text quality as
  similar across decades, but the errors are real.
- **Paragraphs are not numbered.**
- **Order of sections:**
  1. parties;
  2. bench in square brackets;
  3. catchwords separated by dashes;
  4. "Held" summary, sometimes per judge, with dissents noted;
  5. "APPELLATE JURISDICTION: Civil Appeal No…";
  6. counsel;
  7. "The Court delivered judgment as follows";
  8. the opinions, often several, each opening "NAME J.—".

### Era B: about the 1970s to the 2000s. Printed SCR with margin line letters

Examples: `1975_2_873_880`, `S_2005_5_75_82`.

- **Margin line letters A to H** mark positions on the printed page. Pinpoint
  citations use them: "[79-G-H; 80-A]" means page 79, lines G to H, through
  page 80, line A.
  - In extraction they appear either as a column of single letters, or
    inside sentences: "local **C** bullies", "two **D** independent
    witnesses".
  - They must be stripped, but recorded as positions.
- **Running head:** "SUPREME COURT REPORTS [2005] SUPP. 5 S.C.R." and the
  page number on every page.
- **Order of sections:**
  1. parties;
  2. date;
  3. bench;
  4. catchwords, dash-separated, naming the Act and section, for example
     "Penal Code, 1860—Section 302 r/w Section 34";
  5. a facts summary;
  6. "Dismissing the appeal, the Court HELD: 1.1 … [79-G-H]", with numbered
     holdings and page-and-line pinpoints;
  7. cases cited;
  8. jurisdiction;
  9. counsel;
  10. the judgment.
- **Case boundaries are clean.** The PDF ends with the reporter's initials and
  the result: "V.M.K. Appeal Dismissed."
- **Paragraph numbering** appears in some judgments from the 1990s.

### Era C: about 2010 to 2023. Born-digital SCR with a print layout

Examples: `2015_15_147_155`, `2018_3_177_195`, `2023_7_322_346`.

- The text is clean, with no OCR. Margin letters A to H appear as separate
  lines at the start of each page.
- **Running head and footer:** "SUPREME COURT REPORTS [2023] 7 S.C.R." and the
  case title repeated at the foot of the page.
- **Catchwords** begin with the Act name and a colon: "Code of Civil
  Procedure, 1908: s.11 – Res judicata – …".
- **Held points carry paragraph pinpoints:** "1.1 … [Para 29][339-H]". They
  link each editorial holding to the judgment paragraph it summarises.
- **"Case Law Reference" table:** each cited case, with "relied on" or
  "referred to" and the paragraph where it is cited.
- **The judgment has numbered paragraphs** ("19.", "20.").
- **The PDF ends** with the reporter's name and the result ("Devika Gujral /
  Appeal allowed.").

### Era D: 2024 onward. Digital Supreme Court Reports with labelled sections

Examples: `2024_9_770_773`, `2025_1_230_238`, `2026_2_107_127`.

- **Header line with both citations:** "[2025] 1 S.C.R. 230 : 2025 INSC 31".
- **Labelled sections, in order:**
  1. **Issue for Consideration:** a short statement of the question.
  2. **Headnotes†:** dash-separated catchwords, then "Held: …" with
     "[Paras 9, 10, 12.1]".
  3. **Case Law Cited:** each case with its SCR and SCC citations, marked
     "relied on" or "referred to".
  4. **List of Acts**
  5. **List of Keywords**
  6. **Case Arising From:** jurisdiction, case number, and the High Court
     order appealed against.
  7. **Appearances for Parties:** counsel.
  8. **Judgment / Order of the Supreme Court:** the authoring judge (marked
     "*" in the bench and "* Author" in the footnote), then numbered
     paragraphs ("1.<TAB>…"), often under headings such as "FACTUAL
     ASPECT".
- **No margin letters.** The running head reads "Digital Supreme Court
  Reports".

### Marker counts by era (first 3 pages, one PDF per year)

| Marker | Era A (1950s–60s) | Era B (70s–2000s) | Era C (2010–23) | Era D (2024+) |
| --- | --- | --- | --- | --- |
| Labelled sections (Issue, Case Law Cited, Acts, Keywords) | none | none | none | all |
| Neutral citation (`INSC`) | no | no | no | yes |
| "Held" marker | mostly | mostly | yes | yes |
| Numbered paragraphs in the first 3 pages | rare | some | common | common |
| PDF includes text from neighbouring cases | **yes** | no | no | no |
| Visible OCR errors | **yes** | some | no | no |

## What this means for chunking

**Recommendation: chunking that knows the document's zones and era.** It
runs in four stages.

### Stage 1: clean, and keep a position map

Remove the printing artefacts:

- margin letters A to H, including those inside sentences;
- running heads and footers;
- page numbers;
- side notes in Era A;
- reporter initials.

For every character kept, record its source page and its margin letter (Eras
B and C). Citations can then say "page 339, line H", the way lawyers cite,
and the quote check can map back to the PDF.

Handle words split across lines with a hyphen ("manage-/ments").

### Stage 2: cut each PDF to its own case (Era A only)

- **Find this case's start** by matching its title block (parties, bench,
  date) against the metadata `title`, `petitioner`, `respondent` and
  `decision_date`.
- **Find its end** at the next case's title block.
- **Drop the text from the neighbouring cases.** Each piece of it is either
  already its own judgment elsewhere in the corpus or is incomplete.
- **Log every PDF where no start is found.** Do not guess.

### Stage 3: split each case into zones, and treat each zone differently

| Zone | What it contains | How to handle it |
| --- | --- | --- |
| **Case header** | Parties, bench, date, citation, case origin, counsel | Metadata fields. Not embedded as chunks. Attached to every chunk. |
| **Issue** (Era D) | Issue for Consideration | One chunk, tagged `issue`. |
| **Headnote / Held** | Catchwords and the numbered Held points | One chunk per Held point. Tagged `editorial`, keeping its `[Para N]` link to the judgment. |
| **References** | Case Law Cited, List of Acts, Keywords | Not chunked. Stored as structured lists: cited case, "relied on" or "referred to", paragraph; Acts; keywords. Used for filters, keyword search and later citation links. |
| **Judgment** | The court's text, possibly several opinions | Main chunks. See Stage 4. |

### Stage 4: chunk the judgment by paragraph and opinion

- **Split first on opinion boundaries.** Each judge's opinion starts with
  "NAME J.—" or "NAME, J." Tag every chunk with its author and whether the
  opinion is majority, concurring or dissenting. A dissent is not the law.
  The answer generator must know which it is quoting, and the reranker or the
  prompt can down-weight dissents.
- **Within an opinion, split on numbered paragraphs.**
  - Merge short consecutive paragraphs up to about 400 words.
  - Split a paragraph over about 400 words at sentence boundaries, overlapping
    by one sentence.
  - Each chunk keeps its paragraph numbers, for example "paras 11–12". These
    become the citation pinpoint.
- **Old judgments without paragraph numbers:** fall back to paragraphs found
  from blank lines and indentation, then to sentence windows of about 400
  words. Record the fallback on the chunk.
- **Keep sub-headings,** such as "FACTUAL ASPECT" or "Analysis", as a section
  label on each chunk.

### Text that is embedded, and text that is shown

- **Embedded and keyword-indexed:** a short header built from metadata at no
  cost, then the chunk text. Example: "*Jit Vinayak Arolkar v. State of Goa
  (2025) · Judgment · Abhay S. Oka J. · paras 9–10 · Penal Code s.415, s.420*".

  The header puts the case name, date, judge and Acts into every chunk.
  Search can then match "Arnesh Kumar" or "section 420" even when the
  paragraph itself does not name them.
- **Shown to the LLM:** the chunk, optionally expanded to its neighbouring
  paragraphs (small chunks to search, a larger window to read).
- **Used for the quote check:** only the cleaned original text, never the
  header.

### Effect on the plan in the brief

- **The headnote leaks test answers.** If test questions are written from Held
  points, they must not be scored against `editorial` chunks. Score them only
  against the judgment paragraph named in the Held point's `[Para N]` link.
  The link already exists in Eras C and D.
- **Citations should point to judgment text.** The quote check should accept
  quotes from `judgment` chunks. A quote from a headnote needs an "editorial
  summary" label in the answer.
- **The Held → [Para N] links are a free source of test labels** for
  2010–2026. Each Held point is a summary of a named paragraph.
- **Era A needs the most work** (cutting out neighbouring cases, side notes,
  OCR noise) but is the smallest share of chunks. A thin end-to-end version
  can start with Eras C and D (2010 onward), whose layout is clean and
  regular, and add older eras later.

## Approaches considered and rejected

- **Fixed-size token windows** (for example 512): rejected by the author.
  Windows cut across paragraphs and opinions, so citations cannot name a
  paragraph.
- **Semantic chunking** (splitting where the embedding similarity of adjacent
  sentences drops): rejected for v1.
  - It costs an embedding call per sentence.
  - It is unstable on OCR noise.
  - It ignores paragraph numbers, which lawyers need.
- **LLM-based chunking or "contextual retrieval"** (an LLM writes a summary for
  each chunk): rejected for v1. About 730,000 LLM calls would exceed the free
  API quotas. The free metadata header gives part of the benefit.

- **Recursive splitting as the main method** (split on blank lines, then line
  breaks, then sentences, until each piece fits a size limit): rejected as
  the main method, kept as the fallback.
  - It knows nothing about the document's zones or opinions.
  - In Era B it cannot spot margin letters, and in Era A it cannot spot text
    from neighbouring cases.
  - Paragraph numbers survive only if they are tracked separately.
  - It is the right tool inside Stage 4, for splitting an over-long
    paragraph or an old judgment that has no paragraph numbers.
- **Agentic chunking for every judgment** (an LLM reads each judgment and
  decides where to cut): rejected for the whole corpus.
  - 38,397 judgments at about 7,000 tokens each is roughly 270 million input
    tokens, far beyond free API quotas.
  - Boundaries change between runs, and the documents already label their
    own structure.
  - Kept as a candidate only for the small set of Era A PDFs where the rules
    cannot find where the case starts or ends.

## Open questions

1. Where exactly do the eras begin and end? Classify each PDF by its markers
   (margin letters present, labelled sections present, neutral citation
   present), not by year.
2. Does Docling keep, drop or separate the margin letters, running heads and
   side notes? Test one PDF from each era.
3. How reliable is opinion-boundary detection in Era A, where several
   opinions and side notes are interleaved?
4. Should `editorial` chunks be searchable at all in v1? Or should they only
   feed metadata and test labels?
