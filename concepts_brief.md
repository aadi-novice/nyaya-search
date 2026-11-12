# Legal Research Assistant over Indian Supreme Court Judgments

* **Author:** aadi-novice
* **Date:** 2026-11-11

## Overview

This is a solo resume project. "Free" means zero cost. The tools do not have
to be open-source or run locally.

**Product claim.** A lawyer asks a research question and gets one of two
things:

- an answer whose every citation points to a real passage the system
  retrieved; or
- an honest "no relevant judgment found".

**Engineering claim.** Search keeps finding the right passages as the index
grows from about 10,000 chunks to a million or more.

The lawyer types the question into a web page, which calls a small API. The
system finds the relevant passages in Supreme Court judgments. An LLM called
through a free hosted API writes an answer that cites only those passages. Code checks every citation
before the answer is shown.

**Source data.** The AWS Open Data "Indian Supreme Court Judgments" set:

- 74 GB of PDFs, packed into one tar file per year from 1950 onward;
- English and regional-language versions;
- metadata in Parquet and JSON.

The bucket holds 43,578 judgment metadata files, so there are about 43,000
judgments, some of them duplicates. The dataset is still growing: a 2026
partition exists, so a snapshot date must be pinned. High Court judgments
come later.

**Scale reality check.** "Millions" applies to chunks, not judgments. About
43,000 judgments split into passages of roughly 400 words gives an estimated
0.5 to 1 million chunks. This must be measured before it is promised.

**The PDFs are messy:**

- some pages are scanned images that need OCR;
- the same judgment appears more than once;
- every page repeats headers, footers and stamps;
- metadata is inconsistent.

**Build approach.** First build a thin version that works end to end on one
year or decade of data. Then deepen each layer in this order, with one gate
per layer:

1. parse;
2. chunk;
3. hybrid search;
4. rerank;
5. generate answers with the citation lock;
6. refusal and notices;
7. API and web page.

**Scope risk.** For a solo developer, the main risk is ending with seven
half-built layers. Two rules guard against it:

- The thin end-to-end version comes first, so a demo exists early.
- Every gate has a time limit and a rule for what happens when it fails.

## Analysis

### Part A: Search at scale

#### Five built-in conflicts, and how to get out of each

1. **A bigger index covers more, but also adds more passages that look almost
   right.** Before indexing:
   - merge duplicate copies into one canonical judgment;
   - strip lines that repeat across pages.

   Optionally, search in two steps: find the judgment first, then the passage
   inside it, using metadata as a filter. The cost: if step one misses the
   judgment, step two cannot recover it, and a wrong metadata field silently
   filters out the right answer.
2. **The test set must stay fixed, but chunking will keep changing.** Store
   each correct answer as a position in the source text, not as a chunk
   number:
   - canonical judgment ID;
   - PDF content hash;
   - page;
   - character span or exact quote.

   A retrieved chunk counts as a hit if it overlaps that span. The cost: the
   cleaned text needs a version number, because re-running OCR or cleaning
   moves character positions.
3. **The claim says "millions", but the corpus may be smaller.** Do not cut
   chunks smaller to inflate the count. Instead, add data year by year and
   plot the curve. If more size is needed, add "distractor" documents from
   the High Court dataset. "Quality holds at 800,000 real chunks" is a
   stronger claim than "3 million tiny chunks".
4. **Free tools versus OCR quality.** Run OCR only on pages whose text layer
   is missing or fails a quality check. Comparing duplicate copies of a
   judgment gives a free estimate of the OCR error rate.
5. **Correct answers need labelling, but there is no labelling budget.**
   Judgments cite earlier judgments, so the text around a citation can serve
   as a question and the cited judgment as its answer. Headnotes are another
   source. Back these with a small hand-checked set at the passage level.

#### Blocking risks for search

- **A drop in quality can have two causes, and the test must separate them.**
  - Approximate vector search may skip results to save time.
  - Similar legal passages may crowd each other out.

  Report three numbers at every size step: exact (brute-force) search,
  approximate search, and a plain keyword baseline (BM25).
- **Duplicates can fake a decline.** If a second copy of the right judgment
  counts as a miss, quality seems to fall as the corpus grows. Every judgment
  needs one canonical ID, and a rule for which copy wins, before the test set
  is frozen.
- **No compute or storage budget yet.**
  - OCR at about 1 page per second over a million pages takes about 11 days.
    Whether that applies depends on running OCR only on pages that need it.
  - Raw vectors take about 3 GB for 1 million chunks of 768 numbers each, and
    about 15 GB for 5 million.
  - About 108 GB of disk will be free after the download.

  Run a timed pilot on one year's tar first.
- **No record of where each chunk came from.** Every chunk should record:
  - the source tar file and the file inside it;
  - page;
  - extraction method (text layer or OCR) and OCR confidence;
  - language;
  - pipeline config version.

#### Serious risks for search

- **Cleaning can delete what lawyers cite.** Removing headers and stamps can
  also remove paragraph numbers, page numbers and citations. Keep a map from
  cleaned text back to source positions, and split chunks on numbered
  paragraphs.
- **Garbage OCR text pollutes search results.** Check text quality per page,
  tag low-quality pages, and break results down by quality.
- **Test questions can leak their answers.** Questions written by the system's
  author, or generated by a model from the answer text, repeat the answer's
  wording. The test set needs:
  - paraphrasing rules;
  - for citation-based questions, the citing passage removed from the
    candidates.
- **The scale target is inside the noise.** With 200 to 300 questions, a 90%
  success rate has about ±3.5 points of sampling error, so a 5-point drop
  cannot be reliably detected. Use the same questions at every size step and
  report a paired bootstrap confidence interval.
- **Windows-specific problems.**
  - long file paths;
  - how Python starts worker processes;
  - OCR, the embedder, the reranker and the LLM all competing for one GPU.

### Part B: Answers, guardrails and the product

#### Blocking risks for answers

- **A citation lock that only checks judgment names passes by construction.**
  A real judgment cited for something it never says would still pass. And if
  failing citations are simply deleted, "zero made-up cases" is always true.

  Proposed fix:
  - The LLM outputs only passage IDs taken from the passages in its prompt.
  - Code turns each passage ID into the case name, paragraph and page.
  - Optionally, require each claim to include a short exact quote that must
    appear word for word in the cited passage.
  - Catch case names the LLM writes in free text (such as "X v. Y", or AIR,
    SCC, SCR or neutral-citation patterns) using an alias table built from
    the metadata.

  The real metric is how often the LLM broke the rule **before** filtering.
- **Refusal can be gamed, and its threshold can leak test answers.**
  - A system that refuses everything scores 100% on "refuses unanswerable
    questions".
  - Reranker scores are not calibrated probabilities. The best score also
    tends to rise as the index grows, so refusal behaviour can shift exactly
    as the scale test grows the index.
  - If the threshold is tuned on the frozen test set, the final numbers are
    contaminated.

  Proposed fix:
  - Also measure false refusals of answerable questions.
  - Tune the threshold on a separate development set.
  - Store the threshold together with the reranker version.
  - Report refusal rates at every size step.

#### Serious risks for answers and the product

- **"Supported claims" is incomplete.**
  - Sentences with no citation escape the check.
  - The automatic judge is undefined, and it might be the same model grading
    its own answers.
  - "90%" has no stated unit.

  Proposed fix:
  - The unit is one sentence plus the passage it cites.
  - Also report the share of factual sentences with no citation.
  - Use a judge from a different model family than the one writing answers.
  - The judge's numbers count only if it agrees with the hand checks at a
    rate set in advance.
- **Notice triggers are undefined, and one would fire almost always.**
  - Detecting "personal legal-advice questions" needs a classifier and
    labelled examples.
  - Most criminal judgments predate 1 July 2024, so the IPC/CrPC/Evidence Act
    notice would fire on nearly every criminal answer.

  Proposed fix:
  - Make the legal-advice and overruled-status notices standing text shown on
    every answer.
  - Detect the statute notice with fixed rules on the cited passages. If it
    fires on more than about half of answers, make it standing text too.
- **The 15-second target has no budget behind it.** The LLM, embedder and
  reranker share one GPU: an NVIDIA RTX 2060 with 6 GB of memory. A 7 to 8
  billion parameter LLM compressed to 4 bits alone takes about 5 GB, which
  leaves almost nothing for the reranker and embedder. Docling's layout
  models also use the GPU during parsing. Ollama may
  unload models between requests. Five to ten passages of 400 words is about
  4,000 to 8,000 prompt tokens. Before the target is frozen:
  - write a time budget per stage;
  - measure the LLM's tokens per second;
  - define whether "arrives" means the first word or the full answer.
- **Free hosted LLM APIs bring their own risks.**
  - **Rate limits.** Requests per minute and per day are capped. One
    evaluation run of 300 questions, times several variants, plus the judge,
    can exceed a day's quota.
  - **Reproducibility.** Providers change or retire free models without
    notice, so scores from different months may not be comparable. Record the
    provider, model name and date with every run.
  - **Privacy.** Questions leave the machine. The judgments are public, but a
    lawyer's question may contain client details. Free tiers may also keep or
    train on what is sent. Show a notice and read each provider's terms.
  - **Lock-in.** A free tier can disappear. Call every provider through one
    small interface, such as an OpenAI-compatible client, so switching
    providers is a config change.
- **Free cloud GPUs suit batch jobs only.**
  - Kaggle and Colab sessions end after hours, and the GPU quota is weekly.
  - Batch jobs must therefore run one year's tar at a time, save their
    results, and be able to resume.
  - These notebooks cannot host the live API.
- **Streaming conflicts with citation checks.** A citation cannot be taken
  back once it is on screen. Return the full checked answer, or stream it and
  then replace it with the checked version.
- **No per-request trace.** Without one, nobody can explain a refusal or a
  removed citation. Log each request:
  - query, index version and config version;
  - retrieved chunk IDs with keyword, vector and rerank scores;
  - top score against the threshold;
  - citations removed by the lock;
  - notices shown;
  - time per stage.
- **No versioning or rollback across layers.** Re-parsing moves character
  spans. Test answers, citation IDs, API responses and the index must all
  record the cleaned-text version. Keep the previous index so a bad rebuild
  can be rolled back.
- **The labelling workload is too large for one person.** The plan adds up to
  200 to 300 test questions, an unanswerable set, 50 hand checks, judge
  validation and 30 lawyer ratings. Treat the lawyer review as qualitative
  evidence, not a gate.

### What the data already shows

**Measured on 2026-11-11.** The full numbers are in
`docs/data_check_report.md`, produced by `scripts/data_check.py`.

- **Corpus size.** 38,397 distinct judgments. The 43,578 metadata rows
  include 5,181 judgments listed twice, in adjacent year folders. Their PDFs
  are byte-identical in all 300 pairs checked.
- **Canonical ID.** `path`, `cnr` and `citation` are each unique per
  judgment, never blank, and never conflict. `nc_display` and `case_id` are
  blank in 162 rows.

  The ID question is therefore settled: use `path`, keep one copy per
  `path`, and treat CNR and citation as aliases.
- **Correction (2026-11-12, `docs/docling_findings.md`):** in the 1950s–60s
  the existing text layer is itself old, poor OCR. Re-running OCR made the
  1953 sample readable, at 6.5 seconds per page on the CPU. Era A needs
  fresh OCR on Kaggle GPUs. Later eras do not.
- **OCR is mostly unnecessary (original finding, superseded for Era A).** In a seeded sample of 200 judgments (2,744
  pages), only 0.2% of pages had under 200 characters of text. Text quality
  does not drop for older decades. Docling can run with OCR off and fall back
  to OCR only for empty pages, so the 11-day OCR estimate no longer applies.
- **The size target.** About 13.7 pages and 5,300 words per judgment. That
  comes to about 527,000 pages and **about 730,000 chunks** of roughly 400
  words. "Millions" of chunks is not reachable with Supreme Court data alone.
  It needs the High Court data.
- **Tar access works.** All 77 English tars match their `index.json`.
  Scanning every tar header takes 25 seconds and gives each PDF's byte
  offset, so any single PDF can be read without unpacking.

**Earlier findings:**

- **The same judgment can appear in two year folders.** For example,
  `year=1958/1959_1_1162_1167.json` and `year=1959/1959_1_1162_1167.json`.
  Year folder plus file name is not a unique key.
- **The neutral citation field (`nc_display`) is sometimes empty.** It was
  blank in 41 of the early files.
- **Case details live in the `raw_html` field.** Title, SCR citation, CNR,
  bench, decision date, case number and outcome must be parsed from it. Those
  same fields are what the answer needs to display a citation.
- **The HTML snippet differs by era.** It is garbled OCR in a 1954 sample but
  a clean headnote in a 2005 sample. Headnotes are useful, but they are also
  a source of test questions, so keeping them risks leaking answers.
- **Random access into the tars is not yet confirmed.** Check one
  `*.index.json` file once the tars arrive.

### The ideal end state

- One pipeline that can be paused and resumed, and that streams each year's
  tar into deduplicated, cleaned, versioned text.
- A frozen test set and development set that score every change.
- An API that returns either an answer whose every citation is a checked
  passage ID with page and span, or a refusal, together with the notices and
  version fields.
- One static web page that shows it.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * A thin end-to-end version on one year or decade, built first.
  * Deepening each layer in order, each with one gate and a time limit.
  * **API:** one synchronous `POST /ask` on localhost. It returns:
    * the answer;
    * citations (passage ID, page, span);
    * a refusal flag;
    * notices;
    * version fields.
  * **Web page:** one static page that calls the API.
  * A per-request trace log.
  * The answer prompt is versioned and frozen together with the test set.
  * **Scale metrics:** retrieval results at fixed, nested size steps (for
    example 10k, 100k, 1M chunks, full corpus).
    * Every step contains all test-answer judgments.
    * Distractors are added in a fixed order with a recorded seed.
  * A frozen, versioned test set plus a separate development set, balanced
    across decades, text-layer versus OCR pages, and question types (holding,
    citation lookup, statute, unanswerable).
  * Reproducible with one command and a pinned dataset snapshot date, with
    honest corpus statistics: judgments, chunks, OCR share, duplicate rate.
* **Success Criteria (proposed):** These are starting points.
  * **Process:** Run a baseline, adjust once, then freeze. Measure gates on the
    pilot slice and keep full-corpus numbers for the final run. Freeze earlier
    layers before tuning later ones.
  * **When a gate fails:** pick one of these and write down the reason:
    * keep working until the time limit runs out;
    * lower the target once;
    * stop and ship the layers that passed.
  * **Parse and chunk:**
    * share of pages extracted;
    * text-quality pass rate;
    * OCR error rate from duplicate copies;
    * share of paragraph numbers preserved;
    * the map back to source positions works in both directions.
  * **Search:** results are merged by canonical judgment ID before scoring.
    Report both judgment-level and passage-level results.
    * **Before reranking:** the right judgment is in the top 50 for at least
      90% of answerable questions.
    * **After reranking:** the right judgment is in the top 5 for at least 75%.
    * **Scale claim:** from 10k chunks to the full corpus, both numbers drop
      by no more than 5 points, measured on the same questions with a paired
      bootstrap confidence interval.
  * **Citation lock:**
    * after filtering, zero citations that do not resolve to a passage in
      the prompt;
    * the rule-break rate before filtering is reported and tracked.
  * **Supported claims:** at least 90% of cited sentences are supported by
    their passage.
    * The judge is a fixed model and prompt from a different family than
      the answer model.
    * Its numbers count only if it agrees with 30 to 50 hand-checked answers
      at a preset rate.
    * The share of uncited factual sentences is also reported.
  * **Refusal:**
    * correct refusal on at least 80% of unanswerable questions;
    * false refusal on no more than about 10% of answerable ones;
    * the threshold is tuned on the development set only.
  * **Notices:**
    * the standing notices are always shown;
    * the statute notice meets a precision target.
  * **Latency:**
    * the full answer within 15 seconds at the 95th percentile;
    * measured warm, one request at a time, on a fixed query set;
    * with fixed limits on input and output tokens;
    * confirmed against a written per-stage budget.
  * **Lawyer review:** 30 answers rated against a written rubric. This is
    qualitative evidence, not a gate.
* **Out-of-Scope Non-Goals:**
  * Consumer users, personal legal advice, and paid APIs or services.
  * Accounts, logins, chat history, multiple users at once, and a polished
    interface.
  * Streaming, unless the streamed text is replaced by the checked answer.
  * A classifier for legal-advice questions.
  * Tracking whether a judgment was later overruled.
  * Mapping old IPC/CrPC/Evidence Act sections to the new codes, unless a
    free official table turns up.
  * Apache Tika, until a count shows non-PDF files actually exist.
  * Adding new judgments in real time.
  * Regional-language judgments in the scale test. Either defer them or run
    them as a separate track.

## Alternatives Weighed

* **Product shape:** a search-only benchmark or an end-to-end assistant.
  * Rejected by author: search-only.
  * Reason: a benchmark proves the scale claim but gives a user nothing to
    use and nothing to demo.
  * Cost of the choice: answer generation, guardrails, an API and a web page,
    each with its own testing.
  * The scale benchmark stays as one part of the product.
* **Build sequence:** strict layer by layer, or a thin end-to-end version
  first. Decided by author: thin version first. Strict layering keeps
  the demo blocked behind a full corpus parse that may take days.
* **Answer and citation form:** three options.
  1. Free-form answer, checked after generation.
  2. The LLM outputs passage IDs only, and code writes the citations.
  3. Quote first, then explain, with an exact-text check on each quote.

  Options 2 and 3 make the citation lock hold by design, at the cost of
  stiffer prose.

  Decided by author: passage IDs plus quotes. Option 1 was rejected because
  it lets a real case be cited for something it never says.
* **Refusal signal:**
  * a fixed reranker-score threshold;
  * a threshold recalibrated at each size step;
  * combined signals:
    * the gap between the top score and the k-th score;
    * agreement between keyword and vector results;
    * the LLM producing no valid citation.

  Undecided.
* **GPU placement:**
  * keep all models loaded and use a smaller LLM;
  * swap models in and out per question;
  * run the embedder and reranker on the CPU and keep the GPU for the LLM.

  Resolved by author:
  * the answer LLM moves to a free hosted API;
  * heavy batch jobs move to free cloud notebooks;
  * the local 6 GB GPU runs only the embedder and reranker for live queries.

  Rejected: a small local LLM, because its answers are weaker.
* **Judge for supported claims:** a local LLM from a different family, a
  model that checks whether a passage supports a sentence, or hand checks
  only. Undecided.
* **Delivery:** a full answer returned at once, or streamed and then replaced.
  Reviewers lean toward returning the full answer at once.
* **How to grow the index:** more real data, distractors around a fixed answer
  set, or smaller chunks. Reviewers favour distractors plus real data added
  year by year. Smaller chunks were rejected because they inflate the count
  without making the test harder.
* **Where correct answers come from:** written by a lawyer, generated by a
  local model, or taken from citation links and headnotes. Reviewers favour
  citations and headnotes, plus a small hand-checked set.
* **OCR coverage:** OCR everything, OCR only pages that fail a text check, or
  skip scanned pages and report how many were skipped. Current lean: only
  pages that fail.
* **Regional languages:** ingest all, defer, or run a separate track.
  Undecided.
* **Boilerplate removal:** strip it up front, or first measure whether it
  reaches the top results. Undecided.
* **What a test answer points to:** resolved: canonical judgment ID plus text
  span.

## Key Decisions & Staging for Next Steps

**Decided by the author:**

* **Project:** a free, open-source, end-to-end legal-help assistant for
  lawyers, built solo.
* **Data:** Supreme Court data first, High Court data later. Data is stored
  under `docs/` and downloaded by `scripts/download_sc.py`.
* **Parsing (decided 2026-11-12): a mixed setup.**
  * **All eras:** PyMuPDF with page coordinates. Margin letters, running heads
    and page numbers are removed by their position on the page, and a map
    back to page and line is kept for citations.
  * **1950s–60s:** Docling with OCR, run on Kaggle GPUs.
  * **2024+ sections:** found by their fixed labels.
  * The first Kaggle job is a speed and quality test of both parsers on 20 to
    50 PDFs per era. See `docs/docling_findings.md`.
  * Apache Tika is reserved for non-PDF files.
* **Judgment ID:** `path`, with one copy kept per duplicated judgment. See
  `docs/data_check_report.md`.
* **Chunking:**
  * **Order of splitting:** structure first (by zone, then opinion, then
    numbered paragraph).
  * **Recursive splitting** is used only inside over-long paragraphs and
    unnumbered old judgments.
  * **An LLM** is used only for 1950s–60s cases where the rules cannot find
    where the case starts or ends.
  * **Size limits** are enforced in a separate split-and-merge pass.
  * **Context header:** each chunk gets a header built from metadata. No LLM
    writes it.
  * See `docs/document_anatomy.md`.
* **Parse-and-chunk gate:**
  * **Paragraph integrity:** the share of judgment paragraphs not cut by a
    chunk boundary.
  * **Size compliance:** the share of chunks within the size limits.
* **Retrieval additions (see `docs/research_notes.md`):**
  * BM25 is a full partner in hybrid search, not a fallback.
  * Fuzzy quote lookup.
  * "[Para N]" and citation strings are resolved to stored links and fetched
    at answer time.
* **Error analysis:** every wrong answer is labelled as a hallucination, a
  retrieval error or a reasoning error.
* **Thin end-to-end slice:** judgments from **2010 to 2026** (Eras C and D,
  the clean layouts). Older eras are added once the end-to-end version works.
* **Test set:** deferred by the author. It must exist before the search gate
  is measured, because every Success Criterion depends on it. IndicLegalQA
  (CC BY 4.0) is the leading candidate.
* **Regional-language judgments:** out of scope for v1.
* **Additions from the "RAG at 10 Million Documents" review (2026-11-12):**
  * **Rule-based query router, with no LLM:**
    * a citation string → direct metadata lookup;
    * a quoted fragment → fuzzy quote lookup;
    * anything else → hybrid search.

    Every route still retrieves and cites. Nothing is answered from the
    LLM's memory.
  * **Optional metadata filters on `/ask`:** year range, judge, Act,
    outcome. Every applied filter is logged in the trace.
  * **Tables are kept whole** as one chunk unit, and never split mid-row.
  * **Adversarial test set** of about 20 questions: injection attempts,
    requests for personal legal advice, and false-premise questions.
* **Chunking:** follows the document's structure, not a fixed token count.
* **Search:** hybrid, combining keyword and vector search.
* **Reranking:** with a cross-encoder.
* **Guardrails:** citation lock, refusal when evidence is weak, and scope
  notices.
* **Output:** an LLM-written answer with citations, served by an API and a
  web page.
* **Citation form: passage IDs plus quotes.**
  * **What the LLM writes:** every factual sentence cites a passage label
    from its prompt, such as `[P3]`, together with a short quote from that
    passage.
  * **What code does:**
    * turns the label into the case name, citation, paragraph and page,
      using the stored metadata;
    * checks that the quote appears word for word in the passage.
  * **Before matching:** normalise both texts, so that OCR noise does not
    cause false failures:
    * whitespace;
    * letter case;
    * words hyphenated across a line break;
    * curly versus straight quotes.
  * **Quote length:** set a minimum and a maximum (for example 5 to 40
    words). This blocks trivial quotes such as "the court held".
  * **Sentences that fail the check** are removed or flagged. Which of the
    two is open question 2.
  * **What is measured:** how often the rule was broken before the check.
* **"Free" means zero cost**, not necessarily open-source or local.
* **Compute split:**
  * **Batch jobs** (Docling parsing, OCR, embedding the corpus) run on free
    cloud GPU notebooks such as Kaggle or Colab.
    * Each notebook downloads one year's tar directly from the public bucket.
    * It saves cleaned text and vectors, which are then brought back to the
      local machine.
  * **The live assistant** (search, reranking, API, web page) runs on the
    local machine: a 6 GB RTX 2060.
* **Answer LLM:** called through a free hosted LLM API. The provider is
  undecided, and every call goes through one swappable interface.
* **Build sequence:** a thin version that runs end to end comes first, then
  each layer is deepened.
* **Parallel coding agents:** the layers will be built by several coding
  agents in parallel. Before any agent starts, these shared contracts are
  written down:
  * canonical judgment ID;
  * chunk format;
  * citation format;
  * API response format.
* **Still open:** the vector store, embedding model, OCR engine, LLM provider
  and framework. Only decision criteria are recorded.

**Open questions for `/egm-draft`, in rough order:**

1. **Canonical judgment ID.** Resolved by measurement: `path` is the key
   (see `docs/data_check_report.md`).

   * The `S_` prefix on 7,901 PDFs marks SCR Supplement volumes. These are
     separate judgments, not duplicates. See `docs/document_anatomy.md`.
   * Still open: how to handle the 7 metadata rows that have no PDF.
2. **Citation details.** The form is decided: passage ID plus quote.
   * Is the "retrieved set" the passages placed in the prompt?
   * When a sentence fails the quote check, is the sentence removed, or is it
     only flagged?
   * What are the exact quote-length limits and normalisation rules? What about cases that a
   retrieved passage itself cites?
3. **Which LLM?** Criteria:
   * an open-weight licence;
   * fits in GPU memory alongside the reranker;
   * context length;
   * deterministic output.

   How many passages go into the prompt?
4. **Which judge?** Which model and prompt, and what agreement rate with the
   hand checks makes its numbers count?
5. **Test set and development set.** Size, mix of question types, how
   unanswerable questions are made, the anti-leakage rule, and how often the
   refusal threshold is recalibrated.
6. **Notice triggers.** Does each notice scan the question, the answer or the
   cited passages?
7. **The time budget.** How do the 15 seconds split across stages? How much
   GPU memory does the machine have?
8. **Gate failures.** Time limit, a recorded lowering of the target, or stop
   and ship?
9. **Docling and OCR.** Is OCR set inside Docling to run only on failing
   pages, or run as a separate stage? This decides whether the 11-day
   estimate applies. Is Tika a fallback when Docling fails on a PDF?
10. **Headnotes.** Keep the headnote in `raw_html` for display, given that it
    can leak test answers?
11. **Snapshot date.** Which dataset date is pinned?
12. **Is "millions" a hard target?** Measure chunk counts on about 200 sample
    judgments first. Are High Court distractors allowed?
13. **LLM provider and quotas.** Which free hosted API, and which model, for
    answers? Which provider, from a different model family, for the judge?
    Do the daily limits cover a full evaluation run? If not, how is the run
    split across days?
14. **Storage policy.** Maximum chunk count, vector compression, how many index
    versions to keep, and whether raw tars are kept.
15. **Checks once the download finishes.** Confirm that a single PDF can be
    read from a tar using its `*.index.json`, and check free disk space.

**Criteria for choosing the remaining tools (not ranked):**

* **Cost:** zero. A free tier is acceptable.
* **Free-tier limits:** must cover an evaluation run, or the run can be split
  across days.
* **Platform:** works on Windows with CUDA and fits in GPU memory alongside
  the other models.
* **Search:** exact and approximate search over the same vectors, filtering
  by metadata, and a keyword or combined keyword-plus-vector option.
* **Index:** adds partitions without a full rebuild, and keeps vectors on disk
  or compressed.
* **Indian languages:** OCR for Indian scripts and multilingual embeddings.
* **Measured performance:** pages per second, chunks per second, tokens per
  second, peak RAM and GPU memory, and bytes per vector.
* **Determinism:** the same input gives the same index and the same answer.

**Suggested draft structure:**

* Lead with the product claim: a cited answer or an honest refusal. Put the
  scale claim second.
* Give the test method its own section before the technical plan.
* Write one subsection per layer (parse, chunk, search, rerank, generate,
  guardrails, API and web page). Each lists inputs, outputs, failure cases,
  its gate and its time limit.
* Keep a register that names, for each undecided tool, the step that will
  decide it.
