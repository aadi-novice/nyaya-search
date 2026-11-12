# Stage 5: Initial Zoom Out Review

*   **Target:** `docs/designs/nyaya_search.md`, the full file (lines 1–1858).
*   **Reviewers:** `egm-zoom-out` (simplicity and scope) and
    `egm-inventive-catalyst` (trade-offs the design accepts without
    testing). Both completed.
*   **Background reading given to both reviewers:**
    *   `docs/data_check_report.md`
    *   `docs/document_anatomy.md`
    *   `docs/docling_findings.md`
    *   `docs/research_notes.md`
*   **Findings:** 1 Blocker, 13 Major, 13 Nit, after duplicates were
    removed. The inventive-catalyst reviewer can only rate findings Major or
    Nit.

## Where the reviewers overlap

*   **The router:** both say the quote-lookup rule takes short keyword
    queries away from hybrid search (ZO-3 and IC-2).
*   **Size steps:** both say the G4 size steps conflict with the "keep at
    most two index versions" rule (ZO-6 and IC-3).
*   **Opinion type:** both flag the rules that decide opinion type (ZO-N3
    and IC-N3).
*   **Lookup routes:** both say the citation and quote routes do not fit
    the `/ask` pipeline (ZO-4 and IC-2).

## Where the reviewers disagree (the author decides)

*   **The quote route.** Zoom-out says to remove it from v1. Inventive
    catalyst says to keep it as a third ranked list, fused with the other
    two.
*   **Storing the G4 size steps.**
    *   **Zoom-out:** put each step in its own schema, with its own vector
        index.
    *   **Inventive catalyst:** load the full slice once, with a step
        column, and build one partial vector index per step.

## What both reviewers support

*   one PostgreSQL store;
*   PyMuPDF parsing with page coordinates;
*   chunking that follows the document's structure;
*   a rule-based router, with no route that skips retrieval;
*   no agents.

---

## Lens: egm-zoom-out (simplicity and scope)

### ZO-1. The deferred development set blocks Phase 1, and a free source of labels is unused (Blocker)

*   **Where:**
    *   Goals intro (L86–90);
    *   Open Questions, "Test set" (L217–224);
    *   tools table (L714–718);
    *   files 47 and 50;
    *   Verification Plan.
*   **Problem:** these Phase 1 steps all need the development set, but the
    test set is deferred:
    *   choosing the embedding model;
    *   setting the refusal threshold;
    *   tuning the chunk size limits;
    *   the BM25 trigger;
    *   validating the judge.

    Meanwhile `document_anatomy.md` (L238–239) names a free source of
    labels for 2010–2026 that the design never uses: the links from Held
    points to judgment paragraphs.
*   **Fix:**
    *   Make the development set a Phase 1 deliverable.
    *   **Build it automatically:** paraphrase each Era C or D Held point
        into a question.
    *   **Gold answer:** the paragraph the Held point links to.
    *   **Scoring:** editorial chunks never count as hits.
    *   **Later:** add IndicLegalQA if the download works.
    *   **Until then:** use one default embedder and one default reranker,
        marked provisional.
    *   **See also:** IC-4 and IC-6.

### ZO-2. v1 is too large for one person, and "thin" is not defined as a list of what to cut (Major)

*   **Where:** the build order (L1065–1070) and the target files
    (L1082–1821).
*   **Problem:**
    *   the plan has more than 52 files;
    *   Phase 0 has 12 deliverables before any PDF is parsed;
    *   Phase 1 adds three query routes, link expansion, opinion typing,
        table detection, notice-rate tracking and version activation.
*   **Fix:** define Phase 1 as the minimum path only:
    1.  manifest;
    2.  parse (Eras C and D);
    3.  zone;
    4.  chunk;
    5.  embed;
    6.  load;
    7.  hybrid search;
    8.  rerank;
    9.  LLM;
    10. citation check;
    11. page.

    Mark everything else "Phase 2+".

### ZO-3. Remove the quote-lookup route and the sentences trigram table from v1 (Major; overlaps IC-2)

*   **Where:**
    *   CUJ 3;
    *   router (L559–567);
    *   the `sentences` table and its trigram index;
    *   the sentence output of file 20;
    *   `fuzzy_sentences` in file 28.
*   **Problem:**
    *   Trigram matching mainly absorbs OCR noise, and the 2010–2026 slice
        is born digital.
    *   Postgres phrase search on `chunks.tsv` already handles exact
        phrases.
    *   The "under about 8 words with no question word" rule sends short
        lawyer keyword queries, such as "anticipatory bail section 438",
        away from hybrid search.
*   **Fix:**
    *   v1 has two routes: citation lookup, and hybrid search with quoted
        phrases passed into the keyword query.
    *   Defer the trigram index to the Era A phase.
    *   If the quote route is kept anyway, trigger it only on explicit
        quotation marks.

### ZO-4. The citation and quote routes conflict with the `/ask` pipeline and the AskResponse contract (Major)

*   **Where:** CUJ 2, CUJ 3, pipeline steps 2–7 (L1593–1610) and
    AskResponse (L537–552).
*   **Problem:**
    *   The pipeline sends every route through reranking and the refusal
        gate, so a direct citation lookup could be refused because of a low
        reranker score.
    *   AskResponse has no shape for "here is a judgment" or "here are
        matching passages" without LLM sentences.
*   **Fix:**
    *   Make citation lookup either redirect to `GET /judgment/{path}`, or
        return `status: lookup` with a `judgment` field.
    *   Skip reranking, the refusal gate and the LLM for that route.
    *   Do the same for quote matches if that route stays.

### ZO-5. Checking free-text party names will mark valid sentences unverified (Major)

*   **Where:** the citation check, step 5 (L639–642); "normalised party
    names" in the aliases (L457); file 37, step 4.
*   **Problem:**
    *   Names such as "State of Maharashtra" and "Union of India" appear in
        thousands of judgments.
    *   Treating them as rule breaks pushes correct sentences into the
        unverified list and raises refusals, which works against G5.
*   **Fix:**
    *   In v1, scan only for reporter-citation patterns (SCR, INSC, SCC,
        AIR) and full "X v. Y" case names, compared against the retrieved
        passages.
    *   Drop party names from the alias table.

### ZO-6. The index-version design conflicts with G4, and distorts what G4 measures (Major; overlaps IC-3)

*   **Where:**
    *   Load (L480–487);
    *   the shared HNSW index keyed by index_version;
    *   the "at most two versions" rule (L987, L1462);
    *   size steps (file 46);
    *   the Docker memory limit (L1111).
*   **Problem:**
    *   **(a) Too few versions.** G4 needs three or more size steps alive at
        once, plus production. Only two versions are allowed, and the loader
        deletes older rows, so running G4 evicts the production index.
    *   **(b) Filtered search skews the result.** One vector index serves
        every version and is filtered with `WHERE index_version = …`.
        Filtered pgvector search can return fewer or worse results. That is
        the same accuracy loss G4 is trying to measure, and it also affects
        the filtered queries in CUJ 4.
    *   **(c) Incomplete versioning.** `aliases` and `refs` have no
        `index_version` column.
*   **Fix (the author chooses one):**
    *   **Zoom-out:** one schema per size step, each with its own vector
        index. Use a backup or a second schema for rollback.
    *   **Inventive catalyst:** load the full slice once, with a
        `step_rank` column. Build partial HNSW indexes,
        `CREATE INDEX … WHERE step <= n`, and check with `EXPLAIN` that the
        query planner uses them.
*   **Under either option:**
    *   exempt evaluation steps from the two-version rule;
    *   note the filtered-search risk for CUJ 4 (pgvector's iterative scan
        settings);
    *   note that `ts_rank_cd` has no corpus-level rarity weighting (IDF)
        and `bm25s` does, so their keyword results measure different
        things.

### ZO-7. The swappable SearchEngine interface and the OpenSearch comparison are speculative (Major)

*   **Where:** L581–583, the last row of the tools table, Alternative N,
    and file 28.
*   **Problem:**
    *   The seven-method interface is shaped by an engine that is not part
        of v1.
    *   OpenSearch's fusion, filters and fuzzy matching do not map onto
        those methods one to one.
*   **Fix:**
    *   Write the Postgres search module as plain functions.
    *   Move the OpenSearch comparison to Future Improvements, keeping High
        Court data as its trigger.
    *   Extract an interface only when a second engine is actually
        written.

### ZO-8. Machinery for eras that v1 excludes (Major)

*   **Where:**
    *   the A–D era classifier (L373–381, file 15);
    *   file 17 (`docling_ocr.py`);
    *   the `ocr` extra;
    *   margin letters stored in the position map.
*   **Problem:**
    *   v1 handles only Eras C and D.
    *   No v1 output reads the margin letters: the Citation object shows
        only pages, and Era D has no margin letters.
*   **Fix:**
    *   The v1 classifier returns only C, D or "out of scope" (logged).
    *   Remove file 17 and the `ocr` extra.
    *   The position map stores only (char_offset, page).

### ZO-N1. `cites_case` links are stored but never used (Nit)

v1 should store only the links from Held points to paragraphs. Defer the
links to cited cases, their treatment fields, and the counts of unresolved
citations.

### ZO-N2. Table detection in the chunker (Nit)

*   **Problem:** detecting tables from aligned word columns is fiddly. The
    one table type the design names, the Case Law Reference table, sits in
    the references zone and is not chunked anyway.
*   **Fix:**
    *   Drop table handling and the `oversize_table` flag from v1.
    *   Add them back if the G6 hand check finds tables split across
        chunks.

### ZO-N3. Opinion-type rules (Nit; see IC-N3)

*   **Problem:** "the first opinion is the majority" is wrong whenever a
    dissent is printed first.
*   **Fix:**
    *   Assign a type only on a strong matching phrase, otherwise
        `unknown`.
    *   Never infer majority from position.

### ZO-N4. Promoting the statute notice and tracking its trigger rate (Nit)

v1 is heavy on criminal and procedural law, so the notice will fire often.
Make it a standing notice and delete the rate tracking.

### ZO-N5. Normalisation is specified twice, with steps in different orders (Nit)

*   **Where:** L629–634 and L1238–1243.
*   **Problem:**
    *   The two places put the steps in a different order.
    *   Parsing already rejoins hyphenated words, so the normaliser's
        hyphen step has nothing left to act on.
*   **Fix:**
    *   Keep one specification, in file 11, and have the Technical Plan
        point to it.
    *   Drop the hyphen step, or say what input it is for.

### ZO-N6. The rule for choosing between duplicate copies (Nit)

*   **Problem:** the duplicate PDFs are byte-identical, and the year folder
    matches the decision year in only 88% of rows, so choosing by folder
    adds nothing.
*   **Fix:** keep the first row for each `path`, in a fixed sort order.

### ZO-N7. Existing code is not reused (Nit)

`scripts/data_check.py` already contains a tested tar-offset scan
(`tar_analysis`, L65–87) and a reader (`read_member`, L97). File 13 should
port that code rather than write it again.

### ZO-N8. A multiprocessing pool from the start (Nit)

Start with a single process for the roughly 13,000 born-digital judgments.
Add a pool only if the measured run time calls for one.

### ZO-N9. Forward-looking items are scattered through the plan (Nit)

*   **Where:**
    *   the JSON Schema export "for any future Rust layer";
    *   the details of the Rust trigger;
    *   the lawyer-review export (file 49a).
*   **Fix:**
    *   Keep the schema export only if the web page uses it.
    *   Move file 49a and the Rust details to Future Improvements, as one
        line each.

### ZO-N10. At about 250,000 vectors, G4 shows little about approximate search (Nit)

*   **Problem:** exact search over about 0.75 GB of vectors probably fits
    within the time budget.
*   **Fix:** state that G4 at this size mainly tests crowding by
    near-duplicate passages. The approximate-search question becomes
    meaningful at 730,000 vectors or more.

---

## Lens: egm-inventive-catalyst (trade-offs accepted without testing)

### IC-1. Requiring an exact copied quote is not needed to check support (Major)

*   **Where:**
    *   answer generation and the citation check (L612–651, L1538–1578);
    *   Alternative D;
    *   G1.
*   **Problem:**
    *   The substring check proves only that the words exist, not that they
        support the claim.
    *   Asking the model to copy 5 to 40 words exactly makes the prose
        stiffer.
    *   It marks sentences unverified for copying slips alone: a dropped
        word, an ellipsis, a changed tense.
    *   Alternative D compared only against whole-passage IDs. Pointing at
        individual sentences was never considered.
*   **Fix:**
    *   Label the sentences in the prompt (P3.S4), reusing the character
        offsets in the `sentences` table.
    *   The model cites `{label, span?: [word_from, word_to]}`.
    *   Code takes the quote word for word from `chunk.text`, so a copying
        error cannot happen.
    *   The check confirms that the label exists, the span is in range, and
        the length is within limits.
    *   Run exact-quote mode as a variant for one evaluation run and compare
        rule-break rates.
*   **Lighter fallback:** on a failed match, line up the model's quote
    with the source using RapidFuzz and show the source span, never the
    model's text. Still count the miss as a rule break.
*   **Costs:**
    *   extra prompt tokens;
    *   word spans are needed for sentences over 40 words;
    *   the G1 and G8 "rule breaks before the check" metric must be
        redefined;
    *   with the fallback, a near-match that drops a "not" gets repaired to
        the true source text, so the G2 judge must catch the misstatement.

### IC-2. The router makes one exclusive guess, with no way back (Major; overlaps ZO-3 and ZO-4)

*   **Problem:** short keyword queries skip keyword and vector search
    entirely. That goes against the IL-PCSR finding that keyword search
    must stay strong (`research_notes.md` L17).
*   **Fix, if the quote route is kept:**
    *   Only the citation-pattern route is exclusive.
    *   Fuzzy sentence lookup becomes a third ranked list, fused with the
        others by reciprocal rank fusion.
    *   It dominates only on explicit quotation marks, or on a high trigram
        similarity.
    *   Say whether quote results go on through reranking, the LLM and the
        check.
*   **Costs:**
    *   one more query per request, over about 5 million sentence rows;
    *   it needs a time budget and a minimum fragment length;
    *   three-list fusion changes the fusion behaviour measured for G3.

### IC-3. G4 size steps against the version-retention rule (Major)

Merged into ZO-6. The partial-HNSW option described there is this
reviewer's fix.

### IC-4. Gold labels are tied to `clean_text_version` (Major)

*   **Problem:** every parser change forces a new file of test-answer spans,
    and the design never says how labels are remapped. Phase 1 will tune
    the margin bands and running-head rules many times. Either the frozen
    test set keeps breaking, or the parser freezes too early.
*   **Fix:**
    *   Store each gold label as (path, paragraph numbers, a short anchor
        quote), optionally with the PDF page and bounding box. The PDFs
        never change.
    *   At evaluation time, resolve each label to a character span in the
        current text: find the paragraph first, then fuzzy-match the
        anchor.
    *   Report labels that fail to resolve, and fail the run above a set
        count.
*   **Costs:**
    *   a resolver module to maintain;
    *   the paragraph number must take priority over the anchor, because a
        repeated quotation can match twice;
    *   a paragraph split by a parser change must be flagged, not silently
        widened.

### IC-5. The hosted LLM is a single point of failure (Major)

*   **Problem:** a rate limit, a timeout or invalid JSON returns `status:
    error`, and invalid JSON is not retried. The user gets nothing, even
    though the reranked passages are already in hand and are useful on
    their own.
*   **Fix:**
    *   On an LLM failure, return `status: passages_only` with the top-k
        reranked passages, labelled "no summary generated".
    *   Configure an ordered list of providers, and fail over on a 429 or
        5xx error.
    *   Allow one repair retry on a JSON parse error, sending the error
        text back to the model.
    *   Record the reason for the failure in the trace.
*   **Costs:**
    *   a fifth status for the page, the tests and the evaluation harness;
    *   G5 must count it as neither an answer nor a refusal;
    *   evaluation runs must pin one provider and stop on failover.

### IC-6. Headnote handling is framed as a two-way choice (Major)

*   **Problem:** even when scoring counts only judgment spans, questions
    written from Held points will match the Held text during retrieval and
    inflate recall.
*   **Fix:**
    *   Index Held points only as pointers.
    *   Replace each Held hit with the judgment chunks its "[Para N]" link
        names.
    *   Only judgment chunks reach the reranker and the LLM.
    *   Report G3 both with and without the Held-pointer list.
*   **Costs:**
    *   an extra retrieval list;
    *   Held points with no resolvable "[Para N]" become dead entries and
        must be counted;
    *   one evaluation run is done twice.

### IC-7. The Kaggle round trip sits inside the development loop (Major)

*   **Problem:** every parser fix forces a manual Kaggle round trip to
    re-embed. The 2025 slice, about 1,000 judgments times about 19 chunks,
    could plausibly be embedded on the local RTX 2060 in minutes to an
    hour.
*   **Fix:**
    *   Key each vector by a hash of (embedding model, revision, header +
        text), and re-embed only new hashes.
    *   Embed locally below a measured chunk count, and send larger batches
        to Kaggle.
    *   Measure local throughput in `scripts/select_models.py`.
    *   Apply the 0.999 cosine check to both local and Kaggle vectors.
*   **Costs:**
    *   a vector cache;
    *   a change to a judgment's metadata invalidates every hash in that
        judgment;
    *   local embedding runs only while the API is stopped.

### IC-N1. Users wait up to 15 seconds for passages that are ready at about 2.5 seconds (Nit)

*   **Fix:**
    *   The page calls a new `/search` endpoint and shows the reranked
        passages straight away.
    *   It then calls `/ask` with the same `request_id`, and the server
        reuses the cached candidates.
    *   Only the written answer waits for the citation check. No streaming
        is needed.
*   **Costs:**
    *   a second endpoint;
    *   a candidate cache;
    *   two calls linked in the trace;
    *   a label for passages the answer does not use.

### IC-N2. One fixed refusal threshold will drift as the index grows (Nit)

*   **Fix:**
    *   Run a fixed set of probe questions against each new
        `index_version`, as part of the load checks.
    *   Set or confirm the threshold before `activate`, and store it with
        the version.
    *   Report the probe metrics, so that an automatic threshold does not
        hide a real drop in quality.

### IC-N3. Use the headnote as a second source for opinion type (Nit)

*   **Fix:** parse "per X J." and the "dissenting" or "minority" markers in
    the editorial zone, as an independent hint. Record any disagreement
    with the phrase rules as a flag. This adds one flag value to the
    ChunkRecord contract.
