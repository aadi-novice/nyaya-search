# Stage 6: Comprehension Test

*   **Target:** `docs/designs/nyaya_search.md`. The design part is lines
    1–1050 and the implementation plan is lines 1051–1858.
*   **Reviewer:** `egm-comprehension`. It read the document alone, with no
    other files and no document standard.

**What this proposes**

Nyaya Search is a research assistant for one user, running on localhost. A
lawyer asks a question about Indian Supreme Court case law and gets one of
two things:

*   an answer in which every sentence cites a retrieved passage and quotes
    it exactly; or
*   "no relevant judgment found".

The quote is checked by code, not by the LLM. Only after the check is it
turned into a full citation: case, SCR citation, paragraph and page.

The build pipeline has five steps:

1.  It builds a list of judgments from the AWS dataset, with duplicates
    removed.
2.  It parses the 2010–2026 PDFs with PyMuPDF, using word coordinates to
    remove margin letters, running heads and page numbers by their position
    on the page.
3.  It splits each judgment into zones and opinions, then into chunks by
    numbered paragraph.
4.  It links each Held point to the paragraphs it names, and each cited
    case to its judgment.
5.  It embeds the chunks on Kaggle and loads everything into one versioned
    PostgreSQL and pgvector store.

At query time, a rule-based router picks one of three routes:

*   citation lookup;
*   fuzzy quote lookup;
*   hybrid search: keyword and vector search, merged by reciprocal rank
    fusion.

Results are reranked on the local GPU, then pass a refusal gate based on the
reranker score. A free hosted LLM writes the answer as JSON sentences, each
carrying passage labels and quotes. Sentences that fail the check go into a
hidden `unverified` list.

An evaluation harness measures goals G1 to G10. One of them is whether search
quality holds as the index grows from about 10,000 chunks to the full
2010–2026 slice.

**How the existing system works, per this document**

There is no existing product. What exists:

*   **The downloaded AWS dataset:** a tar file of PDFs for each year, plus
    Parquet metadata. It holds 38,397 distinct judgments, 5,181 duplicates,
    and 7 metadata rows with no PDF.
*   **Four research scripts** (download, data check, layout study, Docling
    test), which stay untouched.
*   **An existing Python virtual environment.**
*   **Earlier findings, summarised in the design:**
    *   Docling merged margin letters into the text.
    *   The PDF layouts fall into four eras (A to D), which can be told
        apart by their markers.

The live system is entirely new.

## Findings

### 1. Major: "Option B" is never defined (line 589)

*   **Problem:** the text says "Option B, with separate stores", but
    Alternative B (line 735) is "Build strictly layer by layer". The
    context, and lines 889 and 1720, show that Alternative M is meant
    (Qdrant + a BM25 library + SQLite).
*   **Fix:** write "Alternative M (Qdrant + a BM25 library + SQLite)".

### 2. Major: the index-version lifecycle does not fit together

*   **Where:** lines 486–487, 987–990, 1120, 1187–1189 and 1453–1462.
*   **Problem:**
    *   The schema's status values are building, active and retired, but
        `load()` sets the status `ready`, which is not one of them.
    *   The active version is set in two places: config, or
        `ACTIVE_INDEX_VERSION` in the environment, and also by
        `activate()` writing to the database. The document never says which
        one the API trusts.
    *   `activate()` retires the previous version, and line 1462 deletes
        older rows, yet rollback is described as "switching one config
        value". Can a retired version be re-activated, or are its rows
        already gone?
    *   The previous version is kept "until the new one passes its checks",
        but those checks are never named.
*   **Fix:**
    *   Name one source of truth for the active version.
    *   Add `ready` to the status values, or drop it.
    *   Name the checks run after loading.
    *   Say how long the previous version stays restorable.

### 3. Major: what each route does after retrieval is unclear

*   **Where:** lines 559–570, 263–279 and 1593–1610.
*   **Problem:**
    *   **Citation lookup** says "no LLM call", but the pipeline then lists
        reranking, the refusal gate, the LLM and the check for every route.
    *   **Quote lookup:** CUJ 3 says its hits are "ranked first and returned
        with their citations". It does not say whether they go through
        reranking, the refusal gate and the LLM.
*   **Fix:** add a table showing, for each route, which stages run and
    which status and answer shape it returns.

### 4. Major: "factual sentence" is never defined

*   **Where:** line 615, against lines 644 and 1574.
*   **Problem:**
    *   The prompt requires citations only on *factual* sentences, so the
        model may legitimately write uncited sentences, such as transitions
        or "the passages do not address X".
    *   But the check moves every sentence with no passing citation into
        `unverified` with the reason `no_citation`, and counts it as a rule
        break.
    *   This blurs what G1's rule-break rate measures, and can hide an
        honest "the passages don't answer this" sentence.
*   **Fix:** do one of these:
    *   require a citation on every sentence;
    *   define how a non-factual sentence is recognised and treated;
    *   leave `no_citation` out of the rule-break count when the sentence
        is non-factual.

### 5. Nit: three places disagree on what an unverified sentence shows

*   **Where:** line 645 and lines 102–103, against line 1652.
*   **Problem:** the Technical Plan says it keeps "its label and quote". G1
    says only that citation details are left out. The web page spec says
    it shows "only the label and the reason".
*   **Fix:** pick one rule and use it in all three places.

### 6. Nit: the eras are defined late and incompletely

*   **Where:** lines 373–378 and 1025.
*   **Problem:** year ranges appear only for Eras A and B, at line 1025.
    The Era D labels are not listed at line 373.
*   **Fix:** add an approximate year range for each era, and list the Era D
    labels where the eras are first defined.

### 7. Nit: the Overview says parsing runs on Kaggle (line 30)

*   **Problem:** the plan parses 2010–2026 on the PC, and uses Kaggle only
    for embedding and for OCR of the old eras.
*   **Fix:** reword the Overview to "embedding, and OCR for older eras".

### 8. Nit: "Phase 4: Era A" is never defined (line 1321)

*   **Problem:** the build order defines only Phases 0 to 3.
*   **Fix:** add a Phase 4 to the build order, or relabel this item.

### 9. Nit: legal acronyms are never expanded

*   **Which:** CNR, SCC, AIR, INSC, BNS, BNSS, BSA, and partly IPC and
    CrPC.
*   **Fix:** expand each on first use, or add a short glossary.

### 10. Nit: papers and methods are named without a reference

*   **Which:** CRAwLeR, Lightfoot, the Adaptive Chunking paper, the German
    Civil Code study, LegalSeg, *Decisions by Discrimination* and DcLBC.
*   **Fix:** add a reference or a one-line description for each, or point
    each one to `docs/research_notes.md`.

### 11. Nit: the document leans on "the brief" for content the reader needs

*   **Where:** lines 126, 233, 964 and 1036.
*   **Problem:** line 126 mentions the brief's claim without saying what it
    was. Line 233 says the decision criteria "are in the brief", although
    the tools table already gives them.
*   **Fix:** state the claim at line 126, and point line 233 to the tools
    table instead.

### 12. Nit: "Goldfish review findings" is process jargon (line 1004)

*   **Fix:** write "external review findings", or define the term.

### 13. Nit: how `partial` is scored is undefined

*   **Where:** lines 1744–1748, against line 109.
*   **Problem:** the judge returns supported, partial or unsupported, but
    G2 does not say whether `partial` counts as supported.
*   **Fix:** state how `partial` is scored.

### 14. Nit: formatting and order

*   **Indenting:** at lines 126–129, "The measurement:" is attached to the
    deferral bullet, so its three rules read as part of that note. Move it
    out as its own bullet.
*   **Order:** Alternative R comes before Alternative Q. Put them in order.

## Overall

The document mostly stands on its own. Terms specific to this design (zones,
Held points, the position map, passage labels, the refusal gate,
`index_version`, the citation check) are defined before or where they are
used. The Major findings are structural gaps, not missing vocabulary:

*   the undefined "Option B";
*   the index-version lifecycle;
*   what each route does after retrieval;
*   the uncited-sentence rule.
