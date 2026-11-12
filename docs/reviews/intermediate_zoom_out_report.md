# Stage 8: Intermediate Zoom Out

*   **Target:** `docs/designs/nyaya_search.md`, the full file (lines
    1–1858).
*   **Reviewer:** `egm-zoom-out`. It completed, after reading the Stage 5, 6
    and 7 reports plus the four background files.
*   **Note:** the document has not been edited since those reports were
    written.
*   **Findings:** 0 Blockers, 6 Majors, 2 Nits. Every finding is about how
    the earlier fixes interact with each other. None repeats an earlier
    finding.

## Major findings

### 1. The quote route is the first decision, and about seven fixes depend on it

*   **Where:**
    *   **Earlier findings:** ZO-3, IC-2, ZO-4, Comprehension #3, and the
        Critic's "short-query rule" and "lookup routes".
    *   **Design:** L270–279 and L559–567.
    *   **Plan:** files 20, 28, 29 and 31.
*   **Problem:** ZO-3 removes the quote route, IC-2 fuses it in as a third
    ranked list, and the Critic proposes a middle option. If ZO-3 is
    applied, these become moot:
    *   IC-2, including its time budget for about 5 million sentence rows;
    *   the Critic's short-query fix, and its labelled set of routing
        questions;
    *   the quote-route parts of ZO-4, Comprehension #3 and the Critic's
        lookup finding;
    *   the `sentences` table and its trigram index.

    IC-1 does not need the `sentences` table either: sentences can be split
    and labelled when the prompt is built.
*   **Fix:** adopt ZO-3. Pass quoted phrases into the keyword query as
    phrases, and close the dependent findings.

### 2. Status values grow one fix at a time, and two fixes disagree on what happens when the LLM fails

*   **Where:**
    *   **Earlier findings:** IC-5, the Critic's "LLM quota and failure",
        ZO-4 and the Critic's lookup routes (`status: lookup`), and the
        Critic's `PassageRef`.
    *   **Design:** L537–552.
    *   **Plan:** file 34, and the edge cases in file 39.
*   **Problem:**
    *   Together, the fixes take `status` to six values.
    *   On an LLM failure, IC-5 returns the passages. The Critic returns
        `error`, or `refused` after a repair attempt.
    *   IC-5's provider failover plus a repair retry cannot fit inside the
        15-second G7 target.
*   **Fix:**
    *   Add a single new status, `passages`, carrying a list of
        `PassageRef`. Use it both for citation lookup and for LLM failure.
    *   Give `nearest_passages` the same type.
    *   Allow one repair retry, only if there is time left before the
        15-second target.
    *   Drop provider failover from v1.

### 3. Three versioning fixes conflict, and the lifecycle fix waits on the choice

*   **Where:**
    *   **Earlier findings:** ZO-6 (one schema per step), IC-3 (a partial
        HNSW index per step) and the Critic's "filtered vector search" (one
        partition per `index_version`). Comprehension #2 waits on this.
    *   **Design:** L480–487 and L987–990.
    *   **Plan:** files 9, 27 and 46.
*   **Problem:** all three treat the G4 size steps as if they were
    production versions. That is what causes the clash with the
    "two-version" rule.
*   **Fix:**
    *   **G4 steps:** build each one in a temporary schema, and drop it
        after the run. That makes ZO-6(a), ZO-6(c) and the exemption for
        evaluation steps moot.
    *   **Production:** keep only the active version and one previous
        version.
    *   **Active version:** set it only through `activate` in the database,
        and remove `ACTIVE_INDEX_VERSION` from the environment.
    *   **Search mode:** first time an exact vector search over about
        250,000 vectors (from ZO-N10).
        *   If it fits the 0.5-second retrieval budget, production uses
            exact search with SQL filters.
        *   That removes the filtered-search risk for CUJ 4.
        *   HNSW is then needed only inside G4 runs.

### 4. The development-set and headnote fixes partly overlap and partly conflict

*   **Where:**
    *   **Earlier findings:** ZO-1, IC-4, IC-6, and the Critic's "headnote
        handling" and "sub-paragraphs".
    *   **Design:** L217–239 and L595–599.
    *   **Plan:** file 43.
*   **Problem:**
    *   Once ZO-1 builds the development set from Held points, scoring
        editorial hits differently (the Critic's fix) does not stop the
        recall inflation that IC-6 warns about. Those inflated numbers would
        then steer the choice of embedder.
    *   Labels built from Held points are already (path, paragraph label),
        which is IC-4's format without the anchor quote. So IC-4's resolver
        is not needed in v1.
    *   Storing paragraph labels as strings, the Critic's fix, has to come
        first for both.
*   **Fix:**
    *   Gold labels are (path, paragraph label). Drop the
        `clean_text_version` spans and the anchor resolver.
    *   After expansion, replace each Held hit with the judgment chunks it
        links to. This is a small change to `expand.py`, file 32.
    *   Drop IC-6's suggestion to report G3 twice.

### 5. Together, the fixes undo ZO-2's scope cut

*   **Where:** ZO-2, against these:
    *   IC-5, IC-7, IC-N1, IC-N2, IC-N3 and IC-4;
    *   the Critic's Act normalisation, routing set, per-run token budgets
        and partitions.
*   **Problem:** together they add about ten components to the Phase 1
    that ZO-2 is trying to cut down to 11 steps.
*   **Fix:**
    *   **Defer:**
        *   IC-N1, IC-N2 and IC-N3;
        *   the IC-5 failover;
        *   the IC-4 resolver;
        *   the IC-7 hash cache.
    *   **Cut the Act filter from v1** (the Critic's Era C finding).
        *   With ZO-N1 also applied, nothing in v1 reads the `refs` table.
        *   Remove that table and its GIN index too.
    *   **Keep local embedding (IC-7), without the cache,** because it
        removes work.
        *   Files 25 and 26 and the 0.999 cosine check across machines (file
            33) then leave Phase 1.
        *   Measure local throughput first.

### 6. The Risks table (a Critic Blocker) would list risks for features being cut

*   **Problem:** several of its proposed rows disappear under other fixes:
    *   misclassifying an era goes away under ZO-8;
    *   a dissent tagged as the majority goes away under ZO-N3;
    *   false matches on party names go away under ZO-5;
    *   the filtered-search risk goes away under finding 3.
*   **Fix:** write the Risks table last, listing only the risks that remain
    after the other fixes.

## Nits

### 7. Pairs where one fix makes the other unnecessary

*   **ZO-N3 and IC-N3:** ZO-N3 removes the need for IC-N3's disagreement
    flag, so drop IC-N3.
*   **ZO-N6 and the Critic's tie-break:** dropping the year-folder
    preference removes the need for the tie-break.
*   **ZO-8:**
    *   It makes the Critic's letter-assignment rule moot, and IC-4's
        bounding box.
    *   The margin-band measurement is still needed for stripping.
*   **The Critic's token finding:** the "8,000-token model" option is a
    dead end, because cross-encoder rerankers are mostly limited to 512
    tokens. Take the word cap instead. That raises the chunk count, which
    changes the figures for ZO-N10 and G4.

### 8. IC-1's lighter fallback weakens G1 (L92–108)

*   **Problem:** a citation repaired by RapidFuzz would be shown as
    verified. That bends G1's word-for-word guarantee, and can hide a
    dropped "not".
*   **Fix:** do not adopt it. For v1, keep exact quotes with the Critic's
    minimum length. Test IC-1's label-and-span mode later, as one
    evaluation variant.

## Recommended minimal set of fixes

1.  ZO-1, done as in finding 4.
2.  ZO-2, ZO-3 and ZO-5.
3.  Finding 2: one `passages` status and the `PassageRef` type.
4.  Finding 3: G4 steps kept out of production versions, and exact search
    if it is fast enough.
5.  ZO-8, ZO-N1, ZO-N3 and ZO-N6.
6.  From the Critic:
    *   the token cap;
    *   paragraph labels stored as strings;
    *   the step that measures the margin bands;
    *   closing the Open Questions;
    *   naming "Option B" as Alternative M;
    *   a measurable BM25 trigger;
    *   a Risks table, written last.
7.  Comprehension #4, and the remaining Comprehension Nits as edits.
8.  All the style and wording items.

Defer everything else listed in finding 5.
