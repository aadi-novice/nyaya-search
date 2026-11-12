# Stage 7: Critic Review

*   **Target:** `docs/designs/nyaya_search.md`, the design part only (lines
    1–1050).
*   **Reviewers:** `egm-design` and `egm-structural`, both checked against
    the feature-spec standard, and `egm-style`, which used its own checks.
    All three completed.
*   **Background reading:**
    *   `docs/data_check_report.md`
    *   `docs/document_anatomy.md`
    *   `docs/docling_findings.md`
    *   `docs/research_notes.md`
*   **Verdict:** not ready to freeze. There are 2 Blockers, 22 Majors and 8
    Nits after duplicates were removed. A finding raised by more than one
    reviewer is listed once and names the others.

## Design lens (egm-design)

### Blocker: there is no Risks & Mitigations section

Structural also raised this. Line 783 says risks "are recorded", but there is
no such record, and the standard requires one.

*   **Fix:** add a `## Risks & Mitigations` table with the columns Risk,
    Failure mode, Mitigation and Owning goal or test. Cover at least:
    *   free-tier LLM rate limits, outages and model retirement;
    *   the provider keeping the data sent to it;
    *   filtered vector searches returning too few rows;
    *   the embedder cutting off the end of long chunks;
    *   misclassifying a judgment's era;
    *   a dissent tagged as the majority opinion;
    *   false matches from the citation patterns in check step 5;
    *   Postgres full-text search being weaker than BM25;
    *   the refusal threshold drifting as the index grows;
    *   memory pressure under WSL2;
    *   the judge model disagreeing with the hand checks.

### Major: filtered vector search can quietly return too few rows

*   **Where:** L350–351, L484, L574–577, L592 and CUJ 4.
*   **Problem:**
    *   pgvector's HNSW index scans first and applies the `WHERE` filter
        afterwards.
    *   With `ef_search = 40` and a narrow filter, such as a year range plus
        an Act, a request for the top 50 can return far fewer rows.
    *   `index_version` is itself a filter, so every query is filtered.
    *   This lowers G4 scores for reasons that have nothing to do with
        corpus size.
*   **Fix:**
    *   one table or partition per `index_version`;
    *   pgvector 0.8 or later, with `hnsw.iterative_scan = relaxed_order`
        and `ef_search` of at least 100;
    *   log the rows returned against k in the trace;
    *   include filtered queries in G4.

### Major: chunk sizes are given in words, but the models limit tokens

Style and structural also raised this.

*   **Where:** L433–436, L440 and L714–715.
*   **Problem:**
    *   The config says token counts, but the starting values are in words.
    *   A 700-word maximum is about 900–1,000 tokens.
    *   Most embedders and cross-encoders stop at about 512 tokens, so the
        end of a long chunk is never embedded or reranked.
    *   The anatomy document splits at about 400 words, not 700.
*   **Fix:**
    *   State the limits in the chosen model's tokens.
    *   Either cap chunk plus header at that limit (about 350 words for 512
        tokens), or require models with an 8,000-token window.
    *   Add the G6 metric "share of chunks cut off = 0".
    *   Explain the departure from the anatomy document.

### Major: the lookup routes fit neither G1 nor the API contract

Structural also raised this.

*   **Where:** L262–279, L537–552, L569 and L92–96.
*   **Problem:**
    *   CUJ 2 returns a headnote with no LLM answer, and CUJ 3 returns
        ranked sentences.
    *   Neither is a checked answer or a refusal, and `status` has no value
        for them.
    *   The diagram sends every route through reranking, the refusal gate
        and the LLM.
*   **Fix:**
    *   Add `status: lookup` and a field for the lookup result.
    *   Give the response shape for each route.
    *   Say whether quote lookup calls the LLM.
    *   Reword G1 to cover the routes that do not use the LLM.

### Major: the quote check proves the words exist, not that they support the claim

*   **Where:** L615–616, L635 and L766–768.
*   **Problem:** a 5-word quote such as "the court is of the view" matches
    almost any chunk.
*   **Fix:**
    *   Require at least 8 words, or reject quotes made up mostly of common
        words.
    *   Reword Alternative D and G1 to say "the quote exists in the
        passage". Support is measured only by G2.
    *   Consider refusing any answer whose share of verified sentences falls
        below a set minimum.

### Major: the Act filter has no data for Era C

*   **Where:** L549, L555 and CUJ 4.
*   **Problem:** "List of Acts" exists only in Era D, so CUJ 4's filter
    would quietly drop judgments from 2018–2023.
*   **Fix:**
    *   Take the Act from the Era C catchwords (the text before ":").
    *   Normalise all Act names into one list.
    *   Add a check to CUJ 4 that Era C judgments are returned.

### Major: the short-query rule sends keyword searches to quote lookup

*   **Where:** L564–566.
*   **Problem:** a query such as "anticipatory bail section 438" would skip
    hybrid search.
*   **Fix:**
    *   Send only quoted strings to quote lookup.
    *   For short unquoted queries, run both and fuse the results.
    *   Measure routing accuracy on a labelled set of short queries.

### Major: paragraph numbers cannot hold sub-paragraphs

*   **Where:** L507–508 and L529.
*   **Problem:** "12.1" does not fit in `para_start` and `para_end`, which
    are typed `int`.
*   **Fix:**
    *   Store paragraph labels as strings, or as a list.
    *   Update the Citation `paras` field to match.
    *   Define how "[Para N]" resolves when N is a sub-paragraph.

### Major: the duplicate rule has no tie-break, and the slice is not defined

Structural also raised this.

*   **Where:** L363–366, L25 and L380–381.
*   **Problem:** only 88.1% of rows match their year folder, so in some
    duplicate pairs neither copy matches, or both do.
*   **Fix:**
    *   Break ties by the lexically lowest tar name.
    *   State that the 2010–2026 slice is chosen by the parsed
        `decision_date`.
    *   Say whether files inside that range that classify as another era
        are excluded, and how they count toward the 13,007 figure.

### Major: margin-letter stripping rests on an unmeasured assumption

Style also raised this.

*   **Where:** L387–401.
*   **Problem:**
    *   No coordinate band has been measured yet.
    *   Assigning a line letter to each character needs the letters'
        positions recorded before they are deleted, and that step is
        missing.
*   **Fix:**
    *   Add an early step that measures the margin bands on the 50-PDF
        sample, and that can stop the build.
    *   Define how letters are assigned: the nearest letter at or above
        the word's vertical position, on the same page.
    *   If no band is found for a file, log it and strip nothing.

### Major: the same 50-PDF sample is used both to tune and to judge

Structural also raised this.

*   **Where:** L145–146 and L396–397.
*   **Fix:** after the bands are fixed, draw a separate held-out G6 sample,
    and record its seed.

### Major: LLM quota and failure handling are not defined

*   **Where:** L622–624, L696–697, L716 and L980.
*   **Problem:**
    *   Each question uses about 5,000–8,000 input tokens, and an evaluation
        run makes hundreds of calls.
    *   There is no defined behaviour for a 429, a timeout or invalid JSON.
    *   The 10-second LLM budget leaves no room for a retry inside the
        15-second target.
*   **Fix:**
    *   Set token budgets per request and per run.
    *   Add limits on retries and backoff, and map running out of retries
        to `status = error`.
    *   Allow one repair attempt on invalid JSON, then refuse.
    *   Say how failed requests count in G7.

### Major: headnote handling is both decided and left open

*   **Where:** L237–239, against L412–413, L267–268 and L533.
*   **Problem:** G3 does not say whether a hit on an editorial chunk counts
    as finding the judgment. Counting it would leak answers.
*   **Fix:**
    *   Keep editorial chunks searchable.
    *   Score G3 and G4 only on judgment-zone spans.
    *   Close the open question.

### Nit: RAM on Windows is not broken down

*   **Where:** L706 and L982–983.
*   **Fix:** give the RAM each process uses, the memory limit in the WSL2
    `.wslconfig`, and the `maintenance_work_mem` used to build the HNSW
    index.

### Nit: the API's bind address is not stated in the design part

*   **Where:** about L975.
*   **Fix:** traces hold users' questions, so state that the API binds to
    127.0.0.1.

## Structural lens (egm-structural)

### Blocker: there is no Risks section

Listed under the design lens above.

### Major: Open Questions contradict decisions made elsewhere

All three reviewers raised this.

*   **The vector store** is "not chosen" (L232), but pgvector is committed
    (L318, L342–345, L887).
*   **The snapshot date** is open (L236), but it is pinned to 2026-11-11
    (L994).
*   **The 7 rows with no PDF** are open (L234), but `has_pdf = false` is
    already decided (L382–383).
*   **Fix:** close or delete each of these questions.

### Major: the G4 list is malformed

Style also raised this.

*   **Where:** L125–133.
*   **Problem:** "The measurement:" breaks in the middle of a sentence, and
    its bullets attach to the wrong parent.
*   **Fix:** put "The measurement:" on its own line, and indent its three
    bullets under it.

### Major: a cross-reference points to the wrong alternative

*   **Where:** L589.
*   **Problem:** "Option B, with separate stores" means Alternative M.
*   **Fix:** write "Alternative M (Qdrant + a BM25 library + SQLite)".

### Major: rejections are credited to "the author" with no linked decision

Design and style also raised this.

*   **Where:** L732–954.
*   **Problem:** the standard forbids crediting a rejection to a person
    without a record of the decision.
*   **Fix:** lead each rejection with its technical trade-off, or link each
    one to a dated entry in `concepts_brief.md`.

### Major: alternatives are out of order, and the decision field is worded inconsistently

All three reviewers raised this.

*   **Problem:**
    *   Alternative R comes before Alternative Q.
    *   The decision field reads "Not rejected", "Kept only as…", "Not
        adopted" or "Deferred", depending on the entry.
*   **Fix:**
    *   Put Q before R.
    *   Name the field "Decision", with fixed values: Rejected, Deferred,
        Fallback or Planned comparison.

### Major: two headings sit at the wrong level

*   **Where:** L1001 and L1014.
*   **Problem:** `### Additional Resources` and `### Future Improvements`
    read as subsections of the Non-Functional Requirements.
*   **Fix:** promote both to `##`.

### Major: `nearest_passages` is typed `[Citation]`

*   **Where:** L546.
*   **Problem:** the type forces `quote` and `quote_verified` onto passages
    shown with a refusal.
*   **Fix:** define a `PassageRef` type without the quote fields, or make
    both fields nullable.

### Nit: sections are out of the standard's order

*   **Where:** L215 and L969.
*   **Problem:** Open Questions comes before the CUJs, and the Non-Functional
    Requirements come after Alternatives.
*   **Fix:** follow the standard's order, with Open Questions last.

### Nit: some references cannot be followed

*   **Where:** L126 and L1003–1012.
*   **Problem:**
    *   "The brief's 'a million or more' claim" is never introduced.
    *   File paths are plain code spans, relative to the repository root.
*   **Fix:** quote the claim once, and use relative Markdown links.

## Style lens (egm-style)

### Major: untested parsing is described as fact

*   **Where:** L333–334, L387–397 and L799–801.
*   **Problem:** it matches the design finding on margin letters.
*   **Fix:** write "is expected to remove… not yet tested; G6's hand check
    is the acceptance test".

### Major: the duplicate claim is overstated

*   **Where:** L364–366.
*   **Problem:** the evidence is 300 sampled pairs, and one duplicate has its
    PDF in only one tar.
*   **Fix:** say exactly that.

### Major: some research claims are not supported by `research_notes.md`

*   **L847–848, "worst size compliance":** cite the table, or delete the
    clause.
*   **L834, the German Civil Code study:** reword to "section-based chunks
    beat LLM-guided, semantic and fixed-size chunking on German statutes".
*   **L858–859, CRAwLeR:** check against the paper, or delete. Replace
    "measurably helps" with the specific result and its source.

### Major: the margin-letter claim is wrong for Era C

*   **Where:** L62.
*   **Problem:** in Era C, PyMuPDF keeps margin letters on lines of their
    own. It is Docling that merges them into sentences.
*   **Fix:** write "appear in the text layer; some extractors merge them
    into sentences".

### Major: "about" in targets and code rules makes them untestable

*   **Where:** L135–136, L564, L601–603 and L661–662.
*   **Fix:** give exact starting values (≤10%, <8 words, ≤100 candidates,
    >50%), and say where each one is tuned.

### Major: "if BM25 clearly wins" is not a measurable trigger

*   **Where:** L584–591 and L718.
*   **Fix:** write "if BM25 recall@50 on the development set beats Postgres
    full-text search by at least X points", with X fixed before the run.

### Nit: size estimates hide their assumptions

*   **Where:** L125, L190–191, L706 and L985–986.
*   **Fix:** state the basis of each number. Examples: "N judgments × 19
    chunks, from the data check" and "at 768 dimensions, float32".

### Nit: value words carry no information

*   **Where:**
    *   L92, L134, L651 and L1040 ("honest", "useful, not a shortcut");
    *   L942–946 (the "resume" pro, "must never be wrong");
    *   L37–40, L580 and L875 (overstatements);
    *   L1003 ("Goldfish" jargon).
*   **Fix:** use neutral wording. Examples:
    *   "G5: Refusal rate";
    *   "k = 60 is fixed";
    *   "aim at the same benefit";
    *   "review findings".
