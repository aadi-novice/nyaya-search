# Consolidated Review Report: Nyaya Search Design

*   **Target:** `docs/designs/nyaya_search.md`. The design part is lines
    1–1050 and the implementation plan is lines 1051–1858.
*   **Mode:** a consolidated report, run through all five stages without
    stopping.
*   **Background reading given to the reviewers:**
    *   `docs/data_check_report.md`;
    *   `docs/document_anatomy.md`;
    *   `docs/docling_findings.md`;
    *   `docs/research_notes.md`.
*   **Date:** 2026-11-12.

## Stage reports

| Stage | Report | Reviewers | Blockers | Majors | Nits |
| --- | --- | --- | --- | --- | --- |
| 5 Initial zoom out | `zoom_out_report.md` | zoom-out, inventive catalyst | 1 | 13 | 13 |
| 6 Comprehension | `comprehension_report.md` | comprehension | 0 | 4 | 10 |
| 7 Critic | `critic_report.md` | design, structural, style | 2 | 22 | 8 |
| 8 Intermediate zoom out | `intermediate_zoom_out_report.md` | zoom-out | 0 | 6 | 2 |
| 9 Implementation readiness | `implementation_readiness_report.md` | implementation readiness, codebase details, style | 5 | ~24 | ~25 |

## Verdict

**The design is not ready to implement.**

*   **The architecture holds up.** Every reviewer accepted the core
    choices:
    *   one PostgreSQL store;
    *   PyMuPDF parsing with page coordinates;
    *   chunking that follows the document's structure;
    *   a rule-based router with no route that skips retrieval;
    *   a citation check done by code;
    *   no agents.
*   **What blocks implementation:**
    1.  **Scope:** v1 is larger than one person can build quickly.
    2.  **Ordering:** some steps need things that only exist later, such as
        the development set and the chosen models.
    3.  **Missing pieces:** the contracts, the routes and the build
        outputs are not yet specified.

Stage 8 showed that applying every proposed fix would undo the scope cut, so
the actions below follow its **minimal fix set**. Several findings become
moot once a decision is made, and these are marked.

## Decisions the author must make first

These have more than one defensible answer, and other fixes wait on them.

*   **D1. Quote-lookup route.**
    *   **Option A:** remove it from v1, and send quoted phrases into the
        keyword search as phrases.
    *   **Option B:** keep it as a third ranked list, fused with the other
        two.
    *   **What it settles:** about seven findings, plus the `sentences`
        table and its trigram index.
*   **D2. Index versions and the G4 size steps.**
    *   **Option A:** build each size step in a temporary schema and drop it
        after the run (Stage 8's recommendation).
    *   **Option B:** load the full slice once and build a partial HNSW
        index for each step.
    *   **Option C:** one schema for each step.
    *   **Also decide:** whether production uses exact vector search, once
        a timing at about 250,000 vectors shows it fits the time budget.
*   **D3. Behaviour when the LLM fails.**
    *   **Option A:** return the reranked passages under one new status,
        `passages`, which citation lookup also uses (Stage 8).
    *   **Option B:** return `error`.
*   **D4. Development set.**
    *   **Proposal:** build it in Phase 1 from the links between Era C/D
        Held points and judgment paragraphs. Each Held point is paraphrased
        into a question, and the paragraph it links to is the gold answer.
    *   **The trade-off:** this reverses the earlier "test set deferred",
        but every measured choice in Phase 1 depends on it.
*   **D5. Provisional models.** Name an embedding model, a reranker, a
    vector dimension and a refusal threshold to use until the development
    set exists.

## Actions by priority

### 🔴 Blockers

| ID | Action | Sources |
| --- | --- | --- |
| B1 | Make the development set a Phase 1 deliverable. Gold labels are (path, paragraph label). Held hits are swapped for the judgment chunks they link to before scoring. | ZO-1, IC-4, IC-6, S8-#4, Critic "headnote" |
| B2 | Complete the shared contracts: `ParsedDoc`, the position map, `Era`, `Zones`, `Opinion`, the sentence record, the search return types, the `refs` columns, and the `Settings` keys with their defaults. | S9-B1 |
| B3 | Name the provisional models, dimension and threshold. Choose how the migration gets the vector dimension. Make Phase 1's pass criteria independent of the threshold. | S9-B2, D5 |
| B4 | Name every build output and the order of loading: judgments, aliases, links, clean text with position map, and the Kaggle export. | S9-B3 |
| B5 | Add a table showing, for each route, which steps run and what it returns. Add `status: passages` and a `PassageRef` type. Add fallbacks for Era C and unresolved aliases. | ZO-4, Comp #3, Critic "lookup routes", S8-#2, S9-B4, D3 |
| B6 | Fix the index-version lifecycle: building → ready → active → retired; the active version recorded only in the database; G4 steps outside production. | Comp #2, ZO-6, IC-3, Critic "filtered search", S8-#3, S9-B5, S9-M1, D2 |
| B7 | Add a Risks & Mitigations section, written last, listing only the risks that remain after the other fixes. | Critic Blocker, S8-#6 |

### 🟡 Major: scope cuts (Stage 8's minimal set)

| ID | Action | Sources |
| --- | --- | --- |
| M1 | Define Phase 1 as 11 steps only: manifest, parse (C/D), zone, chunk, embed, load, hybrid search, rerank, LLM, citation check, page. Label everything else Phase 2+. | ZO-2, S8-#5 |
| M2 | Decide the quote route (D1). If it is removed, drop the `sentences` table and its trigram index, and close the dependent findings. | ZO-3, IC-2, Critic "short-query", S8-#1 |
| M3 | Check free-text citations only against reporter patterns (SCR, INSC, SCC, AIR) and full "X v. Y" names. Drop party-name aliases. | ZO-5 |
| M4 | The era classifier returns only C, D or out of scope. Remove `docling_ocr.py` and the `ocr` extra from v1. The position map stores (char_offset, page) only. | ZO-8, S8-#7 |
| M5 | Cut from v1: case-citation links, the `refs` table and the Act filter, table detection, and notice-rate tracking (the statute notice becomes a standing notice). | ZO-N1, ZO-N2, ZO-N4, Critic "Act filter", S8-#5 |
| M6 | Defer: provider failover, the IC-1 sentence-span mode (later, as an evaluation variant), the `/search` endpoint, the probe-set threshold check, the opinion-type hint from headnotes, and the vector hash cache. | IC-1, IC-5, IC-7, IC-N1–N3, S8-#5, S8-#8 |
| M7 | Write the Postgres search as plain functions. Move the OpenSearch comparison to Future Improvements. | ZO-7 |

### 🟡 Major: correctness and specification

| ID | Action | Sources |
| --- | --- | --- |
| M8 | Express chunk limits in model tokens, capping chunk plus header at about 350 words for 512-token models. Add a G6 metric: share of chunks cut off = 0. | Critic "chunk size", S9-M3, S8-#7 |
| M9 | Store paragraph labels as strings, to allow "12.1". Define how "[Para N]" resolves. | Critic "sub-paragraphs", S9-M5 |
| M10 | Set a minimum quote length of at least 8 words, or reject quotes made mostly of common words. G1 says "the quote exists in the passage"; support is measured by G2. | Critic "quote proves existence" |
| M11 | Define the uncited-sentence rule: a citation on every sentence, or a defined non-factual class excluded from rule breaks. | Comp #4 |
| M12 | Handle LLM failures: a token budget per request and per run; one JSON repair retry only if time remains before 15 s; rate-limit and timeout errors mapped to the D3 behaviour. | Critic "LLM quota", S8-#2 |
| M13 | Measure the margin bands in an early step that can stop the build. Judge G6 on a separate held-out sample with a recorded seed. Describe the parsing as untested. | Critic "margin letters", "same 50-PDF sample", Style "untested parsing" |
| M14 | Add a mapping from metadata columns to schema columns (`judge`, `citation`, `nc_display`, `disposal_nature`, `decision_date` as a DATE, plus `decision_year`), and a lookup from `disposal_nature` to outcome. | S9-CD-M1, S9-CD-M6 |
| M15 | Removing duplicates: keep the first row per `path` in a fixed order (has a PDF, then the later scrape). `<year>` means the decision year. The manifest covers all years; recount the rows without a PDF after removing duplicates. | ZO-N6, Critic "duplicates", S9-CD-M2, S9-CD-M4 |
| M16 | Join tar offsets on `<path>_EN.pdf` and store `offset_data`. Port the code from `scripts/data_check.py` lines 65–100. | ZO-N7, S9-CD-M3 |
| M17 | Move era classification into `parse-chunk`, starting from `MARKERS` in `layout_study.py`. | S9-CD-M5 |
| M18 | Search details: an OR query ranked with `ts_rank_cd`; `hnsw.iterative_scan` and `ef_search` if HNSW is used; added chunks ranked at their parent's rank + 0.5 with the cap applied after expansion; rerank on header + text; JSON output `{"sentences": [...]}`. | S9-M2, S9-M9 |
| M19 | Embedding: fix the header + text format and the query/passage prefixes; the Kaggle job writes reference vectors for the startup check; PyTorch pinned to a CUDA wheel index. | S9-M3, S9-M4 |
| M20 | Chunker and zoner: zone start and end patterns per era; the Era D author marker "*"; phrase rules limited to the start of each opinion; a named sentence splitter with abbreviations; `majority` never inferred from position. | S9-M5, S9-M6, ZO-N3 |
| M21 | Build failures go to `build/skipped.csv` and `build/errors.jsonl`, with a count threshold that fails the build. Wrap each API request in a 30-second timeout. | S9-M10, S9-CD-M11 |
| M22 | Evaluation definitions: an `exclude` field; `scored_zones`; the G4 pass rule (a confidence-interval bound); "wrong answer"; how `partial` counts; the hand-label format and agreement measure; `tune-threshold`; `latency`. | S9-M11, Comp #13 |
| M23 | Tests and verification: name each test file and its cases; 10 written Phase 1 questions with expected route and judgment; a checklist per CUJ; the margin-letter test uses the in-sentence pattern, not "any lone letter". | S9-M12, Style "A–H test" |
| M24 | Dependencies and packaging: an `eval` extra, `pytest-cov` and `numpy`, drop `tomli-w`, entry points named, `__init__.py` files and `db/init/00_extensions.sql` listed, and a note that `uv sync` removes Docling. | S9-CD-M7–M10 |
| M25 | Make the BM25 trigger measurable: recall@50 gap ≥ X points on the development set, with X fixed before the run. | Style "clearly wins" |
| M26 | Make `config/settings.toml` truly complete: every threshold, seed, timeout and path, with its type and starting value. | Style M1 |

### 🟢 Nits: edits that need no decision

| ID | Action | Sources |
| --- | --- | --- |
| N1 | Change "Option B" to "Alternative M". | Comp #1, Critic structural |
| N2 | Close the Open Questions already decided (vector store, snapshot date, rows without a PDF, headnotes). | Critic structural |
| N3 | Fix the indentation of the G4 list. Put Alternative Q before R. Use a "Decision" field with fixed values. Lead each rejection with its technical trade-off. | Comp #14, Critic structural |
| N4 | Promote Additional Resources and Future Improvements to `##`. Follow the standard's section order, with Open Questions last. | Critic structural |
| N5 | Use one rule for what an unverified sentence shows (text greyed, labels and reason, no citation details) in G1, the Technical Plan and the page spec. | Comp #5, Style M5 |
| N6 | Fix the Overview's claim that Kaggle parses. Define Phase 4, or relabel it. Add year ranges for the eras and the Era D labels. | Comp #6–8 |
| N7 | Expand the acronyms (CNR, SCC, AIR, INSC, BNS/BNSS/BSA). Add a reference for each named paper. Remove "Goldfish" and references to "the brief". | Comp #9–12, Critic structural |
| N8 | Correct the research claims (the German code study wording, the CRAwLeR claim, "worst size compliance"). The Era C margin-letter claim refers to Docling, not PyMuPDF. | Style |
| N9 | Replace "about" in targets with exact starting values. State the basis of each size estimate. Remove value words. | Style |
| N10 | Fix the README download command, the PyMuPDF import name and pin, `nc_display` without spaces, the `.gitignore` lines, `timings_ms`, and renumber 49a. | S9 nits |
| N11 | Explain that G4 at about 250,000 vectors mostly tests crowding, not approximate search. | ZO-N10 |
| N12 | Start with a single process. Add a pool only if measured run time calls for it. | ZO-N8 |

## Findings made moot by another fix

*   **If D1 removes the quote route:** IC-2, the Critic's short-query fix
    and routing set, and the quote-route halves of ZO-4, Comp #3 and Critic
    "lookup routes".
*   **If D2 builds G4 steps in temporary schemas:** ZO-6(a), ZO-6(c), and
    the exemption of evaluation versions from retention.
*   **After M4 (no Era A/B machinery):** the Critic's margin-letter
    assignment rule, and IC-4's bounding box.
*   **After the paragraph-label gold format (B1):** IC-4's anchor resolver,
    and IC-6's "report G3 twice".
*   **After ZO-N3:** IC-N3.
*   **After ZO-N6:** the Critic's duplicate tie-break.
*   **Other dead ends:**
    *   the "8,000-token model" branch of the chunk-token finding, because
        rerankers are limited to 512 tokens;
    *   IC-1's RapidFuzz repair fallback, because it would weaken G1.

## Traceability

Every finding ID in the stage reports maps to an action above, or to the
moot list:

*   **Stage 5** (`zoom_out_report.md`):
    *   ZO-1 → B1;
    *   ZO-2 → M1;
    *   ZO-3 → M2;
    *   ZO-4 → B5;
    *   ZO-5 → M3;
    *   ZO-6 → B6;
    *   ZO-7 → M7;
    *   ZO-8 → M4;
    *   ZO-N1, N2, N4 → M5;
    *   ZO-N3 → M20;
    *   ZO-N5 → M26 and normalisation in B2;
    *   ZO-N6 → M15;
    *   ZO-N7 → M16;
    *   ZO-N8 → N12;
    *   ZO-N9 → M6;
    *   ZO-N10 → N11;
    *   IC-1 → M6 (deferred variant);
    *   IC-2 → M2;
    *   IC-3 → B6;
    *   IC-4 → B1 (moot part listed above);
    *   IC-5 → M6 and M12;
    *   IC-6 → B1;
    *   IC-7 → M6;
    *   IC-N1, N2, N3 → M6.
*   **Stage 6** (`comprehension_report.md`):
    *   #1 → N1;
    *   #2 → B6;
    *   #3 → B5;
    *   #4 → M11;
    *   #5 → N5;
    *   #6–8 → N6;
    *   #9–12 → N7;
    *   #13 → M22;
    *   #14 → N3.
*   **Stage 7** (`critic_report.md`):
    *   **Design:**
        *   Risks → B7;
        *   filtered search → B6 and M18;
        *   chunk size → M8;
        *   lookup routes → B5;
        *   quote proof → M10;
        *   Act filter → M5;
        *   short query → M2;
        *   sub-paragraphs → M9;
        *   duplicates → M15;
        *   margin letters and sample → M13;
        *   LLM quota → M12;
        *   headnotes → B1;
        *   RAM and bind address → M21 and N9.
    *   **Structural:**
        *   Open Questions → N2;
        *   G4 list → N3;
        *   Option B → N1;
        *   rejection credit and alternatives → N3;
        *   headings and order → N4;
        *   `PassageRef` → B5;
        *   references → N7.
    *   **Style:**
        *   untested parsing → M13;
        *   duplicate claim → M15;
        *   research claims and Era C → N8;
        *   "about" → N9;
        *   BM25 trigger → M25;
        *   size basis and value words → N9.
*   **Stage 8** (`intermediate_zoom_out_report.md`):
    *   #1 → M2 and D1;
    *   #2 → B5, M12 and D3;
    *   #3 → B6 and D2;
    *   #4 → B1;
    *   #5 → M1, M5 and M6;
    *   #6 → B7;
    *   #7 → the moot list;
    *   #8 → M6.
*   **Stage 9** (`implementation_readiness_report.md`):
    *   B1 → B2;
    *   B2 → B3;
    *   B3 → B4;
    *   B4 → B5;
    *   B5 → B6;
    *   M1 → B6;
    *   M2 → M18;
    *   M3 → M8 and M19;
    *   M4 → M19;
    *   M5 → M9 and M20;
    *   M6 → M20;
    *   M7 → M5;
    *   M8 → M3 and M10;
    *   M9 → M18;
    *   M10 → M21;
    *   M11 → M22;
    *   M12 → M23;
    *   CD-M1 and M6 → M14;
    *   CD-M2 and M4 → M15;
    *   CD-M3 → M16;
    *   CD-M5 → M17;
    *   CD-M7 to M10 → M24;
    *   CD-M11 → M21;
    *   Style M1 → M26;
    *   Style M2 → B2;
    *   Style M3 → M1;
    *   Style M4 → M5;
    *   Style M5 → N5;
    *   Style M6 → M23;
    *   all nits → N10.
