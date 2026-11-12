# Goldfish Annotated Review Report

*   **Target:** `docs/designs/nyaya_search.md` (full file).
*   **Findings source:** the consolidated actions in
    `consolidated_review_report.md` (B1–B7, M1–M26, N1–N12). Each action
    traces back to the stage reports.
*   **Author decisions applied (2026-11-12):**
    *   **D1:** the quote-lookup route is removed from v1.
    *   **D2:** G4 size steps go in temporary schemas; production keeps the
        active version plus one previous version.
    *   **D3:** a new `passages` status, used for citation lookup and when
        the LLM fails.
    *   **D4:** the development set is built in Phase 1 from Held → [Para N]
        links.
    *   **D5, models:**
        *   **Embedder:** Qwen3-Embedding-0.6B, run locally (provisional),
            with BAAI/bge-m3 as the challenger.
        *   **Vectors:** 1024 dimensions.
        *   **Reranker:** Qwen3-Reranker-0.6B, run locally, falling back to
            Cohere Rerank 5 Fast through OpenRouter (paid, used only on
            failure).
        *   **Refusal thresholds:** 0.35 for Qwen3 and 0.20 for Cohere
            (provisional).
    *   **Search:** HNSW from the start, with the safeguards against
        under-retrieval.
    *   **Scaling out:** Qdrant is the planned path, with measured triggers.

**Summary:** 2 ESCALATE, 0 CONTEST, 0 DECLINE, and all other findings
PROPOSE FIX.

---

## Escalated (needs the author's knowledge)

### [E1] [`egm-design`, `egm-zoom-out`, `egm-implementation-readiness`] - Severity: Major (consolidated M5, Act filter part)
*   **Affected Scope:** Goals G9 and CUJ 4; Technical Plan "Filters"; file
    29; the `refs` table.
*   **Discrepancy Description:** "List of Acts" exists only in Era D (2024
    onward). An Act filter would quietly drop judgments from 2010–2023.
    Stage 8 proposes cutting the Act filter and the `refs` table from v1.
*   **Rationale:** a filter with no data for most of the slice gives
    silently wrong results.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** the reviewers' facts are right. But the author
    added the Act filter on purpose (the "RAG at 10M" review). The choice
    is between two options:
    *   **(a) Keep the Act filter:** extract Acts from the Era C catchword
        prefixes (the text before the first ":" in the catchwords), and
        normalise Act names into one list.
    *   **(b) Cut it from v1:** keep the year, judge and outcome filters.
*   ***Proposed Edit:*** depends on the author's choice.

### [E2] [`egm-zoom-out`] - Severity: Major (consolidated M7)
*   **Affected Scope:** Alternative N; the tools table row "Search engine
    beyond v1"; file 28.
*   **Discrepancy Description:** the seven-method `SearchEngine` interface
    is shaped by OpenSearch, which is not part of v1. The reviewer proposes
    moving the OpenSearch comparison to Future Improvements and writing the
    search code as plain functions.
*   **Rationale:** an interface designed for an imagined second engine is
    likely to be wrong.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** the author decided "PostgreSQL in v1, plus a
    planned OpenSearch comparison", and has since named Qdrant as the path
    for scaling out. Two candidate second engines now point to different
    plans. The author decides between:
    *   keeping the OpenSearch comparison planned;
    *   replacing it with a Qdrant comparison;
    *   moving both to Future Improvements and writing plain functions now.
*   ***Proposed Edit:*** depends on the author's choice.

---

## Blockers

### [B1] [`egm-zoom-out`, `egm-inventive-catalyst`, `egm-design`] - Severity: Blocker
*   **Affected Scope:** Goals intro; Open Questions "Test set"; the
    evaluation harness; files 43 and 50.
*   **Discrepancy Description:** Phase 1 steps need a development set that
    was deferred.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:***
    *   **New file 43b:** `src/nyaya/eval/devset_from_held.py`.
        1.  It collects Era C/D Held points that have resolvable "[Para N]"
            links.
        2.  It drafts one question per Held point through the free LLM,
            with paraphrasing rules: no run of 5 or more words copied from
            the Held point, and no party names.
        3.  The author hand-checks 50 of them.
        4.  It writes `eval/datasets/dev_v1.jsonl`, about 300 questions.
    *   **Gold label format:** (path, paragraph labels). This replaces
        character spans.
    *   **Scoring:** a Held hit is replaced by the judgment chunks it links
        to, and only judgment-zone chunks count as hits.
    *   **Test set:** stays separate and comes later. IndicLegalQA or
        hand-written questions are candidates. It is never used for tuning.
    *   **Open Question:** the test-set question shrinks to cover the final
        test set only.

### [B2] [`egm-implementation-readiness`, `egm-codebase-details`, `egm-style`] - Severity: Blocker
*   **Affected Scope:** Shared contracts; files 5 and 7.
*   **Discrepancy Description:** shared types are missing, and
    `settings.toml` lists topics rather than keys.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:***
    *   **Add to the Shared contracts:**
        *   `ParsedDoc {path, era, clean_text, position_map: [(char_offset, page)], pages}`
        *   `Era {C, D, out_of_scope}`
        *   `Zones {header, issue?, editorial[], judgment_span}`
        *   `Opinion {author, type, char_start, char_end}`
        *   `SentenceRecord`: removed (D1)
        *   the return types of the search functions
    *   **Settings:** give the full list of `settings.toml` keys, with their
        types and defaults.

### [B3] [`egm-implementation-readiness`] - Severity: Blocker
*   **Affected Scope:** the tools table; file 9; the Verification Plan.
*   **Discrepancy Description:** models, dimension and threshold are needed
    before the step that chooses them.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:***
    *   **Provisional choices:** the D5 models, `vector(1024)`, and the
        thresholds 0.35 (Qwen3) and 0.20 (Cohere).
    *   **Dimension:** fixed at 1024 for both candidate embedders, so the
        migration needs no parameter. Changing the dimension needs a new
        migration and a full rebuild.
    *   **Phase 1 pass criteria:** written so they do not depend on the
        threshold.

### [B4] [`egm-implementation-readiness`, `egm-codebase-details`] - Severity: Blocker
*   **Affected Scope:** files 24 and 27; `GET /judgment`.
*   **Discrepancy Description:** no step writes the files the loader reads.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:***
    *   **Name the build outputs:**
        *   `build/manifest.parquet`
        *   `build/judgments/<year>.parquet`, which includes clean_text and
            position_map
        *   `build/chunks/<year>.jsonl`
        *   `build/aliases.parquet`
        *   `build/links/<year>.parquet`
        *   `build/vectors/<year>.npy` plus `ids/<year>.txt`
    *   **Load order:** judgments, then chunks, then vectors, then aliases,
        then links.
    *   **Schema:** add `clean_text` and `position_map` (JSONB) to
        `judgments`.
    *   **Kaggle export:** removed from Phase 1 (embedding is local, see M6).

### [B5] [`egm-zoom-out`, `egm-comprehension`, `egm-design`, `egm-implementation-readiness`] - Severity: Blocker
*   **Affected Scope:** Router; AskResponse; pipeline file 39; CUJs 2 and
    3.
*   **Discrepancy Description:** the routes cannot fill the response
    contract.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:***
    *   **A table of routes**, showing which stages each one runs:
        *   **citation_lookup:** alias lookup only, returning `status:
            passages` with PassageRefs to the issue or headnote and the
            judgment.
        *   **hybrid:** every stage.
        *   **LLM failure:** `status: passages`.
    *   **New type:** `PassageRef {chunk_id, judgment_id, case_name,
        scr_citation, neutral_citation?, paras?, pages, zone,
        opinion_type?}`, used by `nearest_passages` and by `passages`.
    *   **Fallbacks:** an unresolved alias, or an Era C judgment with no
        issue zone, falls back to the hybrid route.
    *   **Removed:** CUJ 3 (D1).

### [B6] [`egm-comprehension`, `egm-zoom-out`, `egm-inventive-catalyst`, `egm-design`, `egm-implementation-readiness`] - Severity: Blocker
*   **Affected Scope:** Load; NFR rollout; files 9, 27 and 46; `.env`.
*   **Discrepancy Description:** the index-version lifecycle is
    contradictory and conflicts with G4.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:***
    *   **Status lifecycle:** building → ready → active → retired, and
        `activate` accepts only `ready`.
    *   **Active version:** recorded only in the database, and
        `ACTIVE_INDEX_VERSION` is removed.
    *   **Retention:** production keeps the active version and one previous
        version. The previous one stays restorable until the next
        activation succeeds.
    *   **Post-load checks:** the vector IDs equal the chunk IDs; G6 passes;
        and the 10 Phase 1 questions run.
    *   **Keys:** (index_version, path), with `ON DELETE CASCADE`.
    *   **G4 size steps** (D2): each in its own temporary schema `g4_step_n`
        with its own HNSW index, dropped after the run. They never go
        through `index_versions`.
    *   **HNSW from the start:** pgvector 0.8 or later, with `SET LOCAL
        hnsw.iterative_scan = relaxed_order`, `hnsw.ef_search = 100`, and
        rows returned compared with k recorded in the trace.

### [B7] [`egm-design`, `egm-structural`] - Severity: Blocker
*   **Affected Scope:** a new section.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Proposed Edit:*** add `## Risks & Mitigations`, written last. It
    covers:
    *   free-tier LLM limits, outages and model retirement;
    *   provider data retention;
    *   cost of the hosted reranker fallback;
    *   filtered HNSW returning too few rows;
    *   margin-band miscalibration;
    *   a dissent labelled as `unknown` or misattributed;
    *   citation-pattern false matches;
    *   Postgres full-text search weaker than BM25;
    *   threshold drift;
    *   the 4 GB Postgres memory cap and WSL2 memory pressure;
    *   judge disagreement;
    *   paraphrase leakage in the development set.

## Majors

| ID | Suggestion | Proposed edit |
| --- | --- | --- |
| M1 | PROPOSE FIX | Rewrite the build order: Phase 0 contracts; **Phase 1 = the 11-step minimum path plus the development set**; Phase 2 measured gates; Phase 3 full evaluation, lawyer review and model challenger; Future = Era A/B, Qdrant, the deferred items. Label every file with its phase. |
| M2 | PROPOSE FIX | Remove quote_lookup, CUJ 3, the `sentences` table, the trigram index, `fuzzy_sentences` and the `pg_trgm` extension. Quoted text in a question becomes a phrase query (`phraseto_tsquery`) ORed into the keyword search. |
| M3 | PROPOSE FIX | Citation check step 4 scans only for SCR, INSC, SCC and AIR patterns and full "X v. Y" names, against the retrieved passages. Drop party names from `aliases`. |
| M4 | PROPOSE FIX | The classifier returns C, D or `out_of_scope` (logged). Remove `docling_ocr.py`, `kaggle/parse_era_a.py` and the `ocr` extra from v1 (keep a note that `uv sync` removes Docling, see M24). The position map is (char_offset, page). |
| M5 | PROPOSE FIX (except the Act part, E1) | Remove `cites_case` links, table detection and `oversize_table`. The statute notice becomes a standing notice, and rate tracking is removed. |
| M6 | PROPOSE FIX | **Defer:** provider failover for the answer LLM, IC-1 span mode, `/search`, the probe-set threshold check, the headnote opinion hint, and the vector hash cache. **Phase 1 embedding runs locally** on the RTX 2060 (Qwen3-Embedding-0.6B at fp16), with Kaggle as the fallback for full-corpus runs. The cross-machine 0.999 check applies only when Kaggle vectors are used. |
| M7 | ESCALATE (see E2) | — |
| M8 | PROPOSE FIX | Chunk sizes are in whitespace words: target 400, minimum 80, maximum 700. Both chosen embedders accept 8,000 or more tokens, and Qwen3-Reranker accepts 32,000, so nothing is cut off. G6 adds the metric "share of chunks over the model limit = 0". The departure from the anatomy document's 400 is explained: 400 stays the target. |
| M9 | PROPOSE FIX | `para_start` and `para_end` become `para_labels: [string]` ("12", "12.1"). `[Para N]` resolves to chunks whose labels include N or start with "N.". |
| M10 | PROPOSE FIX | The minimum quote length becomes 8 words. G1 and Alternative D say "the quote exists in the passage"; support is measured by G2. |
| M11 | PROPOSE FIX | **Every sentence must carry a citation.** The prompt tells the model to put "the passages do not address X" in a separate `gaps` field. `gaps` is shown as plain text and never counted as a rule break. |
| M12 | PROPOSE FIX | **Budgets:** 6,000 input tokens and 800 output tokens per request. **Retries:** one JSON repair retry, only if 4 or more seconds remain before 15. **Errors:** a 429, timeout or 5xx returns `status: passages`. **Quota:** each run records tokens and the provider's quota headers. |
| M13 | PROPOSE FIX | **New step "calibrate-bands":** on 50 PDFs per era with seed 42, cluster the x-positions of lone uppercase A–H letters per era, write the bands to config, and stop the build if no cluster is found. **Held-out G6 sample:** 50 other PDFs, seed 4242. **Wording:** the parsing is described as expected, not yet tested. |
| M14 | PROPOSE FIX | **Mapping from metadata columns** (`title`, `petitioner`, `respondent`, `judge`, `author_judge`, `citation` → scr_citation, `nc_display` → neutral_citation, normalised with spaces, `cnr`, `disposal_nature` → disposal plus an outcome lookup, `decision_date` dd-mm-yyyy → DATE plus `decision_year`) and the columns dropped (`description`, `raw_html`, `court`, `case_id`, `scraped_at`, `available_languages`). |
| M15 | PROPOSE FIX | **Duplicate rule:** keep the copy that has a PDF, then the later `scraped_at`, then the lowest tar path. `<year>` means the decision year. The manifest covers all years; the no-PDF count is recomputed after removing duplicates. |
| M16 | PROPOSE FIX | Join on `<path>_EN.pdf`, looking in the kept row's tar first and then any tar. Store `offset_data`. Port the code from `scripts/data_check.py` L65–100. |
| M17 | PROPOSE FIX | Era classification moves into `parse-chunk`, using the first 3 pages, with markers seeded from `scripts/layout_study.py` `MARKERS`. |
| M18 | PROPOSE FIX | **Keyword search:** an OR `to_tsquery` over normalised words plus phrase parts, ranked by `ts_rank_cd`. **Expansion:** an added chunk is ranked at its parent's rank + 0.5, and the cap of 100 applies after expansion. **Reranking** uses header + text. **LLM output:** `{"sentences": [...], "gaps": [...]}`. |
| M19 | PROPOSE FIX | **Embedding text:** `"{header}\n\n{text}"`. **Queries:** Qwen3 uses the prefix `"Instruct: Given a legal research question, retrieve Supreme Court judgment passages that answer it\nQuery: "`; documents get no prefix. **Kaggle job:** writes `reference/passages.json` and `reference/vectors.npy`. **PyTorch:** pinned to the CUDA 12.x wheel index through `[tool.uv.sources]`, with an assert on `torch.cuda.is_available()` at startup. |
| M20 | PROPOSE FIX | **Zone patterns per era** for C and D. **Era D author:** the "*" marker and the "* Author" footnote. **Opinion phrases:** matched only in the first 300 characters of each opinion; `majority` is never inferred from position. **Sentence splitting:** a regex splitter with the abbreviations J., JJ., v., vs., Ors., Anr., No., Sr., Adv., i.e., e.g., viz., s., ss., Art., para. |
| M21 | PROPOSE FIX | **Build logs:** `build/skipped.csv` and `build/errors.jsonl`; the build fails if errors exceed 1% of judgments. **Timeout:** 30 seconds per request through `asyncio.wait_for` in a thread pool; on timeout, `status: error` and the trace is written. |
| M22 | PROPOSE FIX | **Dataset fields:** `exclude: [path]` and `scored_zones: ["judgment"]`. **G4 passes** when the upper bound of the 95% confidence interval for the drop is 5 points or less. **Wrong answer:** a verified sentence the judge rates `unsupported`. **Judge scoring:** `partial` counts as unsupported for G2; a sentence with several citations passes if any citation is rated supported. **Hand labels:** `eval/datasets/hand_labels_v1.jsonl`, with the agreement measured by Cohen's κ ≥ 0.6. **New commands:** `nyaya-eval tune-threshold` (maximise correct refusals subject to wrong refusals ≤ 10%) and `nyaya-eval latency`. |
| M23 | PROPOSE FIX | **Named test files and cases** (see Tests). **Phase 1 questions:** 10 written ones with the expected route and judgment. **CUJs:** a checklist for each. **Margin-letter test:** the in-sentence pattern `(?<=[a-z,]) [A-H] (?=[a-z])`. **Database test:** skipped when Docker is absent. |
| M24 | PROPOSE FIX | **Dependencies:** an `eval` extra (bm25s, scipy); add pytest-cov and numpy; drop tomli-w. **Entry points:** `nyaya.build:main`, `nyaya.api.app:main`, `nyaya.eval.cli:main`. **New files:** list the `__init__.py` files and `db/init/00_extensions.sql`. **Docling:** note that `uv sync` removes it, so research scripts need `uv pip install docling`. |
| M25 | PROPOSE FIX | The BM25 trigger fires when BM25's recall@50 beats Postgres full-text search by **3 points or more** on the development set, and the 95% confidence interval excludes zero. |
| M26 | PROPOSE FIX | `config/settings.toml`: a complete key table with name, type and starting value for every threshold, seed, timeout and path. |

## Nits

| ID | Suggestion | Proposed edit |
| --- | --- | --- |
| N1 | PROPOSE FIX | "Option B" → "Alternative M (Qdrant + a BM25 library + SQLite)". |
| N2 | PROPOSE FIX | Close the Open Questions on the vector store, snapshot date, rows without a PDF and headnotes. |
| N3 | PROPOSE FIX | Fix the G4 list indentation. Order Q before R. A `Decision:` field with the values Rejected, Deferred, Fallback or Planned. Lead each rejection with its technical reason. |
| N4 | PROPOSE FIX | `##` for Additional Resources and Future Improvements. Section order: Overview, Problem, Goals, Non-Goals, CUJs, Technical Plan, Alternatives, NFRs, Risks, Resources, Future, Open Questions, Implementation. |
| N5 | PROPOSE FIX | One rule, in all three places: the unverified sentence's text is shown greyed out, with its passage labels and the reason, and no citation details. |
| N6 | PROPOSE FIX | Correct the Overview's Kaggle claim. Use the Future label instead of Phase 4. Add year ranges for the eras (A 1950s–60s, B 1970s–2000s, C 2010–2023, D 2024+) and the Era D labels. |
| N7 | PROPOSE FIX | **Glossary:** CNR, SCR, SCC, AIR, INSC, BNS, BNSS, BSA, IPC, CrPC. **Papers:** each one points to `docs/research_notes.md`. **Wording:** "Goldfish" becomes "external review"; "the brief" is replaced with its content. |
| N8 | PROPOSE FIX | Fix the claims on the German code study, CRAwLeR and size compliance. Era C margin letters: PyMuPDF keeps them on separate lines; Docling merges them. |
| N9 | PROPOSE FIX | Exact starting values replace "about" in targets. State the basis of each size estimate. Remove value words. |
| N10 | PROPOSE FIX | Fix the README download command, `import pymupdf` (pinned at 1.26+ or the installed version), `nc_display` without spaces, the gitignore lines, `timings_ms`, and renumber 49a. |
| N11 | PROPOSE FIX | Note that G4 at about 250,000 chunks mainly tests crowding. |
| N12 | PROPOSE FIX | Use a single process first. Add a pool only if measured time calls for it. |
