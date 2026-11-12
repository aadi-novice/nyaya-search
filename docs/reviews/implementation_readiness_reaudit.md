# Stage 9 Re-audit: Implementation Readiness

*   **Target:** `docs/designs/nyaya_search.md`, the implementation plan (lines
    1191–1996 before this fix pass).
*   **Reviewers:**
    *   `egm-implementation-readiness` and `egm-codebase-details`, both
        checked against the implementation-readiness standard;
    *   `egm-style`.
*   **Purpose:** the one re-check allowed by the review rules, run after the
    first round of fixes. The reviewers checked whether each earlier Blocker
    and Major was resolved, and looked for new problems the rewrite caused.
*   **Not re-raised:** E2 (the second search engine) is an open question by
    design. E1 was decided by the author: keep the Act filter, and extract
    Era C Acts from the catchwords.

## Verdict at re-audit time

*   **Blockers:** none new.
*   **Phase 0:** ready after small fixes.
*   **Phase 1:** not ready, until N1, N2, N7, N8 and N9 and the Partly items
    were fixed. All were addressed in the fix pass described below.

## (a) Status of the earlier Blockers and Majors

*   **Resolved at re-audit:**
    *   Blockers B2, B3 and B5;
    *   Majors M2, M4, M5, M9, M10 and M11;
    *   codebase findings CM3, CM4, CM6, CM7, CM8, CM9 and CM11;
    *   style findings S4–S6.
*   **Partly resolved, then fixed in this pass:**

| Item | What was left | Fix applied |
| --- | --- | --- |
| B1 contracts | `get_judgment` returned an untyped dict; `ParsedDoc` had no page count | `JudgmentRecord` contract; `ParsedDoc.pages` |
| B4 routes | the citation route's payload; no Era C fallback | "What citation_lookup returns": issue then editorial chunks (Era D); the first 3 editorial chunks (Era C); otherwise the first judgment chunk |
| M1 lifecycle | how `g4_step_n` schemas get their tables | `LIKE … INCLUDING DEFAULTS INCLUDING GENERATED`, without foreign keys; rows copied from the active version; own indexes |
| M3 embedding | the header was given only as an example | exact header format string |
| M6 zoning | markers described in prose | line-anchored regexes per zone and era |
| M7 filters | judge matching undefined | `normalize_judge()`, `judges_norm` column with a GIN index |
| M8 citation check | no "X v. Y" rule | extraction regex and a title-containment match |
| M12 tests | smoke questions undefined | `phase1_smoke.jsonl` defined (file 41); `test_acts.py` and `test_para_links.py` added |
| CM1 columns | `outcome_map` empty; `year` column unmapped | measured `disposal_nature` values mapped; `year` dropped, with the reason given |
| CM2 duplicates | the row's source folder unknown; garbled lines | `folder_year` taken from the Parquet file path; text rewritten |
| CM5 eras | `era_markers.c` undefined | explicit Era C rule |
| CM10 files | smoke and hand-label files unlisted | added to file 41, plus the manifest fixture in file 14 |
| S1 settings | hard-coded values remained | moved into the key table, and referenced by key |
| S2 normaliser | `normalize()` was undefined | `normalize_text()` and `normalize_judge()` defined in file 11, and used everywhere |
| S3 phases | goals assigned to two phases | one phase per goal; G6 split |

## (b) New findings from the re-audit, and how each was resolved

| ID | Finding | Resolution |
| --- | --- | --- |
| N1 | The Era C Act rule did not fit the real text: dashes, wrapping, abbreviations, several Acts per group, Acts without a year, OCR ";" | Checked on 15 sampled judgments. Lines joined, dashes normalised, groups split outside abbreviations, every Act captured, a no-year allowlist, ";" accepted |
| N2 | The post-load check could not run: no active version yet, and it called the LLM | `PostgresSearch(index_version=…)`; a smoke check of routing, retrieval and reranking only, passing at 8 of 10 or better |
| N3 | Calibrate needed eras before they were assigned, and wrote TOML | Samples by decision year, confirmed with `classify`; Era C only; writes `build/bands.json`, merged by `load_settings` |
| N4 | The threshold had two homes | The gate reads `index_versions`; config only seeds it |
| N5 | The development set's phase was inconsistent | Phase 1 throughout |
| N6 | G6's phase was inconsistent | The automatic metrics are a Phase 1 gate; the hand check is Phase 2 |
| N7 | `phase1_smoke.jsonl` was undefined | Defined in file 41 |
| N8 | No regex for paragraph links or Held points | Regexes for lists and ranges, OCR ")" and stray letters; an unnumbered-HELD fallback; both eras extract links |
| N9 | Reranker calls unspecified | Qwen3 yes/no-logit scoring from the model card; OpenRouter `POST /api/v1/rerank` through `httpx` |
| N10 | Supplement citations could collide; aliases pointed at paths not loaded | `scr:YYYY:sN:P` form; aliases only for loaded paths, keyed by version |
| N11 | The retry deadline was unnamed | `timeouts.answer_deadline_s` = 15 |
| N12 | CUJ 3 lacked a check on a 2010–2023 judgment | Checklist uses `2018_10_937_952` through its catchword Act |
| Nits | listed in the re-audit | All applied: CNR regex, `?highlight=` meaning, CUDA index URL, `pydantic-settings`, enum casing, full Zone enum, one reference-vector path, the reranker cap, chunk keys, `skipped.csv` columns, version-keyed indexes, rows without a PDF, a `g6-holdout` command, safety-critical labels |

## Accepted trade-offs

The review rules allow only this one re-check, so these remain open:

*   **OpenRouter rerank response fields:** not confirmed from the
    documentation. They are checked with one real call at implementation.
*   **Act and paragraph-link regexes:** checked on 15 sampled judgments, not
    the whole slice. Misses are logged and fixed by adjusting the patterns.
