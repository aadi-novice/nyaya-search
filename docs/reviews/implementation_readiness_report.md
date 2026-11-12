# Stage 9: Implementation Readiness Test

*   **Target:** `docs/designs/nyaya_search.md`, the implementation plan only
    (lines 1051–1858).
*   **Reviewers:**
    *   `egm-implementation-readiness` and `egm-codebase-details`, both
        checked against the implementation-readiness standard;
    *   `egm-style`, using its own checks.

    All three completed. The codebase reviewer checked the plan's claims
    against the actual files in `D:/Coding_me/rag`.
*   **Verdict: not ready.** An engineer would still have to make design
    decisions in several places.
*   **Findings:** 5 Blockers, about 24 Majors, and a set of Nits, after
    duplicates were removed.
    *   When reviewers rated the same problem differently, the highest
        rating is used.
    *   A finding raised by more than one reviewer is listed once.

## Implementation readiness (also holds findings shared across reviewers)

### Blocker 1: the shared contracts are incomplete, so parallel work cannot start

*   **Where:** L1071–1078 and L1156–1174.
*   **Problem:** `contracts.py` does not define these shared types:
    *   `ParsedDoc` and the position map;
    *   `Era`, `Zones` and `Opinion`;
    *   the sentence record: its ID format, and whether offsets point into
        the clean text or the normalised text;
    *   the return types of the `SearchEngine` methods;
    *   the `refs` columns (line 1207 says only "structured rows");
    *   the `Settings` keys (item 5 lists topics, not keys).
*   **Fix:**
    *   Add a pydantic model with exact fields for each type.
    *   Write the full table definition for `refs`.
    *   Give the exact list of `settings.toml` keys, with defaults.

### Blocker 2: the vector size, models and refusal threshold are needed before the step that chooses them

*   **Where:** L1197–1199, L1140, L1674 and L1770–1780.
*   **Problem:**
    *   The Phase 0 migration needs `vector(N)`, and Phase 1 needs an
        embedding model and a reranker. But `select_models.py` needs
        development questions that only arrive later.
    *   A plain `.sql` file cannot be "parameterised".
    *   One `vector(N)` column cannot hold two index versions whose models
        have different dimensions.
    *   The Phase 1 check needs a threshold that is only set "after the
        development run".
*   **Fix:**
    *   Name a provisional model, reranker, dimension and threshold for
        Phase 1.
    *   Pick how the migration gets the dimension: an `{EMBED_DIM}`
        placeholder filled in by `apply_migrations`, or an untyped `vector`
        column with one partial index per version.
    *   Say what happens when the dimension changes.
    *   Make the Phase 1 pass criteria independent of the threshold.

### Blocker 3: no build step writes the files the loader reads

*   **Where:** L1416–1421, L1453–1457 and L1632.
*   **Problem:**
    *   `load()` bulk-loads judgments, aliases, links and refs, but
        `parse-chunk` writes only chunks and sentences.
    *   `export-for-kaggle` has no defined output.
    *   `GET /judgment/{path}` returns clean text that no table stores.
*   **Fix:**
    *   Name the build outputs:
        *   `build/judgments.parquet`;
        *   `build/aliases.parquet`;
        *   `build/links/<year>.parquet`;
        *   `build/refs/<year>.parquet`;
        *   `build/clean/<year>`.
    *   Give the order in which they are loaded.
    *   Add `clean_text` and `position_map` (as JSONB) to `judgments`.
    *   Define `build/kaggle_export/`, including its
        `dataset-metadata.json`.

### Blocker 4: the citation and quote routes cannot fill the response contract

*   **Where:** L1596–1607.
*   **Problem:**
    *   The citation route makes no LLM call, yet `answer[].sentence` and
        `Citation.quote` are defined as LLM output.
    *   Era C has no issue zone.
    *   An unresolved alias has no defined behaviour.
    *   SCC and AIR citations are never in the alias table.
    *   For the quote route, the plan does not say whether reranking, the
        refusal gate and the check run.
*   **Fix:**
    *   For each route, list the steps that run, and the exact status,
        sentence and quote values it returns.
    *   Add a fallback for Era C and for unresolved aliases, for example
        sending them to hybrid search.

### Blocker 5: the loader writes a status value the schema forbids

*   **Where:** L1189 and L1459.
*   **Problem:** the status values are building, active and retired, but the
    loader sets `ready`. All three reviewers raised this.
*   **Fix:**
    *   Use the lifecycle building → ready → active → retired.
    *   `activate` accepts only versions that are `ready`.

### Major 1: the index-version lifecycle

*   **Where:** L1120, L1191–1193, L1202, L1460–1462 and L1704.
*   **Problem:**
    *   The active version is recorded in two places: `.env` and the
        database status.
    *   "Two non-retired versions" conflicts with line 987.
    *   The size-step evaluation loads three or more versions.
    *   The `judgments` key is `path` alone, and `aliases` and `refs` have
        no version column.
*   **Fix:**
    *   One place records the active version.
    *   A `purpose` value, or an `eval` status, keeps evaluation versions
        out of the retention rule.
    *   Key by (index_version, path), or declare those tables shared across
        versions.
    *   Use `ON DELETE CASCADE`.

### Major 2: filtered searches will return too few results

*   **Where:** L1483–1491.
*   **Problem:**
    *   A filtered HNSW query returns fewer than k rows.
    *   `websearch_to_tsquery` requires every word of a natural-language
        question to match.
    *   `%` with `similarity()` fails on short fragments, and the trigram
        threshold is a database setting, not a parameter of the query.
    *   The exact-substring step from the research notes is missing.
*   **Fix:**
    *   `SET LOCAL hnsw.iterative_scan = relaxed_order` plus
        `hnsw.ef_search`, with a pinned pgvector version.
    *   Build an OR query ranked with `ts_rank_cd`.
    *   Use `word_similarity` or `<%`, with a named threshold key, and
        normalise the fragment first.
    *   Try an exact substring match before the fuzzy match.

### Major 3: embedding details are missing

*   **Where:** L1431–1436 and L1519–1526.
*   **Problem:**
    *   The format of header plus text, and any query or passage prefix,
        are not specified.
    *   700-word chunks exceed the usual 512-token limit.
    *   The startup check compares against reference vectors that the
        Kaggle job never writes. All three reviewers raised this.
*   **Fix:**
    *   Fix the format string and the prefix rule.
    *   Set a rule for cutting off long text, or lower the maximum size.
    *   Have the Kaggle job write `reference/passages.json` and
        `vectors.npy`.

### Major 4: PyTorch from PyPI on Windows is CPU-only

*   **Where:** L1091–1096.
*   **Fix:**
    *   Pin PyTorch to a CUDA wheel index through `[tool.uv.sources]`.
    *   Check `torch.cuda.is_available()` at startup.

### Major 5: gaps in the chunker

*   **Where:** L1365–1379.
*   **Problem:**
    *   No step creates the issue and editorial chunks.
    *   The fallback for unnumbered paragraphs, and the handling of
        `oversize_table`, are missing.
    *   `section_label` is never extracted.
    *   `para_start` is an integer, but Era D uses "12.1".
    *   No sentence splitter or abbreviation list ("J.", "v.", "Ors.") is
        named.
    *   It is unclear whether sizes are in words or tokens.
*   **Fix:**
    *   Add an explicit step for each gap.
    *   Store paragraph numbers as strings.
    *   Name the sentence splitter and its abbreviations.
    *   Define size as words separated by whitespace.

### Major 6: the zoning and opinion rules cannot be decided as written

*   **Where:** L1337–1357.
*   **Problem:**
    *   The Era C rule "the line before … HELD" does not say which zone it
        starts or ends.
    *   Era D Held points have no "1.1" numbering.
    *   The Era D "*" author marker is ignored.
    *   The source of the bench list is not stated.
    *   The phrase rules have no scope or order of precedence, and conflict
        with line 421.
*   **Fix:**
    *   Give a regular expression for where each zone starts and ends, per
        era.
    *   Match phrases only in about the first 300 characters of each
        opinion.
    *   Put one ordered list of phrases in config.

### Major 7: the Act filter has no data for 2010–2023

*   **Where:** L1485, L1207 and L1222.
*   **Problem:** only Era D has "List of Acts". Matching by judge and by
    outcome is also undefined.
*   **Fix:**
    *   Extract Acts from the Era C catchword prefixes.
    *   Define how Act, judge and outcome are matched.

### Major 8: details of the citation check are open

*   **Where:** L1564–1576.
*   **Problem:** several details are never stated:
    *   how case names are matched;
    *   what "outside the passages" means;
    *   which failure reason wins when several apply;
    *   the type of the `aliases` argument.
*   **Fix:** specify each of these.

### Major 9: link expansion, reranker input and the JSON shape

*   **Where:** L1511–1546.
*   **Problem:**
    *   Chunks added by link expansion have no rank, and it is unclear
        whether the cap of 100 applies before or after expansion.
    *   The reranker's input is not stated.
    *   JSON output modes need an object at the top level, but the planned
        output is an array.
*   **Fix:**
    *   Rank an added chunk at its parent's rank plus 0.5, and apply the cap
        of 100 after expansion.
    *   Rerank on header plus text.
    *   Use `{"sentences": [...]}`.

### Major 10: build failures are not handled

*   **Where:** L1298–1300 and L1416–1423.
*   **Problem:** the plan does not say what happens to judgments classified
    as another era or as unknown, or to an exception in one judgment during
    a parallel run.
*   **Fix:**
    *   Write skipped judgments to `build/skipped.csv`, and errors to
        `build/errors.jsonl`.
    *   Fail the build above a set count.

### Major 11: evaluation definitions are missing

*   **Where:** L1665–1749.
*   **Problem:**
    *   No field holds the anti-leakage exclusion.
    *   Editorial chunks are not excluded when scoring questions made from
        Held points.
    *   The G4 pass rule is undefined: is it the point estimate or a bound
        of the confidence interval?
    *   "Wrong answer" is undefined.
    *   How the judge scores a sentence with several citations, and how
        `partial` counts, are not stated.
    *   The hand-label format and the agreement measure are missing.
    *   No step tunes the threshold.
*   **Fix:**
    *   Define each of these.
    *   Add an `exclude` field and a `scored_zones` rule.
    *   Add `nyaya-eval tune-threshold`, with an explicit goal.

### Major 12: tests and verification are too thin

*   **Where:** L1797–1854.
*   **Problem:**
    *   Many modules have no tests, yet line 1854 requires 70% coverage, and
        `pytest-cov` is not a dependency.
    *   The Phase 1 check "answered or refused with a full trace" passes on
        any output.
    *   "The CUJs work by hand" has no checklist.
    *   The 50-PDF sample has no seed.
    *   G7 has no command.
*   **Fix:**
    *   Name the test files and their edge cases, including a database test
        that is skipped when Docker is absent.
    *   Write out the 10 Phase 1 questions, each with its expected route and
        judgment.
    *   Give a checklist for each CUJ, and a seed for the sample.
    *   Add `nyaya-eval latency`, which passes at a 95th percentile of 15
        seconds or less.

### Nits

*   **`Versions` (L1169):** add a config version, which G8 needs.
*   **Margin letters and pages (L1316):** state how a margin letter attaches
    to text, and how page numbers in the PDF map to printed page numbers.
*   **`normalize_citation` (L1398):** state its canonical output.
*   **Short-query rule (L1504):** give fixed values for the word limit and
    the list of question words.
*   **Lawyer ratings (L1766):** give a command and a format for importing
    them.
*   **Model selection (L1774):** name the candidate models.
*   **HNSW build (L1111):** say whether `maintenance_work_mem` is raised for
    it.

## Codebase details

### Major 1: the plan's column names do not match the metadata, and no mapping is given

*   **Where:** L1185–1193 and L1289.
*   **Problem:**
    *   The actual columns are `judge`, `citation`, `nc_display` (stored
        as `2025INSC20`), `disposal_nature`, and `decision_date` (text in
        the form dd-mm-yyyy).
    *   They do not match the plan's `judges`, `scr_citation`,
        `neutral_citation` and `disposal`.
    *   The filter calls the field `outcome`, while the schema calls it
        `disposal`.
*   **Evidence:** `docs/metadata/parquet/year=2025/metadata.parquet` and
    `docs/metadata/json/year=2025/2025_1_105_116.json`.
*   **Fix:**
    *   Add a mapping table: target column, source column, parse rule and
        type.
    *   Add a fixed lookup from `disposal_nature` to outcome.
    *   Decide whether `available_languages`, `court`, `case_id` and
        `description` are kept or dropped.

### Major 2: removing duplicates, and what "year" means, are undefined

*   **Where:** L1285–1290.
*   **Problem:**
    *   The `year` column matches the folder in only 88% of rows, so
        `<year>` in every per-year file name is ambiguous.
    *   There is no tie-break, and no preference for the copy that has a
        PDF.
*   **Fix:**
    *   Take the folder year from the path.
    *   Define `<year>` as the year of the decision date.
    *   Choose between duplicates by: has a PDF, then the folder matches the
        decision year, then the later scrape.

### Major 3: the join with the tar offsets is unspecified

*   **Where:** L1273–1277.
*   **Problem:** `TarInfo.offset` is the offset of the file's header, while
    `offset_data` is the offset of its contents, which is what
    `data_check.py:87` reads.
*   **Fix:**
    *   Join on the file name `<path>_EN.pdf`, looking in the kept row's
        tar first and then in any tar.
    *   Store `offset_data`.
    *   Port the code from `scripts/data_check.py`, lines 65–100.

### Major 4: the manifest counts cover the whole corpus, but the step's scope is not fixed

*   **Where:** L1827–1828 and L1416.
*   **Problem:**
    *   The figures 38,397 and 5,181 hold only when all years are included.
    *   The "7 without a PDF" figure was counted before duplicates were
        removed.

    All three reviewers raised this.
*   **Fix:**
    *   State that `manifest` covers all years.
    *   Recount the judgments without a PDF after removing duplicates.

### Major 5: era classification sits in the wrong step

*   **Where:** L1290, L1306 and L1312.
*   **Problem:**
    *   Classifying inside the manifest step forces that step to read every
        PDF.
    *   It ties together groups the plan builds in parallel.
*   **Fix:**
    *   Move classification to `parse-chunk`, using the first 3 pages.
    *   Start the marker rules from `MARKERS` in `scripts/layout_study.py`
        (lines 17–33) and from `docling_test.py:32`.
    *   List the running-head patterns.

### Major 6: an index names a column that does not exist

*   **Where:** L1220–1221.
*   **Fix:**
    *   Add a `decision_year SMALLINT` column.
    *   Store `decision_date` as a DATE.

### Major 7: dependencies are in the wrong group, or missing

*   **Where:** L1099–1100.
*   **Problem:**
    *   `bm25s` and `scipy` are listed as development-only, but the shipped
        `nyaya-eval` imports them.
    *   `pytest-cov` and `numpy` are missing.
    *   `tomli-w` is never used.
*   **Fix:**
    *   Add an `eval` extra.
    *   Add the missing packages.
    *   Drop `tomli-w`.

### Major 8: `uv sync` will uninstall Docling 2.150.0

*   **Where:** L1101 and L1825.
*   **Problem:** `scripts/docling_test.py` needs Docling.
*   **Fix:** run `uv sync --extra ocr`, or document that the script needs
    that extra.

### Major 9: the console-script entry points are unspecified

*   **Where:** L1102.
*   **Problem:** the plan gives no `module:function` targets, no
    `eval/cli.py` dispatcher, and no `__init__.py` files.
*   **Fix:** add these to the file list.

### Major 10: some files are referenced but not listed

*   **Where:** L1110, L1321–1327, L1747 and L1766.
*   **Problem:** these files are never listed:
    *   the Postgres init script;
    *   `kaggle/parse_era_a.py`, which belongs to a Phase 4 that the build
        order never defines;
    *   `hand_labels.jsonl`.
*   **Fix:**
    *   Add `db/init/00_extensions.sql` to the list.
    *   Mark Era A as future work.
    *   Fix the path of the hand-label file.

### Major 11: FastAPI and Uvicorn have no per-request timeout

*   **Where:** L1640.
*   **Fix:**
    *   Run each request through `asyncio.wait_for(..., 30)` in a thread
        pool.
    *   On timeout, return `error`, and still write the trace.

### Nits

*   **PyMuPDF (L1092):** version 2.1 is installed and imported as `fitz`.
    State the import name and pin the version.
*   **Neutral citations (L1395–1400):** `nc_display` has no spaces.
    `normalize_citation` must accept both forms, and a test must cover the
    form without spaces.
*   **Docling (L1321–1327):** point to `scripts/docling_test.py`, lines
    41–56, as working usage of the Docling API.
*   **Paragraph regex (L1366):** write it out. Earlier versions exist at
    `data_check.py:29` and `layout_study.py:31`.
*   **README (L1787):** the download command should be `python
    scripts/download_sc.py --dest docs --skip-regional`.
*   **`eval/` (L1078):** it is unclear whether this means the top-level
    `eval/` folder or `src/nyaya/eval/`.
*   **Model cache (L1079):** the claim that the Hugging Face cache is
    "already configured" could not be checked. Name the environment
    variable.

## Style

### Major 1: "single source of tunable values" is not true

*   **Where:** L1131.
*   **Problem:** these values appear elsewhere in the plan but are missing
    from the key list:
    *   the fuzzy-match threshold;
    *   the G6 thresholds;
    *   the judge's agreement rate;
    *   the short-query limit;
    *   the notice threshold;
    *   the encoder floor of 0.999;
    *   timeouts;
    *   the bootstrap count;
    *   seeds;
    *   data paths.

    All three reviewers raised this.
*   **Fix:** list every value with its name, type and starting value.

### Major 2: "the only normaliser" conflicts with `normalize_citation`

*   **Where:** L1246.
*   **Fix:** limit the claim to quote matching, and rename the other
    function `canonical_citation`.

### Major 3: phase numbering is inconsistent

*   **Where:** L1065–1070 and L1659.
*   **Problem:** some goals are assigned to more than one phase.
*   **Fix:** assign each goal to exactly one phase.

### Major 4: one trace cannot hold a "trigger rate", and "about half" cannot be tested

*   **Where:** L1587.
*   **Fix:**
    *   Store a true or false value in each trace.
    *   Report the rate in `nyaya-eval answers`.
    *   Add a config key set to 50%.

### Major 5: "only the label and the reason" reads as hiding the sentence

*   **Where:** L1651–1653.
*   **Problem:** this contradicts G1, which shows the sentence greyed out.
*   **Fix:** write that the sentence text is shown greyed out, with its
    passage labels and the reason.

### Major 6: the test "no lone letter A to H" will fail on correct text

*   **Where:** L1808–1810.
*   **Problem:** real text contains "Schedule C", "Part B" and "Form A".
    All three reviewers raised this.
*   **Fix:** use the pattern `(?<=[a-z,]) [A-H] (?=[a-z])` from
    `docling_test.py:33`, or compare against expected clean text for the
    four sample PDFs.

### Nits

*   **Names:** use the enum values instead of informal names.
*   **Zone enum:** list the values of the `Zone` enum.
*   **Trace field:** `timings` against `timings_ms`. Use one name.
*   **Offsets:** state what character offsets are measured from.
*   **Margin letters:** say "uppercase A to H", and name the band.
*   **Build failures:** "fails loudly" should name an exit code and a
    message.
*   **`load` command:**
    *   it needs a version argument;
    *   the paths to the vector files are unclear.
*   **`.gitignore`:** give the exact lines: `*.parquet` and
    `!tests/fixtures/**/*.parquet`.
*   **Terms:** define "passage" and "chunk" once.
*   **File numbering:** renumber item "49a".
*   **Safety-critical modules:** the list does not match the labels on
    items 11 and 37.
*   **Rating scale:** give anchors for the 1-to-3 scale.
*   **Notices:** merge the two notices bullets.
