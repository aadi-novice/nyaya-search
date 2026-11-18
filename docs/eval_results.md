# Nyaya Search v1: evaluation results

Measured 2026-11-17 on index version `p1-2010-2026a` (13,007 judgments decided
2010–2026, 296,151 chunks; clean text `ct2`, chunker `ch4`), on one machine
(RTX 2060 6 GB, Postgres 17 + pgvector in Docker). Every run named below is a
JSON file under `eval/runs/` (git-ignored) and a row in the `eval_runs` table.

The gates are defined in [the design](designs/nyaya_search.md#goals). Each was
measured once on the final setup; nothing below was tuned on the reported
numbers except the refusal threshold, which the design's own procedure tunes on
the development questions (see the G5 caveat).

## Question sets

| Set | File | Size | Notes |
| --- | --- | --- | --- |
| Development (answerable) | `eval/datasets/dev_draft_v0.jsonl` | 44 | Drafted from headnote ("Held") points before the index existed; **not yet hand-checked** (`dev_v1` does not exist). Gold = judgment + paragraph labels. No question uses filters. |
| Unanswerable | `eval/datasets/unanswerable_v1.jsonl` | 40 | Written by the author. Also used to tune the threshold. |
| Adversarial | `eval/datasets/adversarial_v1.jsonl` | 20 | 7 injection, 7 advice, 6 false premise. |
| Final test set | — | 0 | Not built yet; every number here is a development number. |

With 44 questions, one question is 2.3 points, and a 95% confidence interval on
a share near 0.75 is about ±13 points wide.

## Retrieval (G3), before and after the keyword fix

Judgment level, 44 answerable development questions, recall@5 / recall@50 /
MRR. Only judgment-zone chunks count as hits (design G3). Runs
`retrieval-20261113T034558Z-c341d4` (before) and
`retrieval-20261117T152739Z-1edded` (after); the middle row is
`eval/runs/fts-hybrid-comparison-20261117.json`, the same scoring code with
the keyword leg switched back to Postgres.

| Method | Before (Postgres `ts_rank_cd`) | Postgres `ts_rank` (fixed ranking) | After (BM25, production) |
| --- | --- | --- | --- |
| Keyword only | 0.09 / 0.20 / 0.09 | 0.48 / 0.75 / 0.40 | 0.52 / 0.80 / 0.49 |
| Vector, exact | 0.68 / 0.84 / 0.55 | (unchanged) | 0.68 / 0.84 / 0.55 |
| Vector, HNSW | 0.68 / 0.84 / 0.55 | (unchanged) | 0.68 / 0.84 / 0.55 |
| Hybrid (RRF k=60 + link expansion) | 0.48 / 0.77 / 0.29 | 0.61 / 0.80 / 0.52 | **0.64 / 0.84 / 0.57** |
| Hybrid + Qwen3 rerank (top 50) | 0.70 / 0.77 / 0.52 | 0.70 / 0.80 / 0.55 | **0.73 / 0.84 / 0.56** |

Paragraph level after the fix: hybrid 0.57 / 0.82 / 0.51, hybrid + rerank
0.64 / 0.82 / 0.44.

Keyword time per question (warm, one thread): Postgres `ts_rank` 3.3 s,
BM25 0.04 s.

**G3 targets:** right judgment in the top 50 before reranking for ≥ 90% of
questions: **0.84, fail**. In the top 5 after reranking for ≥ 75%: **0.73,
fail** (one question short of 0.75; the 95% interval easily covers both
sides). Hybrid search now matches vector search at recall@50 instead of
dragging it down (0.84 against 0.77 before); 7 of 44 questions still
miss the top 50.

### Why the keyword leg was weak, and the fix

*   The question words were joined with OR. For dev-0001 that query matched
    282,370 of 296,151 chunks, through words such as "court", "order" and
    "state".
*   `ts_rank_cd` scores by the density of any matched word and has no
    inverse document frequency, so long chunks repeating common legal words
    ranked first. Switching to `ts_rank` with length normalisation 1 (which
    caps each word's contribution) took keyword recall@50 from 0.20 to 0.75;
    dropping words found in more than 5% or 20% of chunks did not help
    further (0.68, 0.73). Postgres has no per-word IDF, and each OR query
    still ranks about 250,000 rows (3.3 s).
*   The BM25 trigger (design, "Hybrid retrieval") had fired: run
    `bm25-trigger-20261113T035246Z-3136e7`, gap 59 points (95% CI 45–73).
    ParadeDB `pg_search` would mean replacing the database image under the
    live data volume, so the keyword leg became `bm25s` in the API process
    (the BM25 library of the design's Alternative M), with Postgres still the
    store of record and the place where filters and quoted phrases are
    applied.
*   After the `ts_rank` fix the trigger no longer fires on 44 questions (run
    `bm25-trigger-20261117T153308Z-6eeca2`: gap 4.5 points, 95% CI −6.8 to
    15.9). BM25 was kept: it was adopted when the trigger fired on the leg as
    built, it is ahead on every row above, and it is about 80 times faster.
    The design records the decision with these numbers.
*   Stemming (Snowball) was tried and left off: recall@50 0.80 → 0.82 but MRR
    0.49 → 0.42.

## Refusal threshold (G5 tuning)

Run `tune-threshold-20261117T160129Z-d1848a` (44 answerable + 40 unanswerable,
Qwen reranker, after the BM25 fix) wrote `threshold_qwen` = 0.99691 to the
active index row through `nyaya-eval tune-threshold` (was the 0.35 seed). On
those questions it refuses 95% of unanswerable and 9.1% of answerable ones.
The probability scale was kept: P("yes") is a strictly increasing function of
the yes-minus-no logit (0.99691 is a logit margin of 5.78), so thresholding
the logit would refuse exactly the same questions. `threshold_cohere` is
still the 0.20 seed (paid fallback, not tuned). Caveat: the same 40
unanswerable questions are used for tuning and for G5 below.

## Answers run (G1, G2, G5, G8, G9): partial

Run `answers-20261117T160221Z-1038ae` (answer model `openai/gpt-oss-120b`,
judge `qwen/qwen3.8-27b`, both on Groq's free tier). **88 of 104 questions
finished**: all 44 dev, all 40 unanswerable, 4 of 20 adversarial. The session
stopped when the answer model stayed rate-limited after every retry (Groq's
daily token limit: 206k tokens used); no rate-limited question was recorded,
so none counts as a refusal. Numbers below are from the partial summary.

| Gate | Target | Measured (partial) | Status |
| --- | --- | --- | --- |
| G1 no citation details on unchecked sentences | 0 | 0 violations; 162 LLM sentences, 99 verified, 63 unverified (39%); 65 rule breaks before the check (47 quote not found, 17 quote length, 1 uncited case name) | pass |
| G2 supported claims | ≥ 90% supported | 75 / 99 = 76% (24 `partial`, which counts as unsupported); judge not yet validated (needs hand labels, `judge-validate`) | fail (unvalidated) |
| G5 unanswerable refused | ≥ 80% | 40 / 40 = 100% | pass |
| G5 answerable refused | ≤ 10% | 8 / 43 = 19% (4 below threshold, 4 with no verified sentence); 1 `passages` (Groq HTTP 400 "Failed to generate JSON") | fail |
| G8 wrong sentences labelled | every one | 24 wrong: 0 hallucination, 11 retrieval error, 13 reasoning error | done |
| G9 hostile input | all pass | 4 / 4 so far (injection); 16 adversarial questions pending | pending |

## Other gates

*   **G3:** fail (0.84 and 0.73 against 0.90 and 0.75; see above).
*   **G4:** not measured. `size-steps` needs no re-chunking or re-embedding
    (it copies rows into temporary schemas), but it was stopped at step 2 to
    free the machine; the temporary schema was dropped.
*   **G6:** automated checks pass on the 2010–2026 build
    (`build/chunk_metrics.json`, 12,796 judgments with PDFs): paragraph
    integrity 100%, size compliance 99.97%, 0 chunks over the model limit.
    The 50-PDF hand check is pending with the author: the sample in
    `eval/runs/g6_holdout` (seed 4242; 41 Era C, 9 Era D) was regenerated on
    2026-11-17 and is identical to the current parser's output; its two
    automatic margin-letter hints (2025_2_1625_1702) are both "Ch D"
    (Chancery Division) in case citations, not furniture.
*   **G7 latency, RAGAS, README table:** pending (see below).

## Pending

*   Finish the answers run (16 adversarial questions), then RAGAS and
    latency, then fill the README results table.
*   `judge-validate` needs the author's hand labels; G6 needs the hand check.

