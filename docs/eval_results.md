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

## Answers run (G1, G2, G5, G8, G9)

Run `answers-20261117T160221Z-1038ae` (answer model `openai/gpt-oss-120b`,
judge `qwen/qwen3.8-27b`, both on Groq's free tier), **all 104 questions**
(44 dev, 40 unanswerable, 20 adversarial), over two sessions
(2026-11-17 and 2026-11-18). The first session stopped at 88 questions when
Groq's daily token limit ran out; no rate-limited question was recorded, so
none counts as a refusal. The run used about 241k answer-model and 105k judge
tokens.

Outcomes: dev 35 answered, 8 refused, 1 `passages` (Groq HTTP 400 "Failed to
generate JSON"); unanswerable 40 refused; adversarial 5 answered, 15 refused.

| Gate | Target | Measured | Status |
| --- | --- | --- | --- |
| G1 no citation details on unchecked sentences | 0 | **0 violations**; 186 LLM sentences, 113 verified, 73 unverified (39%); 75 rule breaks before the check (56 quote not found, 18 quote length, 1 uncited case name) | pass |
| G2 supported claims | ≥ 90% supported | **82 / 113 = 72.6%** (31 `partial`, counted as unsupported); dev 75.8%, adversarial 50%; judge not yet validated (needs hand labels, `judge-validate`) | fail (unvalidated) |
| G5 unanswerable refused | ≥ 80% | **40 / 40 = 100%** | pass |
| G5 answerable refused | ≤ 10% | **8 / 43 = 18.6%** (4 below threshold, 4 with no verified sentence) | fail |
| G8 wrong sentences labelled | every one | 31 wrong: **0 hallucination**, 11 retrieval error, 20 reasoning error | done |
| G9 hostile input | all pass | **18 / 20**: injection 7 / 7, advice 6 / 7, false premise 5 / 6. Failures: adv-13 (advice) and adv-19 (false premise) were answered instead of refused | fail |

### Reading these numbers

*   **What holds:** no unchecked quote or citation reached the user (G1), no
    sentence was a hallucination (G8), every unanswerable question was refused
    and every injection attempt failed.
*   **Where it falls short:** the answer model paraphrases the quotes it is
    told to copy, so 39% of its sentences fail the exact check and are hidden.
    That also drives half of the answerable refusals (G5). Retrieval misses
    (G3) account for the other errors. G2 is an unvalidated judge's view and
    counts `partial` as unsupported.
*   **Next fixes:** stricter quote instructions, or a model that copies
    exactly; retrieval recall; and refusal rules for advice and false-premise
    questions that slip past the threshold.

### Quote failures, diagnosed offline (2026-11-18)

No LLM calls; `nyaya-eval replay-check` re-runs the citation check on the
saved records. **Caveat:** this run's records kept the unverified sentences
and their labels, but not the failing quotes. The diagnosis therefore uses the
sentence written with each quote, or a quotation inside it, as a stand-in;
every number below is an estimate. From 2026-11-18 each `quote_not_found` and
`quote_length` rule break keeps the quote, so the next run can be replayed
exactly.

What the saved quotes do show: none of the 116 verified quotes contains
U+2011 (non-breaking hyphen), 3 contain `-`; the model writes U+2011 in 24
of its sentences (17 of 73 unverified, 7 of 113 verified). NFKC turns U+2011
into U+2010, which `normalize_text()` did not map, so any quote with a
hyphenated word failed.

Similarity is the word-level ratio between the stand-in and its best window
in the cited passage (and in the other passages). The 74 failing citations,
by the closest category:

| Class | `quote_not_found` | `quote_length` |
| --- | --- | --- |
| normalisation gap (U+2011 hyphen) | 4 | – |
| near-verbatim, small edits (ratio ≥ 0.90; 2 of 3 cut by a footnote inside the source sentence) | 3 | – |
| stitched with "..." | 6 | – |
| close paraphrase, partial copy (0.75–0.90) | 4 | – |
| paraphrase (< 0.75) | 36 | – |
| words from another passage | 1 | – |
| side citation of a verified sentence (no stand-in) | 2 | – |
| sentence over 40 words (quote probably too long) | – | 6 |
| length direction unknown | – | 12 |

Median ratio to the cited passage 0.54 (`quote_not_found`), to the best
other passage 0.39; 13 of 54 stand-ins sit closer to another passage. Most
failures are paraphrase, which no matching rule can safely rescue; that is
the prompt's job (`prompts/answer_v2.md`, not yet measured).

**Replay** (stand-in quotes; snapping at 0.90, at most 6 changed words):

| | Recorded | Normalisation fix | + snapping |
| --- | --- | --- | --- |
| verified / unverified sentences | 113 / 73 | 117 / 69 | 120 / 66 |
| G1 unverified share | 39.2% | 37.1% | 35.5% |
| G5 answerable refused | 8 / 43 = 18.6% | 8 / 43 | 7 / 43 = 16.3% (dev-0019 answered) |
| G9 must-refuse answered | adv-13, adv-19 | unchanged | unchanged |

Snapping stays off by default (design, decision 2026-11-18): the gain is
estimated from stand-ins, and is small.

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
*   **G7: pass, measured on 2026-11-18 from 29 of 32 requests.**
    *   **Result:** p50 6.7 s, p95 9.4 s, max 11.3 s, against a 15 s p95
        target. The 29 requests were 2 warm-ups and 27 of the 30 timed
        questions: 24 answered, 4 refused and 1 `passages` (Groq HTTP 400).
    *   **Stages:** median rerank 3.6 s, median LLM 2.9 s, retrieval about
        0.5 s.
    *   **How it was measured:** run with `nyaya-eval latency` on the
        first 30 dev questions, one at a time and warm. The run stopped at
        dev-0029 when Groq's daily token limit for the answer model ran out
        (200k tokens). Because of that, the numbers were recomputed from
        the request traces (`timings_ms.total`) of that run's
        non-rate-limited requests, instead of from the command's own
        summary.
    *   **Discarded first attempt:** `latency-20261118T143716Z-5e26fa`
        was invalid. A network outage turned 26 requests into
        `APIConnectionError` → `passages`, and the tool timed them as
        samples. The tool now retries connection and timeout errors like
        rate limits.
*   **G10:** the index was built with one command
    (`nyaya-build run --years 2010-2026`); a clean rebuild has not been
    repeated to confirm it.

## RAGAS (secondary, never a gate)

Run `ragas-answers-20261117T160221Z-1038ae`, finished on 2026-11-19, scored
the 44 development questions of the answers run above. It uses ragas 0.4.3
and `all-MiniLM-L6-v2` on the CPU for answer relevancy.

| Metric | Score | Over |
| --- | --- | --- |
| Faithfulness | 0.887 | 35 answered questions |
| Answer relevancy | 0.641 | 35 answered questions |
| Context precision (with reference) | 0.800 | 44 questions |
| Context recall | 0.760 | 42 questions; 2 judge replies could not be parsed |

The 8 refused questions, and the 1 with no verified sentence, are left out
of faithfulness and answer relevancy.

**Judge hosts.** The judge model is `qwen3.8-27b` throughout. A question
costs about 25k judge tokens, because faithfulness, context precision and
context recall each send all 8 retrieved passages. That is far more than
one free daily quota, so the work was spread across free tiers:

| Host | Questions | Reasoning setting |
| --- | --- | --- |
| Groq | 29 | provider default |
| AXForge | 11 | provider default (no reasoning tokens observed) |
| Cloudflare Workers AI | 4 | `low` (its default is `xhigh`) |

Each per-question record carries `judge_host`. Hosts may serve the same
weights with different defaults, so treat small differences between
questions with caution. Our exact quote check remains the primary metric.

**Network outage.** On 2026-11-19 an outage was first recorded as 32
finished questions with every metric in error. Those records were removed
(a backup was kept), and `nyaya-eval ragas` now stops on connection and
timeout errors instead (PR #9).

## Pending

*   A complete `nyaya-eval latency` run (G7) on a day with fresh quota.
*   `size-steps` (G4), about an hour on the GPU.
*   `judge-validate` needs the author's hand labels; G6 needs the hand check
    ([guide](hand_checks.md)).
*   A final test set, so that headline numbers are not development numbers.
*   A live answers run on the 44 dev questions with `answer_v2` (and one with
    `quote.snap = true`), then `replay-check` on it with the saved quotes.

