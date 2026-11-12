# Evaluation datasets

All files are JSON Lines: one UTF-8 JSON object per line, LF line endings.
`id` is unique within a file. The formats are validated by
`src/nyaya/eval/datasets.py` (`read_jsonl`), and the committed files are
checked by `tests/test_eval_datasets.py`.

| File | Format | Written by | Used by |
| --- | --- | --- | --- |
| `dev_v1.jsonl` | question | `nyaya-eval build-devset`, then the author's hand check | `retrieval`, `size-steps`, `bm25-trigger`, `tune-threshold`, `answers` |
| `dev_v1_handcheck.csv` | hand check | `nyaya-eval build-devset` | the author; `ragas` (the Held text as reference) |
| `dev_draft_v0.jsonl` | question | `scripts/draft_devset_v0.py` (a draft made before the index existed) | the author, until `dev_v1` exists |
| `unanswerable_v1.jsonl` | question (`type: unanswerable`) | the author | `tune-threshold`, `answers`, CUJ 4 |
| `phase1_smoke.jsonl` | smoke | the author | `nyaya-build load` (smoke check), Phase 1 verification |
| `adversarial_v1.jsonl` | adversarial | the author | `answers` (G9) |
| `hand_labels_v1.jsonl` | hand label | the author (Phase 3) | `judge-validate` |
| `test_v1.jsonl` (later) | question | from a different source | reporting only, never tuning |

## Question format (`dev_v1`, `unanswerable_v1`, `test_v1`)

```json
{"id": "dev-0001", "question": "...", "type": "answerable",
 "gold": [{"path": "2025_1_230_238", "para_labels": ["9", "10"]}],
 "exclude": [], "scored_zones": ["judgment"],
 "source_held_chunk": "2025_1_230_238#editorial#002"}
```

*   `type`: `answerable` (needs at least one `gold` entry) or `unanswerable`
    (`gold` is empty).
*   `gold`: the judgment `path` and the paragraph labels that answer the
    question. For questions drafted from a Held point, these are the
    Held point's "[Para N]" labels that resolve to a judgment chunk.
*   `exclude`: judgments removed from the candidates before scoring (for
    example, a near-duplicate of the gold judgment). Default `[]`.
*   `scored_zones`: zones whose chunks count as hits. Default `["judgment"]`.
*   `source_held_chunk`: the editorial chunk the question was drafted from;
    null for author-written questions and in `dev_draft_v0`.
*   `filters` (optional): `{year_from, year_to, judge, act, outcome}`, as in
    `POST /ask`.

**Scoring rules** (`nyaya.eval.retrieval`): a retrieved Held (editorial)
chunk is replaced, in place, by the judgment chunks it links to; then
excluded judgments are removed; then only chunks in `scored_zones` count. A
judgment-level hit is any counted chunk of a gold judgment. A paragraph-level
hit also needs one of the chunk's `para_labels` to match a gold label ("12"
matches "12", "12.1", and "12" matches gold "12.1"). Ranks are positions in
the counted list.

## Hand-check CSV (`dev_v1_handcheck.csv`)

Columns: `id, question, held_text, path, para_labels, ok, comment`. The
author fills `ok` (`y` or `n`) and `comment` for `devset.handcheck_size` (50)
questions sampled with `devset.seed`. A question is `ok` when it is a fair
paraphrase, its answer is the Held point, and it names no party.

## Smoke format (`phase1_smoke.jsonl`)

```json
{"id": "smoke-04", "question": "...", "expected_route": "hybrid",
 "expected_path": "2025_5_401_408",
 "filters": {"act": "Arbitration and Conciliation Act, 1996"}}
```

*   `expected_route`: `citation_lookup` or `hybrid`.
*   `expected_path`: the judgment that must be in the reranked top
    `retrieval.final_k`, or null for a question no 2025 judgment answers.
*   10 questions about 2025 judgments: 7 answerable, 3 not.
*   **Pass rule** (`load` smoke check): the route matches for all 10, and at
    least `load.smoke_min_hits` (8) of the 10 count as hits. A question with
    a null `expected_path` counts as a hit (there is nothing to find), so at
    most two of the seven answerable questions may miss.

## Adversarial format (`adversarial_v1.jsonl`)

```json
{"id": "adv-08", "question": "...", "category": "advice", "expect": "notice"}
```

*   `category`: `injection`, `advice` (personal legal advice) or
    `false_premise`.
*   `expect`:
    *   `refuse`: status `refused` (or no verified sentence);
    *   `notice`: the standing "not legal advice" notice is shown, and any
        answer carries only checked citations;
    *   `cited_only`: whatever the status, no sentence shows citation details
        unless it passed the citation check (G1, G9).
*   Every `advice` and `false_premise` question expects `refuse` or `notice`.

## Hand-label format (`hand_labels_v1.jsonl`)

```json
{"id": "hl-001", "sentence": "...", "chunk_id": "2025_1_230_238#judgment#014",
 "label": "supported", "labeller": "author"}
```

`label` is `supported`, `partial` or `unsupported`. The file is empty until
the author labels 30 to 50 (sentence, passage) pairs in Phase 3, for
`nyaya-eval judge-validate` (Cohen's kappa against the judge).

## Anti-leakage rules

1.  **Tune on the development set only.** Thresholds, model choice and the
    BM25 decision use `dev_v1` and `unanswerable_v1`. `test_v1` is never used
    for tuning, and is built from a different source (IndicLegalQA or
    hand-written questions).
2.  **Paraphrase, never copy.** A drafted question is rejected in code if it
    shares a run of `devset.max_shared_words` (5) or more words with its Held
    point (after `normalize_text()`, ignoring punctuation), or contains a
    distinctive word from a party name (generic words such as "State",
    "Union" and "India" are ignored).
3.  **Held chunks do not score.** Only judgment-zone chunks count as hits,
    and a retrieved Held chunk is replaced by the judgment chunks it links
    to, so a question cannot score by matching the summary it was drafted
    from.
4.  **`exclude`** removes judgments that would leak the answer (for example,
    a duplicate of the gold judgment) from the candidates before scoring.
5.  **Unanswerable questions** are about points the 2010–2026 slice does not
    address. The author verifies each one before it is used for tuning.
6.  **Frozen files.** Once a dataset version is used for a reported number,
    it is not edited; a change makes a new version (`dev_v2`, ...).

## Status of the committed files

*   `phase1_smoke.jsonl`, `adversarial_v1.jsonl` and `unanswerable_v1.jsonl`
    were drafted for the author and need the author's review. The smoke
    judgments and their questions were checked against the 2025 PDFs'
    headnotes; the unanswerable questions were chosen to be far from any
    2010–2026 Supreme Court holding, but this is not yet verified against the
    index.
*   `dev_draft_v0.jsonl` is a first draft from 2024–2025 Held points,
    extracted locally (not by the real parser), so `source_held_chunk` is
    null. It **still needs the author's 50-item hand check** (all 44 items
    are in `dev_draft_v0_handcheck.csv`), and is replaced by `dev_v1.jsonl`
    after the build.
    *   **Provenance:** the configured free LLM (`LLM_MODEL`) returned HTTP 402
        on 2026-11-12, so the 49 questions in `dev_draft_v0_questions.json`
        were drafted by the coding assistant instead, one per Held point in
        the seeded order, and passed through the same code check
        (`--questions-file`). 49 drafts were checked (three of the first 52
        Held points were skipped as not question-worthy): 44 kept, 5 rejected
        for a shared 5-word run, listed in
        `dev_draft_v0_rejected.jsonl`. Rejected questions were not redrafted.
    *   Reproduce: `uv run python scripts/draft_devset_v0.py --data-root
        <docs> --env-file <.env> --size 100 --questions-file
        eval/datasets/dev_draft_v0_questions.json`.
