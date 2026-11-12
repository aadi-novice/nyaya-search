# Research Notes: Papers and Datasets for Legal RAG

* **Author:** aadi-novice
* **Date:** 2026-11-12
* **Method:** a research agent read the papers below through arXiv HTML pages.
  The reading tool returned summaries rather than raw text, so check any
  figure against the paper before citing it. Items marked "not verified" were
  seen only in search snippets.

## Verdicts

| Paper | Verdict | What to take |
| --- | --- | --- |
| [CRAwLeR: Cross-Reference Aware Legal Retrieval](https://arxiv.org/abs/2606.21676) (Jun 2026) | Skip the method, keep one lesson | Store resolved references as links ("[Para 29]" → judgment paragraph; citation string → case ID), and pull the linked chunk at answer time instead of hoping search finds it. Their method needs an LLM-written prefix for every chunk, which would be about 730,000 calls. |
| [Dhārā](https://arxiv.org/abs/2611.11197) (Nov 2026) | Adopt the fuzzy quote path now, LLM-free. Entity Notes later. | See "Fuzzy quote lookup" below. Entity Notes need LLM calls per chapter and per entity. A free partial version groups chunks by "List of Acts" and "Case Law Cited". |
| [Dhārā Bench](https://arxiv.org/abs/2611.03196) (Nov 2026, abstract only) | Later: a template for our test set | Records alternative sets of passages that each fully support an answer, and measures recall of the *complete* evidence. |
| [IL-PCSR](https://arxiv.org/abs/2511.00268) (EMNLP 2025) | Adopt the finding now | On Indian precedent retrieval, BM25 with 5-grams reached F1 33.3, against 9.9 for an untrained legal dense model. Keyword search must stay strong in hybrid search. |
| [Rhetorical-Role-Aware RAG, Indian SC](https://arxiv.org/abs/2608.06828) (Aug 2026) | Skip as evidence | 30 judgments and 25 queries, with no comparison against paragraph chunking. Filtering by role is a later idea. |
| [LegalSeg](https://arxiv.org/abs/2502.05836) (NAACL Findings 2025) | Later | 7,120 Indian judgments, about 1.49M sentences, 7 rhetorical roles. Best F1 0.77. Needs a trained classifier. |
| [Legal RAG Bench](https://arxiv.org/abs/2603.01710) (Mar 2026) | Adopt now | Sort each wrong answer into one of three causes: **hallucination**, **retrieval error** or **reasoning error**. In their tests the embedder mattered more than the LLM. The authors sell the winning embedder. |
| [Chunking German Legal Code](https://arxiv.org/abs/2605.19806) (May 2026) | Adopt the finding now | Chunks that follow sections and subsections beat LLM-contextual, RAPTOR, semantic, LLM-guided and fixed-size chunking. The tests used statutes, not judgments. |
| [DcLBC](https://arxiv.org/abs/2611.12097) (Nov 2026) | Later | Targets confusion between chunks of the same document. Needs encoder training and late chunking. First measure the gap between judgment-level and passage-level recall. |
| [Lightfoot](https://arxiv.org/abs/2611.11010) (Nov 2026) | Skip the method | One LLM call per passage. Build the citation graph for free from "Case Law Cited" and citation patterns instead. |
| [Decisions by Discrimination](https://aclanthology.org/2026.aacl-long.250/) (AACL 2026) | Later | Retrieve evidence for each competing answer separately. Possibly useful for contested legal questions. |

## Top three ideas for v1

1. **Structure-first chunks, a free context header and resolved links.**
   - Put the case name, year, judge or opinion and paragraph numbers in
     front of each chunk as plain text. No LLM writes this.
   - Resolve "[Para N]" and citation strings to chunk IDs and case IDs.
   - At answer time, also pull in the chunks those links point to.
2. **Measure retrieval separately, and label every wrong answer** as a
   hallucination, a retrieval error or a reasoning error. Keep BM25 strong,
   and consider phrase or n-gram matching.
3. **Fuzzy quote lookup**, adapted from Dhārā's Fuzzy Verse Retrieval without
   the LLM:
   1. find quoted or fragment-like queries with a regex and a short-query
      rule;
   2. normalise the text;
   3. look up exact substrings, then trigram similarity, over judgment
      sentences (PostgreSQL `pg_trgm` or an in-process index);
   4. map the hits to chunks with RapidFuzz;
   5. rank those chunks first.

   Lawyers often remember a phrase from a famous holding, such as "basic
   structure".

## Datasets we can reuse

| Dataset | Contents | Licence | Use |
| --- | --- | --- | --- |
| [IndicLegalQA](https://data.mendeley.com/datasets/gf8n8cnmvc/2) | 10,000 question–answer pairs from 1,256 Supreme Court judgments, built with LLM help and checked by experts | CC BY 4.0 | The best source of free test labels. **Not verified:** the files list looked empty, so download is unconfirmed. |
| [IL-PCSR](https://huggingface.co/datasets/Exploration-Lab/IL-PCSR) | Supreme Court precedent and statute relevance labels | CC BY-NC-SA 4.0, gated, no redistribution | Private evaluation only |
| [LegalSeg](https://github.com/ShubhamKumarNigam/LegalSeg) | Rhetorical-role labels | Repository is Apache-2.0; **the data licence is not verified** | Test labels for a later role tagger |
| [IL-TUR](https://exploration-lab.github.io/IL-TUR) | Rhetorical roles, IL-PCR, ILDC (about 34k Supreme Court documents) | CC BY-NC-SA | Benchmarks. Has no question-answering task. |

## Added 2026-11-12: Adaptive Chunking (arXiv 2603.25333)

[Adaptive Chunking: Optimizing Chunking-Method Selection for RAG](https://arxiv.org/abs/2603.25333) (de Moura Júnior, Lelong and Blangero, Ekimetrics, March 2026). Read in full from the PDF text. Code: github.com/ekimetrics/adaptive-chunking. The paper is CC BY 4.0.

**What it does:**

- Runs several chunkers on each document. Each result is scored with five *intrinsic* metrics, which judge the chunks themselves rather than RAG answers:
  - **References Completeness (RC):** pronoun–entity pairs not split across chunks. It needs the Maverick coreference model and works on English only.
  - **Intrachunk Cohesion (ICC):** how similar each sentence is to its whole chunk, using embeddings.
  - **Document Contextual Coherence (DCC):** how similar each chunk is to a window of about 3,000 tokens around it, using embeddings.
  - **Block Integrity (BI):** the share of the parser's blocks (paragraphs, tables) not cut by a chunk boundary.
  - **Size Compliance (SC):** the share of chunks between 100 and 1,100 tokens.
- For each document it keeps the chunker with the best average score.
- **It adds two chunkers:**
  1. An **LLM-regex splitter.** GPT-5 reads the first 8,000 tokens and writes one regex, which then splits the whole document. That is one LLM call per document.
  2. A **split-then-merge recursive splitter.** It splits down to a target size, then greedily merges neighbouring pieces back up to that size.
- **Post-processing:** chunks over 1,100 tokens are split again, and chunks under 100 tokens are merged with a neighbour.

**Results (33 documents, 99 questions):**

- Retrieval completeness was 67.7%, against 58.1% (LangChain recursive) and 59.1% (one chunk per page).
- 65 of 99 questions were answered, against 49 of 99.
- Answer correctness was 78.0% in the body table, against 70.1% and 73.3%. The abstract gives "72% (from 62–64%)" instead, and the paper does not explain the difference.
- Size post-processing alone added 6 to 16 points to the intrinsic scores.

**Weak points:**

- The test set is small: 33 documents and 99 questions, all written by GPT-4.1. GPT-4.1 also generated the answers and judged them.
- Adaptive selection took about 211 seconds per document. At 38,397 judgments that is about 93 days of single-process time.
- The model it calls a "reranker" (snowflake-arctic-embed-l-v2.0) is an embedding model.

**Fit for this project:**

- **Adopt now:**
  - Size limits enforced as a separate pass (split oversized chunks, merge tiny ones). This matches Stage 4 of our chunker.
  - **Block Integrity and Size Compliance as the missing gate for the parse-and-chunk layer.** Both are cheap and need no model. Our numbered paragraphs are the gold blocks, so BI means "share of judgment paragraphs not cut by a chunk boundary".
- **Later:** ICC and DCC on a sample of about 200 judgments per era, to compare chunker settings. They need embedding runs, which are fine on Kaggle.
- **Skip:**
  - Per-document adaptive selection. Our per-era rules give the same benefit at no cost, and running every chunker on every judgment is too slow.
  - The LLM-regex splitter per document, because our four eras are few enough to write the rules by hand.
  - RC, because the coreference model is too slow for 730k chunks.
