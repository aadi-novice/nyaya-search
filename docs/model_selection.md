# Model selection

Written by `scripts/select_models.py` on 2026-11-13: `dev_draft_v0.jsonl` (22 answerable questions) over the 2025 build, exact cosine search, each embedder followed by Qwen3-Reranker-0.6B.

| Embedder | recall@50 (judgment) | recall@50 (paragraph) | recall@5 reranked (judgment) | recall@5 reranked (paragraph) | MRR reranked (judgment) | chunks/s | peak GPU MiB | encoded on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-Embedding-0.6B | 1.000 | 1.000 | 0.955 | 0.955 | 0.854 | 16.2 | 4393 | Kaggle Tesla T4 |
| BAAI/bge-m3 | 0.955 | 0.955 | 0.909 | 0.909 | 0.863 | 46.4 | 1833 | Kaggle Tesla T4 |

Recall@50 is the embedder alone; the reranked columns rerank its top `retrieval.rerank_candidates` chunks. Only judgment-zone chunks count, and a retrieved Held chunk is replaced by the judgment chunks it links to.

**Choice:** (the author fills this in, then sets `embed.model` in config).
