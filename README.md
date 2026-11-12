# Nyaya Search

Nyaya Search is a research assistant for lawyers that works over Indian Supreme Court judgments. Every sentence in an answer cites a retrieved passage and quotes it word for word, and code checks that each quote matches the source. If no passage supports an answer, the system says that no relevant judgment was found. The stack is free to run: PostgreSQL with pgvector, local embedding (Qwen3-Embedding-0.6B) and reranking (Qwen3-Reranker-0.6B), and a free hosted LLM. The one paid part is an optional fallback reranker (Cohere via OpenRouter), used only if the local reranker fails.

## Status

Design revised after external review. No application code yet. Two decisions remain open (see the design's Open Questions).

- Design: [docs/designs/nyaya_search.md](docs/designs/nyaya_search.md)
- Review summary: [docs/reviews/consolidated_review_report.md](docs/reviews/consolidated_review_report.md)

## Data

The corpus is the AWS Open Data dataset "Indian Supreme Court Judgments" (`s3://indian-supreme-court-judgments`, CC BY 4.0, https://registry.opendata.aws/indian-supreme-court-judgments/).

Download the English PDFs and metadata (about 23 GB) with:

```
python scripts/download_sc.py --dest docs --skip-regional
```

The data is not in this repository.

## Repository layout

- `scripts/`
  - `download_sc.py`: downloads judgments and metadata from the dataset.
  - `data_check.py`: runs checks on the downloaded corpus.
  - `layout_study.py`: surveys PDF page layouts.
  - `docling_test.py`: compares PDF parsers.
- `docs/`
  - `data_check_report.md`, `document_anatomy.md`, `docling_findings.md`, `research_notes.md`
  - `designs/`: system design.
  - `reviews/`: design review reports.
- `concepts_brief.md`: initial concept brief.

## License

Code and documentation are licensed under Apache-2.0 (see [LICENSE](LICENSE)). The dataset is licensed CC BY 4.0 by its publishers.
