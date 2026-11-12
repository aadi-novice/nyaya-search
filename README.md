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

## Running the API and page

Set up the environment once with `uv sync --extra dev --extra eval`, copy `.env.example` to `.env` and fill in `DATABASE_URL` and the LLM keys. The API needs an active index (see the build steps in the design).

```
uv run nyaya-api              # serves http://127.0.0.1:8000
uv run nyaya-api --port 8080
```

The server binds 127.0.0.1 only. It loads the settings and models once at startup.

- `POST /ask` takes `{"question": "...", "filters": {"year_from", "year_to", "judge", "act", "outcome"}}` (filters optional) and returns an `AskResponse` (see `contracts/schema.json`). Each request has a hard limit of `timeouts.request_s` (30) seconds; past it the status is `error`.
- `GET /judgment/{path}?highlight={chunk_id}` returns the judgment record plus the highlighted chunk's `char_start` and `char_end`. Opened in a browser (from a citation link), the same URL shows the judgment text with the passage marked.
- `GET /health` returns the active index version, whether the models are loaded, whether the fallback reranker is configured, and the standing notices.
- `/` serves the page in `web/` (plain HTML, CSS and JavaScript; no build step).

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
