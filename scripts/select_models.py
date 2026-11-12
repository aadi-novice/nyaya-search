"""Model selection (design file 44): Qwen3-Embedding-0.6B against BAAI/bge-m3.

Each embedder is paired with Qwen3-Reranker-0.6B and run on `dev_v1` over the
2025 build outputs, in memory (exact cosine search; no database needed):

*   recall@50 of the embedder alone, and recall@5 after reranking its top
    `retrieval.rerank_candidates`, at judgment and paragraph level, with the
    scoring rules of nyaya.eval.retrieval;
*   encode throughput (chunks per second) on the local GPU;
*   peak GPU memory while encoding.

The results are written to docs/model_selection.md. The author chooses; the
choice is then written to config/settings.toml by hand.

    uv run python scripts/select_models.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from nyaya.config import load_settings
from nyaya.contracts import ChunkRecord, Filters
from nyaya.eval.datasets import DATASETS_DIR, DevItem, read_jsonl
from nyaya.eval.retrieval import Ranking, evaluate

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = {
    # model name -> query prefix ("settings" means embed.query_prefix)
    "Qwen/Qwen3-Embedding-0.6B": "settings",
    "BAAI/bge-m3": "",
}


@dataclass
class InMemoryEngine:
    """Just enough of SearchEngine for scoring: chunks and Held links."""

    chunks: dict[str, ChunkRecord]
    links: dict[str, list[str]] = field(default_factory=dict)

    def get_chunks(self, ids: Sequence[str]) -> list[ChunkRecord]:
        return [self.chunks[i] for i in ids if i in self.chunks]

    def links_from(self, chunk_ids: Sequence[str]) -> dict[str, list[str]]:
        return {c: self.links[c] for c in chunk_ids if c in self.links}

    def keyword(self, question, filters, k):  # pragma: no cover - not used
        raise NotImplementedError

    def vector(self, qvec, filters, k, exact):  # pragma: no cover - not used
        raise NotImplementedError


@dataclass
class VectorRetriever:
    """Exact cosine search over normalised document vectors, plus optional reranking."""

    ids: list[str]
    doc_vecs: np.ndarray  # (n, d), L2-normalised
    query_vecs: dict[str, np.ndarray]  # question -> (d,)
    engine: InMemoryEngine
    k: int = 50
    rerank_candidates: int = 100
    rerank: Any = None  # (question, [ChunkRecord]) -> [float]

    def _top(self, question: str, k: int) -> list[str]:
        scores = self.doc_vecs @ self.query_vecs[question]
        k = min(k, len(self.ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.ids[i] for i in idx]

    def run(self, method: str, question: str, filters: Filters | None) -> Ranking:
        if method == "exact":
            return Ranking(self._top(question, self.k))
        if method == "exact_rerank":
            cands = self.engine.get_chunks(self._top(question, self.rerank_candidates))
            scores = self.rerank(question, cands)
            order = sorted(range(len(cands)), key=lambda i: -scores[i])
            return Ranking([cands[i].chunk_id for i in order], float(scores[order[0]]))
        raise ValueError(method)


def load_build(build_dir: Path, year: int) -> InMemoryEngine:
    import pandas as pd

    chunks: dict[str, ChunkRecord] = {}
    with (build_dir / "chunks" / f"{year}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = ChunkRecord.model_validate(json.loads(line))
                chunks[c.chunk_id] = c
    links: dict[str, list[str]] = defaultdict(list)
    lp = build_dir / "links" / f"{year}.parquet"
    if lp.exists():
        df = pd.read_parquet(lp, columns=["from_chunk_id", "to_chunk_id"])
        for a, b in zip(df["from_chunk_id"], df["to_chunk_id"], strict=True):
            links[a].append(b)
    return InMemoryEngine(chunks, dict(links))


def encode(model_name: str, docs: list[str], queries: list[str], prefix: str, batch: int):
    """Encode on CUDA at fp16. Returns doc vectors, query vectors, chunks/s, peak MiB."""
    import torch
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "CUDA is required for the throughput measurement"
    model = SentenceTransformer(
        model_name, device="cuda", model_kwargs={"torch_dtype": torch.float16}
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dv = model.encode(docs, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    qv = model.encode([prefix + q for q in queries], batch_size=batch, normalize_embeddings=True)
    del model
    torch.cuda.empty_cache()
    return (
        np.asarray(dv, dtype=np.float32),
        np.asarray(qv, dtype=np.float32),
        len(docs) / seconds,
        peak_mib,
    )


def render_report(results: dict[str, dict[str, Any]], *, dataset: str, year: int, n: int) -> str:
    lines = [
        "# Model selection",
        "",
        f"Written by `scripts/select_models.py` on {date.today().isoformat()}: `{dataset}` "
        f"({n} answerable questions) over the {year} build, exact cosine search, "
        "each embedder followed by Qwen3-Reranker-0.6B.",
        "",
        "| Embedder | recall@50 (judgment) | recall@50 (paragraph) | recall@5 reranked (judgment) "
        "| recall@5 reranked (paragraph) | MRR reranked (judgment) | chunks/s | peak GPU MiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        s = r["summary"]
        lines.append(
            f"| {name} | {s['exact']['judgment']['recall@50']:.3f} "
            f"| {s['exact']['paragraph']['recall@50']:.3f} "
            f"| {s['exact_rerank']['judgment']['recall@5']:.3f} "
            f"| {s['exact_rerank']['paragraph']['recall@5']:.3f} "
            f"| {s['exact_rerank']['judgment']['mrr']:.3f} "
            f"| {r['chunks_per_s']:.1f} | {r['peak_mib']:.0f} |"
        )
    lines += [
        "",
        "Recall@50 is the embedder alone; the reranked columns rerank its top "
        "`retrieval.rerank_candidates` chunks. Only judgment-zone chunks count, and a "
        "retrieved Held chunk is replaced by the judgment chunks it links to.",
        "",
        "**Choice:** (the author fills this in, then sets `embed.model` in config).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compare embedders for Nyaya Search")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--dataset", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "model_selection.md")
    args = ap.parse_args(argv)

    from nyaya.models.encoders import Reranker

    from nyaya.eval.wiring import rerank_fn

    settings = load_settings()
    engine = load_build(REPO / settings.paths.build, args.year)
    items = [
        i
        for i in read_jsonl(args.dataset, DevItem)
        if i.type == "answerable" and all(p.startswith(f"{args.year}_") for p in i.gold_paths)
    ]
    if not items:
        print(f"no answerable {args.year} questions in {args.dataset}", file=sys.stderr)
        return 2
    ids = list(engine.chunks)
    docs = [f"{engine.chunks[i].header}\n\n{engine.chunks[i].text}" for i in ids]
    questions = [i.question for i in items]
    rerank = rerank_fn(Reranker(settings))

    results: dict[str, dict[str, Any]] = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        prefix = CANDIDATES.get(name, "")
        if prefix == "settings":
            prefix = settings.embed.query_prefix
        dv, qv, rate, peak = encode(name, docs, questions, prefix, args.batch)
        retriever = VectorRetriever(
            ids=ids,
            doc_vecs=dv,
            query_vecs=dict(zip(questions, qv, strict=True)),
            engine=engine,
            k=50,
            rerank_candidates=settings.retrieval.rerank_candidates,
            rerank=rerank,
        )
        report = evaluate(items, retriever, engine, ["exact", "exact_rerank"])
        results[name] = {"summary": report.summary(), "chunks_per_s": rate, "peak_mib": peak}
        print(name, json.dumps(results[name]["summary"]["exact_rerank"]["judgment"]))

    args.out.write_text(
        render_report(results, dataset=args.dataset.name, year=args.year, n=len(items)),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
