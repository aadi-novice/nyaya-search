"""Model selection (design file 44): Qwen3-Embedding-0.6B against BAAI/bge-m3.

Each embedder is paired with Qwen3-Reranker-0.6B and run on `dev_v1` over the
2025 build outputs, in memory (exact cosine search; no database needed):

*   recall@50 of the embedder alone, and recall@5 after reranking its top
    `retrieval.rerank_candidates`, at judgment and paragraph level, with the
    scoring rules of nyaya.eval.retrieval;
*   encode throughput (chunks per second) on the local GPU;
*   peak GPU memory while encoding.

Document vectors can instead come from a Kaggle run (kaggle/embed_corpus.py):
`--vectors MODEL=DIR` reads DIR/vectors/<year>.npy (+ .ids.txt), checks DIR's
reference vectors against the local embedder (as startup_check does), and
takes throughput and memory from DIR/embed_report.json. Queries are always
embedded locally, with the model's query prefix.

The results are written to docs/model_selection.md. The author chooses; the
choice is then written to config/settings.toml by hand.

    uv run python scripts/select_models.py --year 2025
    uv run python scripts/select_models.py --year 2025 \\
        --vectors Qwen/Qwen3-Embedding-0.6B=kaggle_out/Qwen__Qwen3-Embedding-0.6B \\
        --vectors BAAI/bge-m3=kaggle_out/BAAI__bge-m3
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


def encode(settings, model_name: str, docs: list[str], queries: list[str], batch: int):
    """Encode on CUDA at fp16 with nyaya's Embedder (the model's padding side and
    query prefix). Returns doc vectors, query vectors, chunks/s, peak MiB."""
    import torch

    from nyaya.models.encoders import Embedder

    assert torch.cuda.is_available(), "CUDA is required for the throughput measurement"
    base = torch.cuda.memory_allocated()  # the reranker, already loaded, is not counted
    embedder = Embedder(settings, model_name=model_name)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dv = embedder.embed_documents(docs, batch_size=batch)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
    qv = np.stack([embedder.embed_query(q) for q in queries])
    del embedder
    torch.cuda.empty_cache()
    return dv, qv, len(docs) / seconds, peak_mib


def parse_vectors_args(values: Sequence[str]) -> dict[str, Path]:
    """`MODEL=DIR` pairs -> {model: DIR}."""
    out: dict[str, Path] = {}
    for v in values:
        model, sep, path = v.partition("=")
        if not sep or not model.strip() or not path.strip():
            raise SystemExit(f"--vectors takes MODEL=DIR, got {v!r}")
        out[model.strip()] = Path(path.strip())
    return out


def kaggle_doc_vectors(vdir: Path, year: int, ids: Sequence[str], dim: int) -> np.ndarray:
    """DIR/vectors/<year>.npy rows in `ids` order; the IDs must be the same set."""
    from nyaya.store.loader import check_ids, read_vectors

    vids, vecs = read_vectors(vdir, dim, [year])
    if not vids:
        raise SystemExit(f"{vdir}: no vectors/{year}.npy")
    failures = check_ids(ids, vids)
    if failures:
        raise SystemExit(f"{vdir}: vectors do not match the {year} chunks: " + "; ".join(failures))
    row = {cid: i for i, cid in enumerate(vids)}
    return vecs[[row[c] for c in ids]]


def kaggle_run_stats(vdir: Path) -> dict[str, Any]:
    """chunks/s, peak MiB and device from DIR/embed_report.json (n/a if absent)."""
    path = vdir / "embed_report.json"
    if not path.exists():
        return {"chunks_per_s": None, "peak_mib": None, "device": "Kaggle"}
    rep = json.loads(path.read_text(encoding="utf-8"))
    env = rep.get("environment", {})
    peak = rep.get("peak_vram_allocated_gb")
    return {
        "chunks_per_s": rep.get("chunks_per_s"),
        "peak_mib": peak * 1024 if peak is not None else None,
        "device": f"Kaggle {env.get('device', 'GPU')}",
    }


def kaggle_vectors(settings, model_name: str, vdir: Path, year: int, ids, queries):
    """Kaggle document vectors, checked against the local embedder, plus local query vectors."""
    import torch

    from nyaya.models.encoders import Embedder, embed_profile, startup_check

    profile = embed_profile(settings, model_name)
    dv = kaggle_doc_vectors(vdir, year, ids, profile.dim)
    embedder = Embedder(settings, model_name=model_name)
    embed = settings.embed.model_copy(update={"dim": profile.dim})
    check = startup_check(settings.model_copy(update={"embed": embed}), embedder, vdir)
    if not check["reference_checked"]:
        raise SystemExit(f"{vdir}: no vectors/reference files to check the local embedder with")
    print(f"{model_name}: Kaggle reference vectors match, min cosine {check['min_cosine']:.5f}")
    qv = np.stack([embedder.embed_query(q) for q in queries])
    del embedder
    torch.cuda.empty_cache()
    return dv, qv


def _num(v: Any, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def render_report(results: dict[str, dict[str, Any]], *, dataset: str, year: int, n: int) -> str:
    lines = [
        "# Model selection",
        "",
        f"Written by `scripts/select_models.py` on {date.today().isoformat()}: `{dataset}` "
        f"({n} answerable questions) over the {year} build, exact cosine search, "
        "each embedder followed by Qwen3-Reranker-0.6B.",
        "",
        "| Embedder | recall@50 (judgment) | recall@50 (paragraph) | recall@5 reranked (judgment) "
        "| recall@5 reranked (paragraph) | MRR reranked (judgment) | chunks/s | peak GPU MiB "
        "| encoded on |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        s = r["summary"]
        lines.append(
            f"| {name} | {s['exact']['judgment']['recall@50']:.3f} "
            f"| {s['exact']['paragraph']['recall@50']:.3f} "
            f"| {s['exact_rerank']['judgment']['recall@5']:.3f} "
            f"| {s['exact_rerank']['paragraph']['recall@5']:.3f} "
            f"| {s['exact_rerank']['judgment']['mrr']:.3f} "
            f"| {_num(r.get('chunks_per_s'), '.1f')} | {_num(r.get('peak_mib'), '.0f')} "
            f"| {r.get('device', 'local GPU')} |"
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
    ap.add_argument("--models", default="", help="comma-separated (default: every configured)")
    ap.add_argument(
        "--vectors",
        action="append",
        default=[],
        metavar="MODEL=DIR",
        help="document vectors for MODEL from a Kaggle run (repeatable)",
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "model_selection.md")
    ap.add_argument("--build", type=Path, default=None, help="build directory (default: config)")
    args = ap.parse_args(argv)

    from nyaya.eval.wiring import rerank_fn
    from nyaya.models.encoders import QwenReranker, document_text, embed_profiles

    settings = load_settings()
    vectors = parse_vectors_args(args.vectors)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [
        p.model for p in embed_profiles(settings)
    ]
    engine = load_build(args.build or REPO / settings.paths.build, args.year)
    items = [
        i
        for i in read_jsonl(args.dataset, DevItem)
        if i.type == "answerable" and all(p.startswith(f"{args.year}_") for p in i.gold_paths)
    ]
    if not items:
        print(f"no answerable {args.year} questions in {args.dataset}", file=sys.stderr)
        return 2
    ids = list(engine.chunks)
    docs = [document_text(engine.chunks[i].header, engine.chunks[i].text) for i in ids]
    questions = [i.question for i in items]
    rerank = rerank_fn(QwenReranker(settings))

    results: dict[str, dict[str, Any]] = {}
    for name in models:
        if name in vectors:
            dv, qv = kaggle_vectors(settings, name, vectors[name], args.year, ids, questions)
            stats = kaggle_run_stats(vectors[name])
        else:
            dv, qv, rate, peak = encode(settings, name, docs, questions, args.batch)
            stats = {"chunks_per_s": rate, "peak_mib": peak, "device": "local GPU"}
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
        results[name] = {"summary": report.summary(), **stats}
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
