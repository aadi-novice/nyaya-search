"""G6 automatic chunk metrics (design, goal G6 and file 23).

*   **paragraph_integrity:** the share of numbered judgment paragraphs that lie
    whole inside one chunk. A paragraph that alone exceeds `chunk.max_words`
    is left out (it has to be split).
*   **size_compliance:** the share of judgment chunks with `chunk.min_words`
    <= words <= `chunk.max_words`. A chunk that is a whole opinion on its own
    (a short order) cannot be merged with anything and is left out; the strict
    share, which counts it, is reported too. Editorial and issue chunks are one
    per Held point and are reported but not gated.
*   **over_model_limit:** chunks whose embedded text (header, blank line,
    text) has more tokens than the model accepts, counted with the embedder's
    tokenizer.

Results go to build/chunk_metrics.json; the build exits with GATE_EXIT_CODE
and names each failing metric when a threshold is missed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from nyaya.chunk.chunker import count_words, find_paragraphs
from nyaya.config import Settings
from nyaya.contracts import ChunkRecord, Opinion, ParsedDoc, Zone, Zones

GATE_EXIT_CODE = 3
# Both candidate embedders accept 8,000 tokens or more (design step 5); the
# reranker accepts 32,000. Not a setting in config/settings.toml (yet).
DEFAULT_TOKEN_LIMIT = 8000

TokenCounter = Callable[[str], int]


def embedded_text(c: ChunkRecord) -> str:
    """What is embedded: the header, a blank line, then the chunk text."""
    return f"{c.header}\n\n{c.text}"


def _spans(zones: Zones, opinions: list[Opinion]) -> list[tuple[int, int]]:
    return [(o.char_start, o.char_end) for o in opinions] or [zones.judgment_span]


def paragraph_integrity(
    parsed: ParsedDoc,
    zones: Zones,
    opinions: list[Opinion],
    chunks: list[ChunkRecord],
    max_words: int,
) -> tuple[int, int]:
    """(intact, total) numbered judgment paragraphs, leaving out over-long ones."""
    text = parsed.clean_text
    jc = [(c.char_start, c.char_end) for c in chunks if c.zone is Zone.JUDGMENT]
    intact = total = 0
    for s, e in _spans(zones, opinions):
        for p in find_paragraphs(text, s, e):
            if count_words(text[p.start : p.end]) > max_words:
                continue
            total += 1
            if any(a <= p.start and p.end <= b for a, b in jc):
                intact += 1
    return intact, total


def size_compliance(
    zones: Zones,
    opinions: list[Opinion],
    chunks: list[ChunkRecord],
    min_words: int,
    max_words: int,
) -> dict[str, int]:
    """Counts: judgment chunks in size, total, and whole-opinion chunks left out."""
    jc = [c for c in chunks if c.zone is Zone.JUDGMENT]
    exempt_ids = set()
    for s, e in _spans(zones, opinions):
        inside = [c for c in jc if s <= c.char_start and c.char_end <= e]
        if len(inside) == 1 and count_words(inside[0].text) < min_words:
            exempt_ids.add(inside[0].chunk_id)
    ok = ok_strict = 0
    for c in jc:
        fits = min_words <= count_words(c.text) <= max_words
        ok_strict += fits
        if c.chunk_id not in exempt_ids:
            ok += fits
    return {
        "in_size": ok,
        "total": len(jc) - len(exempt_ids),
        "in_size_strict": ok_strict,
        "total_strict": len(jc),
        "whole_opinion_short": len(exempt_ids),
    }


def over_model_limit(
    chunks: Iterable[ChunkRecord], count_tokens: TokenCounter, limit: int = DEFAULT_TOKEN_LIMIT
) -> list[str]:
    """IDs of chunks whose embedded text is over `limit` tokens."""
    return [c.chunk_id for c in chunks if count_tokens(embedded_text(c)) > limit]


def doc_metrics(
    parsed: ParsedDoc,
    zones: Zones,
    opinions: list[Opinion],
    chunks: list[ChunkRecord],
    settings: Settings,
    count_tokens: TokenCounter | None = None,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
) -> dict[str, Any]:
    """Raw counts for one judgment; add them up with summarize()."""
    cfg = settings.chunk
    intact, total = paragraph_integrity(parsed, zones, opinions, chunks, cfg.max_words)
    size = size_compliance(zones, opinions, chunks, cfg.min_words, cfg.max_words)
    words = [count_words(c.text) for c in chunks if c.zone is Zone.JUDGMENT]
    other = [count_words(c.text) for c in chunks if c.zone is not Zone.JUDGMENT]
    return {
        "path": parsed.path,
        "paragraphs_intact": intact,
        "paragraphs_total": total,
        **size,
        "judgment_chunk_words": words,
        "editorial_chunk_words": other,
        "over_model_limit": over_model_limit(chunks, count_tokens, token_limit)
        if count_tokens
        else [],
        "tokens_counted": count_tokens is not None,
    }


def _quantiles(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    v = sorted(values)

    def q(p: float) -> float:
        return float(v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))])

    return {
        "min": v[0],
        "p10": q(0.1),
        "p50": q(0.5),
        "p90": q(0.9),
        "max": v[-1],
        "mean": sum(v) / len(v),
    }


def summarize(docs: Iterable[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    """Totals and the three G6 metrics over all judgments' doc_metrics()."""
    docs = list(docs)
    tot = {
        k: sum(d[k] for d in docs)
        for k in (
            "paragraphs_intact",
            "paragraphs_total",
            "in_size",
            "total",
            "in_size_strict",
            "total_strict",
            "whole_opinion_short",
        )
    }
    over = [cid for d in docs for cid in d["over_model_limit"]]
    words = [w for d in docs for w in d["judgment_chunk_words"]]
    ed_words = [w for d in docs for w in d["editorial_chunk_words"]]
    integrity = (
        tot["paragraphs_intact"] / tot["paragraphs_total"] if tot["paragraphs_total"] else 1.0
    )
    compliance = tot["in_size"] / tot["total"] if tot["total"] else 1.0
    strict = tot["in_size_strict"] / tot["total_strict"] if tot["total_strict"] else 1.0
    return {
        "judgments": len(docs),
        "paragraph_integrity": integrity,
        "size_compliance": compliance,
        "size_compliance_strict": strict,
        "over_model_limit": len(over),
        "over_model_limit_ids": over,
        "tokens_counted": all(d["tokens_counted"] for d in docs) if docs else False,
        "counts": tot,
        "judgment_chunk_words": _quantiles(words),
        "editorial_chunk_words": _quantiles(ed_words),
        "thresholds": {
            "min_paragraph_integrity": settings.g6.min_paragraph_integrity,
            "min_size_compliance": settings.g6.min_size_compliance,
            "max_over_model_limit": 0,
        },
    }


def failing_metrics(summary: dict[str, Any], settings: Settings) -> list[str]:
    """Names of the G6 metrics that miss their threshold."""
    failed = []
    if summary["paragraph_integrity"] < settings.g6.min_paragraph_integrity:
        failed.append("paragraph_integrity")
    if summary["size_compliance"] < settings.g6.min_size_compliance:
        failed.append("size_compliance")
    if summary["over_model_limit"] > 0:
        failed.append("over_model_limit")
    return failed


def write_chunk_metrics(summary: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def load_token_counter(settings: Settings) -> TokenCounter:  # pragma: no cover - needs the model
    """Token counter from the embedder's tokenizer (Hugging Face cache)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        settings.embed.model, revision=settings.embed.revision or None
    )
    return lambda s: len(tok(s, add_special_tokens=True)["input_ids"])
