"""Held links: editorial chunks to the judgment paragraphs they name (file 25).

A Held point's para_refs entry N links to the judgment chunks of the same
judgment whose para_labels contain N or start with "N.". When N is a
sub-paragraph ("12.1") that no chunk labels, it falls back to the chunk with
its whole paragraph ("12"). When the label is found in several opinions
(numbering restarts per opinion), the majority opinion wins; with no single
majority opinion every match is linked and the reference counts as ambiguous.

References that resolve to nothing are counted in build/link_summary.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nyaya.contracts import ChunkRecord, OpinionType, Zone

Link = tuple[str, str]


def _matches(ref: str, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    hit = [
        c
        for c in chunks
        if ref in c.para_labels or any(lab.startswith(ref + ".") for lab in c.para_labels)
    ]
    if not hit and "." in ref:
        base = ref.split(".")[0]
        hit = [c for c in chunks if base in c.para_labels]
    return hit


def _pick_opinion(hits: list[ChunkRecord]) -> tuple[list[ChunkRecord], bool]:
    """Keep one opinion's chunks: the majority one. Returns (chunks, ambiguous)."""
    groups: dict[tuple[str | None, Any], list[ChunkRecord]] = {}
    for c in hits:
        groups.setdefault((c.opinion_author, c.opinion_type), []).append(c)
    if len(groups) == 1:
        return hits, False
    majority = [g for k, g in groups.items() if k[1] is OpinionType.MAJORITY]
    if len(majority) == 1:
        return majority[0], False
    return hits, True


def resolve_held_links(chunks: Iterable[ChunkRecord]) -> tuple[list[Link], dict[str, Any]]:
    """(links, summary) for all chunks given (any number of judgments)."""
    by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for c in chunks:
        by_doc[c.judgment_id].append(c)
    links: list[Link] = []
    seen: set[Link] = set()  # list membership made a 400-page judgment take 6 minutes
    summary: dict[str, Any] = {
        "held_points": 0,
        "held_points_with_refs": 0,
        "para_refs": 0,
        "resolved": 0,
        "unresolved": 0,
        "ambiguous": 0,
        "unresolved_refs": [],
    }
    for doc_chunks in by_doc.values():
        judgment = [c for c in doc_chunks if c.zone is Zone.JUDGMENT]
        for ed in (c for c in doc_chunks if c.zone is Zone.EDITORIAL):
            summary["held_points"] += 1
            if ed.para_refs:
                summary["held_points_with_refs"] += 1
            for ref in ed.para_refs:
                summary["para_refs"] += 1
                hits = _matches(ref, judgment)
                if not hits:
                    summary["unresolved"] += 1
                    summary["unresolved_refs"].append({"chunk_id": ed.chunk_id, "ref": ref})
                    continue
                summary["resolved"] += 1
                hits, ambiguous = _pick_opinion(hits)
                summary["ambiguous"] += int(ambiguous)
                for h in hits:
                    link = (ed.chunk_id, h.chunk_id)
                    if link not in seen:
                        seen.add(link)
                        links.append(link)
    summary["resolution_rate"] = (
        summary["resolved"] / summary["para_refs"] if summary["para_refs"] else None
    )
    return links, summary


def held_links(chunks: Iterable[ChunkRecord]) -> list[Link]:
    """[(from_chunk_id, to_chunk_id)] from editorial chunks to judgment chunks."""
    return resolve_held_links(chunks)[0]


def merge_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Add per-year (or per-judgment) summaries into one."""
    out: dict[str, Any] = {
        "held_points": 0,
        "held_points_with_refs": 0,
        "para_refs": 0,
        "resolved": 0,
        "unresolved": 0,
        "ambiguous": 0,
        "unresolved_refs": [],
    }
    for s in summaries:
        for k in (
            "held_points",
            "held_points_with_refs",
            "para_refs",
            "resolved",
            "unresolved",
            "ambiguous",
        ):
            out[k] += s[k]
        out["unresolved_refs"].extend(s["unresolved_refs"])
    out["resolution_rate"] = out["resolved"] / out["para_refs"] if out["para_refs"] else None
    return out


def write_link_summary(summary: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path
