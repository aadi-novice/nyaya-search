"""`nyaya-eval replay-check`: the citation check re-run on a saved answers run, offline.

No LLM, no GPU, no database. Each record already holds what the check needs:

*   the retrieved chunk ids in rank order, so `P<n>` is `retrieved[n-1]`;
*   `contexts`: header, blank line, text for each retrieved chunk (the quote
    check runs on the text after the first blank line);
*   the verified sentences with their quotes;
*   the unverified sentences with their labels, and the rule breaks.

What older records lack is the LLM's quote on a failed citation. Since
2026-11-18 the check writes it into the rule break's `detail`, and that quote is
used. For a record without it, `proxy=True` stands in the sentence itself or a
quotation inside it (the candidate closest to the cited passage, skipping any
that the old check would have found, since the real quote was not found).
Proxy numbers are an estimate, never a measurement, and are labelled so.

Step 4 is not re-resolved offline: a reporter citation counts as resolved
unless the record logged it as `uncited_citation`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nyaya.answer.citation_check import check
from nyaya.answer.quote_snap import best_window
from nyaya.config import Settings
from nyaya.contracts import ChunkRecord, PassageRef, Status, Zone
from nyaya.eval.answers import NEITHER_STATUSES, REFUSAL_STATUSES

_QUOTE_KINDS = frozenset({"quote_length", "quote_not_found", "unknown_label"})
_STEP4_KINDS = frozenset({"no_citation", "uncited_case_name", "uncited_citation"})
_YEAR_SUFFIX = re.compile(r"\s*\(\d{4}\)\s*$")
_INLINE_QUOTE = re.compile(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]")

# normalize_text() as it was before 2026-11-18 (no U+2010/U+2011/U+2212 mapping):
# a proxy candidate that this would have found cannot be the quote that failed.
_OLD_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-",
    }
)  # fmt: skip


def _old_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_OLD_MAP)
    return " ".join(text.split()).lower()


def chunk_text(context: str) -> str:
    """The chunk text of a `contexts` entry (header, blank line, text)."""
    return context.split("\n\n", 1)[1] if "\n\n" in context else context


def _title(context: str) -> str:
    header = context.split("\n\n", 1)[0] if "\n\n" in context else ""
    return _YEAR_SUFFIX.sub("", header.split(" \u00b7 ")[0])


def _zone(chunk_id: str) -> Zone:
    part = chunk_id.split("#")[1] if chunk_id.count("#") >= 2 else "judgment"
    return Zone(part) if part in {"issue", "editorial", "judgment"} else Zone.JUDGMENT


def passages(rec: Mapping[str, Any]) -> tuple[dict[str, PassageRef], dict[str, ChunkRecord]]:
    """label_map and chunks rebuilt from a record's `retrieved` and `contexts`."""
    refs: dict[str, PassageRef] = {}
    chunks: dict[str, ChunkRecord] = {}
    known = {c["chunk_id"]: c for s in rec.get("answer", []) for c in s["citations"]}
    for n, (chunk_id, context) in enumerate(
        zip(rec.get("retrieved", []), rec.get("contexts", []), strict=False), start=1
    ):
        judgment_id = chunk_id.split("#")[0]
        zone = _zone(chunk_id)
        if chunk_id in known:
            cit = {k: v for k, v in known[chunk_id].items() if k in PassageRef.model_fields}
            refs[f"P{n}"] = PassageRef(**cit)
        else:
            refs[f"P{n}"] = PassageRef(
                chunk_id=chunk_id,
                judgment_id=judgment_id,
                case_name=_title(context),
                scr_citation="",
                pages=[],
                zone=zone,
            )
        chunks[chunk_id] = ChunkRecord(
            chunk_id=chunk_id,
            judgment_id=judgment_id,
            zone=zone,
            para_labels=[],
            para_refs=[],
            text=chunk_text(context),
            header="",
            char_start=0,
            char_end=1,
            pages=[],
            flags=[],
            clean_text_version="replay",
            chunker_version="replay",
        )
    return refs, chunks


def _failed_quotes(rec: Mapping[str, Any]) -> list[dict[str, list[dict[str, Any]]]]:
    """Per unverified sentence (in order): label -> its failing rule breaks.

    Sentences are aligned by walking the sentence indices of the rule breaks:
    an index is the next unverified sentence when it has a step-4 break, or
    when every one of that sentence's labels failed there.
    """
    by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for b in rec.get("rule_breaks", []):
        by_index[int(b.get("sentence_index", -1))].append(b)
    unverified = rec.get("unverified", [])
    out: list[dict[str, list[dict[str, Any]]]] = [{} for _ in unverified]
    u = 0
    for index in sorted(by_index):
        if u >= len(unverified):
            break
        bs = by_index[index]
        failing = [b for b in bs if b.get("kind") in _QUOTE_KINDS and not b.get("snap")]
        step4 = any(b.get("kind") in _STEP4_KINDS for b in bs)
        if step4 or sorted(b.get("label") for b in failing) == sorted(unverified[u]["labels"]):
            for b in failing:
                out[u].setdefault(str(b.get("label")), []).append(b)
            u += 1
    return out


def _candidates(sentence: str, min_words: int) -> list[str]:
    inline = [m.group(1) for m in _INLINE_QUOTE.finditer(sentence)]
    return [sentence] + [q for q in inline if len(q.split()) >= min_words]


def proxy_quote(sentence: str, text: str, min_words: int, *, failed: bool = True) -> str:
    """The candidate closest to `text` ('' if none).

    For a citation that `failed`, a candidate the old check would have found is
    skipped: it cannot be the quote that failed.
    """
    best, best_ratio = "", -1.0
    for cand in _candidates(sentence, min_words):
        if failed and _old_normalize(cand) in _old_normalize(text):
            continue
        ratio = best_window(cand, text)[0]
        if ratio > best_ratio:
            best, best_ratio = cand, ratio
    return best


@dataclass
class Rebuilt:
    llm_output: dict[str, Any]
    label_map: dict[str, PassageRef]
    chunks: dict[str, ChunkRecord]
    quote_source: dict[tuple[int, str], str]  # (sentence, label) -> "answer" | "trace" | "proxy"
    uncited: frozenset[str]


def rebuild(rec: Mapping[str, Any], *, proxy: bool, min_words: int) -> Rebuilt:
    """The LLM output of a record, as far as it can be recovered."""
    label_map, chunks = passages(rec)
    sentences: list[dict[str, Any]] = []
    source: dict[tuple[int, str], str] = {}
    for s in rec.get("answer", []):
        cits = [{"label": c["passage_label"], "quote": c["quote"]} for c in s["citations"]]
        for c in cits:
            source[(len(sentences), c["label"])] = "answer"
        sentences.append({"sentence": s["sentence"], "citations": cits})
    for u, failed in zip(rec.get("unverified", []), _failed_quotes(rec), strict=True):
        cits = []
        for label in u["labels"]:
            breaks = failed.get(label, [])
            kinds = {b.get("kind") for b in breaks}
            saved = next((b["detail"] for b in breaks if b.get("detail") is not None), None)
            ref = label_map.get(label)
            if saved is not None:
                quote, where = saved, "trace"
            elif kinds & {"quote_length", "unknown_label"} or not proxy or ref is None:
                quote, where = "", "unknown"  # fails again: the quote was not saved
            else:
                text = chunks[ref.chunk_id].text
                quote = proxy_quote(u["sentence"], text, min_words, failed=bool(breaks))
                where = "proxy"
            source[(len(sentences), label)] = where
            cits.append({"label": label, "quote": quote})
        sentences.append({"sentence": u["sentence"], "citations": cits})
    uncited = frozenset(
        str(b.get("detail"))
        for b in rec.get("rule_breaks", [])
        if b.get("kind") == "uncited_citation"
    )
    return Rebuilt(
        {"sentences": sentences, "gaps": list(rec.get("gaps", []))},
        label_map,
        chunks,
        source,
        uncited,
    )


def _llm_answered(rec: Mapping[str, Any]) -> bool:
    """The LLM wrote sentences for this record (otherwise the replay changes nothing)."""
    return bool(rec.get("answer") or rec.get("unverified"))


def replay_record(rec: Mapping[str, Any], settings: Settings, *, proxy: bool) -> dict[str, Any]:
    """One record re-checked: new answer, unverified, rule breaks, status and snaps."""
    if not _llm_answered(rec) or rec.get("status") in NEITHER_STATUSES:
        return {**rec, "snaps": []}
    rb = rebuild(rec, proxy=proxy, min_words=settings.quote.min_words)
    any_id = next(iter(rb.label_map.values())).judgment_id if rb.label_map else None
    result = check(
        rb.llm_output,
        rb.label_map,
        rb.chunks,
        settings,
        resolve_alias=lambda canonical: None if canonical in rb.uncited else any_id,
    )
    snaps = [
        {
            "id": rec.get("id"),
            "sentence_index": b.sentence_index,
            "label": b.label,
            "quote": b.detail,
            "quote_source": rb.quote_source.get((b.sentence_index, b.label or ""), "?"),
            **(b.snap or {}),
        }
        for b in result.rule_breaks
        if b.snap
    ]
    status = Status.ANSWERED.value if result.verified else Status.REFUSED.value
    return {
        **rec,
        "status": status,
        "answer": [s.model_dump(mode="json") for s in result.verified],
        "unverified": [u.model_dump(mode="json") for u in result.unverified],
        "rule_breaks": [b.to_dict() for b in result.rule_breaks],
        "snaps": snaps,
    }


def metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """G1 counts, the G5 answerable-refusal share and the adversarial must-refuse failures."""
    n_ver = sum(len(r["answer"]) for r in records)
    n_unv = sum(len(r["unverified"]) for r in records)
    dev = [r for r in records if r.get("set") == "dev" and r["status"] not in NEITHER_STATUSES]
    refused = [r["id"] for r in dev if r["status"] in REFUSAL_STATUSES]
    adv_fail = [
        r["id"]
        for r in records
        if r.get("set") == "adversarial"
        and (r.get("g9") or {}).get("expect") == "refuse"
        and r["status"] not in REFUSAL_STATUSES
        and r["answer"]
    ]
    return {
        "llm_sentences": n_ver + n_unv,
        "verified_sentences": n_ver,
        "unverified_sentences": n_unv,
        "unverified_share": round(n_unv / (n_ver + n_unv), 4) if n_ver + n_unv else None,
        "snapped_citations": sum(len(r.get("snaps", [])) for r in records),
        "answerable_counted": len(dev),
        "answerable_refused": len(refused),
        "answerable_refused_share": round(len(refused) / len(dev), 4) if dev else None,
        "answerable_refused_ids": refused,
        "must_refuse_answered": adv_fail,
    }


def replay(
    records: Iterable[Mapping[str, Any]], settings: Settings, *, proxy: bool
) -> dict[str, Any]:
    """Recorded metrics, replayed metrics, and every snap."""
    records = list(records)
    replayed = [replay_record(r, settings, proxy=proxy) for r in records]
    return {
        "proxy": proxy,
        "settings": {
            "snap": settings.quote.snap,
            "snap_min_similarity": settings.quote.snap_min_similarity,
            "snap_max_edits": settings.quote.snap_max_edits,
        },
        "recorded": metrics([{**r, "snaps": []} for r in records]),
        "replayed": metrics(replayed),
        "snaps": [s for r in replayed for s in r["snaps"]],
        "changed": [
            {"id": a["id"], "before": a["status"], "after": b["status"]}
            for a, b in zip(records, replayed, strict=True)
            if a["status"] != b["status"]
        ],
    }
