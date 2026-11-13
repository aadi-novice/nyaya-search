"""`nyaya-eval build-devset`: development questions drafted from Held points.

Design: Technical Plan, "Development set", and file 40.

*   **Selection:** Held points (editorial chunks) with at least one resolved
    "[Para N]" link, spread across years and judgments with `devset.seed`.
*   **Drafting:** a free LLM writes one question per Held point, following the
    paraphrasing rules.
*   **Rejection (in code, not left to the LLM):** a question sharing a run of
    `devset.max_shared_words` or more words with its Held point, or naming a
    party, is rejected. One retry tells the model why.
*   **Output:** `eval/datasets/dev_v1.jsonl` and the hand-check CSV.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nyaya.contracts import ChunkRecord, Zone
from nyaya.eval.datasets import DevItem, Gold
from nyaya.text.normalize import normalize_text

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that appear in party names but say nothing about who the party is.
GENERIC_PARTY_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "another",
        "anr",
        "appellant",
        "appellants",
        "applicant",
        "association",
        "authority",
        "bank",
        "bench",
        "board",
        "by",
        "central",
        "chairman",
        "chief",
        "co",
        "collector",
        "commissioner",
        "committee",
        "company",
        "corporation",
        "council",
        "court",
        "customs",
        "department",
        "deputy",
        "director",
        "district",
        "etc",
        "excise",
        "for",
        "general",
        "government",
        "govt",
        "high",
        "in",
        "income",
        "india",
        "indian",
        "its",
        "limited",
        "ltd",
        "m",
        "managing",
        "mr",
        "mrs",
        "ms",
        "municipal",
        "ndash",
        "of",
        "officer",
        "on",
        "or",
        "ors",
        "other",
        "others",
        "petitioner",
        "police",
        "private",
        "pvt",
        "registrar",
        "respondent",
        "respondents",
        "s",
        "secretary",
        "service",
        "society",
        "state",
        "station",
        "tax",
        "the",
        "thr",
        "through",
        "trust",
        "union",
        "uoi",
        "v",
        "vs",
    }
)

SYSTEM_PROMPT = """You write research questions for testing a legal search engine over \
Indian Supreme Court judgments.

You are given one Held point: an editor's summary of a holding. Write ONE question that \
a lawyer might type, whose answer is that holding.

Rules:
- Paraphrase. Never copy a run of {n} or more consecutive words from the Held point.
- Never name a party, a person, a company or a place from the case.
- Ask about the legal point, not about "this case" or "the appellant".
- One sentence, 8 to 30 words, ending with a question mark.

Reply with JSON only: {{"question": "..."}}"""


class JsonLLM(Protocol):
    """The part of `answer.llm_client.LLMClient` this module needs."""

    def complete_json(self, system: str, user: str, timeout: float) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HeldCandidate:
    path: str
    year: int
    held_text: str
    para_labels: list[str]  # resolved "[Para N]" labels: the gold paragraphs
    source_held_chunk: str | None
    petitioner: str = ""
    respondent: str = ""
    title: str = ""


@dataclass
class DraftResult:
    items: list[DevItem] = field(default_factory=list)
    candidates: dict[str, HeldCandidate] = field(default_factory=dict)  # by item id
    rejected: list[dict[str, str]] = field(default_factory=list)
    llm_calls: int = 0
    aborted: bool = False  # stopped after repeated provider failures


# --- the paraphrase rule -------------------------------------------------------------


def words(text: str) -> list[str]:
    """Lower-case word tokens after normalize_text(); punctuation is dropped."""
    return _WORD_RE.findall(normalize_text(text))


def shared_run(question: str, source: str, n: int) -> str | None:
    """The first run of `n` consecutive words of `question` also found in `source`."""
    q, s = words(question), words(source)
    if len(q) < n or len(s) < n:
        return None
    grams = {tuple(s[i : i + n]) for i in range(len(s) - n + 1)}
    for i in range(len(q) - n + 1):
        gram = tuple(q[i : i + n])
        if gram in grams:
            return " ".join(gram)
    return None


def party_terms(petitioner: str, respondent: str, title: str) -> set[str]:
    """Distinctive words from the party names (generic words such as "State" removed)."""
    text = " ".join([petitioner or "", respondent or "", title or ""])
    return {w for w in words(text) if len(w) >= 3 and not w.isdigit()} - GENERIC_PARTY_WORDS


def check_paraphrase(question: str, cand: HeldCandidate, max_shared_words: int) -> str | None:
    """The reason to reject `question`, or None if it follows the rules."""
    q = words(question)
    if len(q) < 5:
        return "too_short"
    run = shared_run(question, cand.held_text, max_shared_words)
    if run is not None:
        return f"shared_run: {run}"
    terms = party_terms(cand.petitioner, cand.respondent, cand.title)
    for w in q:
        if w in terms:
            return f"party_name: {w}"
    return None


# --- selection -------------------------------------------------------------------------


def order_candidates(cands: Iterable[HeldCandidate], seed: int) -> list[HeldCandidate]:
    """A deterministic order spread across years, then across judgments.

    Years take turns. Within a year, each judgment gives one Held point before
    any judgment gives a second.
    """
    rng = random.Random(seed)
    by_year: dict[int, dict[str, list[HeldCandidate]]] = defaultdict(lambda: defaultdict(list))
    for c in sorted(cands, key=lambda c: (c.year, c.path, c.source_held_chunk or "", c.held_text)):
        by_year[c.year][c.path].append(c)

    streams: list[list[HeldCandidate]] = []
    for year in sorted(by_year):
        paths = sorted(by_year[year])
        rng.shuffle(paths)
        points = {p: rng.sample(by_year[year][p], len(by_year[year][p])) for p in paths}
        stream: list[HeldCandidate] = []
        depth = max(len(v) for v in points.values())
        for k in range(depth):
            stream.extend(points[p][k] for p in paths if k < len(points[p]))
        streams.append(stream)

    out: list[HeldCandidate] = []
    for k in range(max((len(s) for s in streams), default=0)):
        out.extend(s[k] for s in streams if k < len(s))
    return out


def resolved_labels(para_refs: Sequence[str], linked_labels: Iterable[Sequence[str]]) -> list[str]:
    """The refs that a linked judgment chunk carries (as "N", or as "N.x")."""
    have = [lab for labels in linked_labels for lab in labels]
    out = []
    for ref in para_refs:
        if any(lab == ref or lab.startswith(ref + ".") for lab in have) and ref not in out:
            out.append(ref)
    return out


# --- drafting --------------------------------------------------------------------------


def _user_prompt(cand: HeldCandidate, previous: str | None, reason: str | None) -> str:
    parties = ", ".join(x for x in (cand.petitioner, cand.respondent) if x) or "(unknown)"
    msg = f"Held point:\n{cand.held_text}\n\nParty names you must not use: {parties}\n"
    if previous is not None:
        msg += (
            f'\nYour previous question was rejected ({reason}):\n"{previous}"\n'
            "Write a new question that follows the rules.\n"
        )
    return msg


def draft_devset(
    cands: Iterable[HeldCandidate],
    llm: JsonLLM,
    *,
    size: int,
    max_shared_words: int,
    seed: int,
    retries: int = 1,
    timeout: float = 60.0,
    id_prefix: str = "dev",
    max_consecutive_errors: int = 3,
) -> DraftResult:
    """Draft questions in `order_candidates` order until `size` are kept.

    Stops early (setting `aborted`) after `max_consecutive_errors` provider
    failures in a row, so a dead key or an exhausted quota does not burn calls.
    """
    result = DraftResult()
    errors_in_row = 0
    system = SYSTEM_PROMPT.format(n=max_shared_words)
    for cand in order_candidates(cands, seed):
        if len(result.items) >= size or errors_in_row >= max_consecutive_errors:
            result.aborted = errors_in_row >= max_consecutive_errors
            break
        previous: str | None = None
        reason: str | None = None
        for _attempt in range(retries + 1):
            result.llm_calls += 1
            try:
                reply = llm.complete_json(system, _user_prompt(cand, previous, reason), timeout)
            except Exception as exc:  # any provider failure: move to the next point
                errors_in_row += 1
                status = getattr(exc, "status_code", None)
                detail = f"{type(exc).__name__}" + (f" {status}" if status else "")
                result.rejected.append(_rejection(cand, "", f"llm_error: {detail}"))
                break
            errors_in_row = 0
            question = reply.get("question") if isinstance(reply, dict) else None
            if not isinstance(question, str):
                result.rejected.append(_rejection(cand, "", "bad_response"))
                break
            question = " ".join(question.split())
            reason = check_paraphrase(question, cand, max_shared_words)
            if reason is None:
                item_id = f"{id_prefix}-{len(result.items) + 1:04d}"
                result.items.append(
                    DevItem(
                        id=item_id,
                        question=question,
                        type="answerable",
                        gold=[Gold(path=cand.path, para_labels=cand.para_labels)],
                        source_held_chunk=cand.source_held_chunk,
                    )
                )
                result.candidates[item_id] = cand
                break
            result.rejected.append(_rejection(cand, question, reason))
            previous = question
    return result


def _rejection(cand: HeldCandidate, question: str, reason: str) -> dict[str, str]:
    return {
        "path": cand.path,
        "source_held_chunk": cand.source_held_chunk or "",
        "question": question,
        "reason": reason,
    }


HANDCHECK_FIELDS = ["id", "question", "held_text", "path", "para_labels", "ok", "comment"]


def write_handcheck_csv(path: Path | str, result: DraftResult, n: int, seed: int) -> Path:
    """`n` kept questions, sampled with `seed`, for the author's hand check.

    The author fills `ok` (y/n) and `comment`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = random.Random(seed).sample(result.items, min(n, len(result.items)))
    items.sort(key=lambda i: i.id)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HANDCHECK_FIELDS)
        w.writeheader()
        for item in items:
            cand = result.candidates[item.id]
            w.writerow(
                {
                    "id": item.id,
                    "question": item.question,
                    "held_text": cand.held_text,
                    "path": cand.path,
                    "para_labels": " ".join(cand.para_labels),
                    "ok": "",
                    "comment": "",
                }
            )
    return path


# --- reading build outputs ----------------------------------------------------------------


def load_candidates_from_build(build_dir: Path | str, years: Iterable[int]) -> list[HeldCandidate]:
    """Held points with resolved links, from `nyaya-build parse-chunk` outputs.

    Reads `chunks/<year>.jsonl`, `links/<year>.parquet` and
    `judgments/<year>.parquet` under `build_dir`.
    """
    import pandas as pd

    build_dir = Path(build_dir)
    out: list[HeldCandidate] = []
    for year in years:
        chunks_path = build_dir / "chunks" / f"{year}.jsonl"
        links_path = build_dir / "links" / f"{year}.parquet"
        judg_path = build_dir / "judgments" / f"{year}.parquet"
        for p in (chunks_path, links_path, judg_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} (run `nyaya-build parse-chunk --years {year}`)")
        chunks: dict[str, ChunkRecord] = {}
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = ChunkRecord.model_validate(json.loads(line))
                    chunks[c.chunk_id] = c
        links = pd.read_parquet(links_path, columns=["from_chunk_id", "to_chunk_id"])
        targets: dict[str, list[str]] = defaultdict(list)
        for frm, to in zip(links["from_chunk_id"], links["to_chunk_id"], strict=True):
            targets[frm].append(to)
        meta = pd.read_parquet(judg_path)
        info = {
            r["path"]: r
            for r in meta[
                [c for c in ("path", "title", "petitioner", "respondent") if c in meta.columns]
            ].to_dict("records")
        }
        for c in chunks.values():
            if c.zone is not Zone.EDITORIAL or not c.para_refs or c.chunk_id not in targets:
                continue
            linked = [chunks[t].para_labels for t in targets[c.chunk_id] if t in chunks]
            labels = resolved_labels(c.para_refs, linked)
            if not labels:
                continue
            m = info.get(c.judgment_id, {})
            out.append(
                HeldCandidate(
                    path=c.judgment_id,
                    year=year,
                    held_text=c.text,
                    para_labels=labels,
                    source_held_chunk=c.chunk_id,
                    petitioner=str(m.get("petitioner") or ""),
                    respondent=str(m.get("respondent") or ""),
                    title=str(m.get("title") or ""),
                )
            )
    return out


# --- the LLM ---------------------------------------------------------------------------------


class OpenAICompatLLM:
    """A minimal OpenAI-compatible JSON client for drafting (temperature 0).

    Used when `nyaya.answer.llm_client.LLMClient` is not available; it reads the
    same LLM_BASE_URL, LLM_API_KEY and LLM_MODEL secrets.
    """

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 300):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str, timeout: float) -> dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            timeout=timeout,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return parse_json_reply(resp.choices[0].message.content or "")


def parse_json_reply(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a Markdown code fence around it."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        raise ValueError("no JSON object in reply")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data
