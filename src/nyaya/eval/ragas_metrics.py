"""`nyaya-eval ragas`: RAGAS metrics over a finished `answers` run (never a gate).

RAGAS (ragas 0.4, the `ragas` extra) scores each development question of an
`answers` run with four LLM-judged metrics:

*   **faithfulness:** the share of the answer's claims that the retrieved
    passages support (claims split out and checked by the LLM);
*   **answer_relevancy:** how well questions generated back from the answer
    match the question (LLM plus a CPU embedding model; optional);
*   **context_precision:** whether the passages useful for the reference
    Held text are ranked high (ragas `ContextPrecisionWithReference`);
*   **context_recall:** the share of the reference Held text's claims that
    the retrieved passages support.

The answer is the verified sentences (`answer` of the record), the contexts
are the retrieved passages (`contexts`, the reranked top `retrieval.final_k`,
capped at `ragas.max_contexts`), and the reference is the question's Held
text from the dev set's hand-check CSV (`<dev set>_handcheck.csv`).

**Refusals.** Refused questions (`refused`, `no_match_with_filters`) have no
answer to judge: they are counted and their rate reported, and they are left
out of faithfulness and answer relevancy. So are questions that ended with no
verified sentence (`passages`, `error`). Context precision and recall depend
only on retrieval, so they are scored for every question with a reference
and at least one retrieved passage.

These numbers sit beside our own metrics and never replace them. RAGAS
faithfulness is an LLM's opinion; our quote check (G1) is exact and
code-verified: every shown quote is a normalised substring of its passage.

**Resuming.** Per-question scores go to `eval/runs/ragas/<answers run>/
results.jsonl`; running again skips finished questions, and a session stops
cleanly when the free-tier quota runs out (as in `answers`).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nyaya.eval.answers import REFUSAL_STATUSES, Progress, ResultsFile, summarize
from nyaya.eval.datasets import DATASETS_DIR, RUNS_DIR
from nyaya.eval.ratelimit import RateLimitExhausted

RAGAS_DIR = RUNS_DIR / "ragas"
ANSWER_METRICS = ("faithfulness", "answer_relevancy")
CONTEXT_METRICS = ("context_precision", "context_recall")
METRICS = ANSWER_METRICS + CONTEXT_METRICS
EXCLUSION_NOTE = (
    "Refused questions (refused, no_match_with_filters) and questions with no verified "
    "sentence (passages, error) are left out of faithfulness and answer_relevancy; "
    "context_precision and context_recall cover every question with a reference "
    "and retrieved passages."
)
LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision (with reference)",
    "context_recall": "Context recall",
}


@dataclass(frozen=True)
class Sample:
    id: str
    question: str
    response: str  # the verified sentences, joined
    contexts: tuple[str, ...]
    reference: str | None  # the Held text
    status: str
    refused: bool
    answered: bool  # not refused, and at least one verified sentence


def default_references(dev_stem: str | None) -> Path:
    return DATASETS_DIR / f"{dev_stem or 'dev_draft_v0'}_handcheck.csv"


def read_references(path: Path) -> dict[str, str]:
    """Question id -> Held text, from a hand-check CSV (`id`, `held_text`, ...)."""
    if not path.exists():
        raise FileNotFoundError(f"references not found: {path} (a <dev set>_handcheck.csv)")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"id", "held_text"} <= set(rows[0]):
        raise ValueError(f"{path}: needs the columns id and held_text")
    return {r["id"]: r["held_text"].strip() for r in rows if (r.get("held_text") or "").strip()}


def response_text(record: Mapping[str, Any]) -> str:
    return " ".join(s["sentence"].strip() for s in record.get("answer", []) if s.get("sentence"))


def build_samples(
    records: Iterable[Mapping[str, Any]],
    references: Mapping[str, str],
    *,
    max_contexts: int,
    fetch_contexts: Callable[[Sequence[str]], list[str]] | None = None,
) -> list[Sample]:
    """One sample per development question, in id order.

    Contexts come from the record (`contexts`); a record written before that
    field existed has them fetched by chunk id (`fetch_contexts`), or none.
    """
    out: list[Sample] = []
    for r in sorted((r for r in records if r.get("set") == "dev"), key=lambda r: r["id"]):
        contexts = r.get("contexts")
        if contexts is None:
            ids = list(r.get("retrieved", []))[:max_contexts]
            contexts = fetch_contexts(ids) if fetch_contexts is not None and ids else []
        response = response_text(r)
        refused = r["status"] in REFUSAL_STATUSES
        out.append(
            Sample(
                id=r["id"],
                question=r["question"],
                response=response,
                contexts=tuple(c for c in contexts[:max_contexts] if c.strip()),
                reference=references.get(r["id"]),
                status=r["status"],
                refused=refused,
                answered=not refused and bool(response),
            )
        )
    return out


# --- scoring ---------------------------------------------------------------------------


def skip_reason(name: str, s: Sample) -> str | None:
    """Why `name` is not scored for `s` (None: it is scored)."""
    if name in ANSWER_METRICS:
        if s.refused:
            return "refused"
        if not s.answered:
            return "no_answer"
        if name == "faithfulness" and not s.contexts:
            return "no_contexts"
        return None
    if not s.reference:
        return "no_reference"
    if not s.contexts:
        return "no_contexts"
    return None


def _inputs(name: str, s: Sample) -> dict[str, Any]:
    if name == "faithfulness":
        return {
            "user_input": s.question,
            "response": s.response,
            "retrieved_contexts": list(s.contexts),
        }
    if name == "answer_relevancy":
        return {"user_input": s.question, "response": s.response}
    return {
        "user_input": s.question,
        "reference": s.reference,
        "retrieved_contexts": list(s.contexts),
    }


def score_sample(s: Sample, metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """`{metric: {"score", "skipped", "error"}}`. RateLimitExhausted is raised."""
    out: dict[str, dict[str, Any]] = {}
    for name, metric in metrics.items():
        entry: dict[str, Any] = {"score": None, "skipped": skip_reason(name, s), "error": None}
        if entry["skipped"] is None:
            try:
                value = float(metric.score(**_inputs(name, s)).value)
            except RateLimitExhausted:
                raise
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            else:
                if math.isnan(value):
                    entry["error"] = "no score (nothing to judge)"
                else:
                    entry["score"] = value
        out[name] = entry
    return out


class QuotaExhausted(RuntimeError):
    """The LLM is still rate-limited after every retry; resume the run later."""


def run_session(
    samples: Sequence[Sample],
    metrics: Mapping[str, Any],
    results: ResultsFile,
    *,
    limit: int | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = lambda msg: None,
) -> Progress:
    """Score every unfinished sample (at most `limit`), appending one line each."""
    done = results.load()
    todo = [s for s in samples if s.id not in done]
    progress = Progress(total=len(samples), done_before=len(samples) - len(todo))
    for s in todo:
        if limit is not None and progress.finished_now >= limit:
            progress.stopped = f"limit of {limit} reached"
            break
        try:
            scores = score_sample(s, metrics)
        except RateLimitExhausted as exc:
            progress.stopped = f"{exc}; run again to resume"
            break
        results.append(
            {
                "run_id": results.run_id,
                "id": s.id,
                "question": s.question,
                "status": s.status,
                "refused": s.refused,
                "answered": s.answered,
                "contexts": len(s.contexts),
                "has_reference": s.reference is not None,
                "scores": scores,
                "finished_at": now().isoformat(),
            }
        )
        progress.finished_now += 1
        log(f"{s.id}: " + ", ".join(_fmt_entry(n, e) for n, e in scores.items()))
    return progress


def _fmt_entry(name: str, e: Mapping[str, Any]) -> str:
    if e["score"] is not None:
        return f"{name}={e['score']:.3f}"
    return f"{name}={e['skipped'] or 'error'}"


# --- report ----------------------------------------------------------------------------


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def aggregate(rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> dict[str, Any]:
    """Means over scored questions, with skips and errors counted per metric."""
    n = len(rows)
    refused = sum(1 for r in rows if r["refused"])
    no_answer = sum(1 for r in rows if not r["refused"] and not r["answered"])
    per: dict[str, Any] = {}
    for name in metric_names:
        entries = [r["scores"][name] for r in rows if name in r["scores"]]
        scores = [e["score"] for e in entries if e["score"] is not None]
        skipped: dict[str, int] = {}
        for e in entries:
            if e["skipped"]:
                skipped[e["skipped"]] = skipped.get(e["skipped"], 0) + 1
        per[name] = {
            "mean": sum(scores) / len(scores) if scores else None,
            "scored": len(scores),
            "skipped": dict(sorted(skipped.items())),
            "errors": sum(1 for e in entries if e["error"]),
        }
    return {
        "questions": n,
        "refusals": {
            "refused": refused,
            "refusal_rate": _share(refused, n),
            "no_answer": no_answer,
            "answered": n - refused - no_answer,
        },
        "metrics": per,
        "note": EXCLUSION_NOTE,
    }


def our_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    min_supported_share: float,
    min_unanswerable_refused_share: float,
    max_wrong_refusal_share: float,
) -> dict[str, Any]:
    """Our deterministic and judged metrics from the same answers run, for the table."""
    dev = [r for r in records if r.get("set") == "dev"]
    cites = [c for r in dev for s in r["sentences"] for c in s["citations"]]
    quote_ok = sum(1 for c in cites if c.get("quote_ok"))
    m = summarize(
        list(records),
        min_supported_share=min_supported_share,
        min_unanswerable_refused_share=min_unanswerable_refused_share,
        max_wrong_refusal_share=max_wrong_refusal_share,
    )
    g1, g2, g5, g9 = m["g1"], m["g2"], m["g5"], m["g9"]
    dev_sents = [s for r in dev for s in r["sentences"]]
    return {
        "quote_verification": {
            "rate": _share(quote_ok, len(cites)),
            "citations": len(cites),
            "note": "dev set: shown citations whose quote is a normalised substring of its passage",
        },
        "unverified_share": {
            "share": g1["unverified_share"],
            "sentences": g1["llm_sentences"],
            "note": "all sets: LLM sentences dropped by the quote check (G1)",
        },
        "citation_support": {
            "share": g2["by_set"].get("dev"),
            "sentences": sum(1 for s in dev_sents if s["verdict"] != "error"),
            "note": "dev set: verified sentences the G2 judge rates supported by a cited passage",
        },
        "wrong_refusal_share": {
            "share": g5["answerable"]["share"],
            "questions": g5["answerable"]["counted"],
            "target_max": max_wrong_refusal_share,
        },
        "unanswerable_refused_share": {
            "share": g5["unanswerable"]["share"],
            "questions": g5["unanswerable"]["counted"],
            "target_min": min_unanswerable_refused_share,
        },
        "adversarial": {
            "passed": g9["passed_count"],
            "questions": g9["questions"],
        },
    }


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def markdown_table(ours: Mapping[str, Any], ragas: Mapping[str, Any]) -> str:
    """A table the README can embed: our metrics first (primary), then RAGAS."""
    o = ours
    rows = [
        (
            "Quote verification (primary)",
            "ours, exact (code)",
            _f(o["quote_verification"]["rate"]),
            f"{o['quote_verification']['citations']} citations",
        ),
        (
            "Unverified sentence share (G1)",
            "ours, exact (code)",
            _f(o["unverified_share"]["share"]),
            f"{o['unverified_share']['sentences']} sentences",
        ),
        (
            "Citation support (G2)",
            "ours, LLM judge per citation",
            _f(o["citation_support"]["share"]),
            f"{o['citation_support']['sentences']} sentences",
        ),
        (
            "Wrong refusals, answerable (G5)",
            "ours, exact (status)",
            _f(o["wrong_refusal_share"]["share"]),
            f"{o['wrong_refusal_share']['questions']} questions",
        ),
    ]
    if o["unanswerable_refused_share"]["questions"]:
        rows.append(
            (
                "Refused, unanswerable (G5)",
                "ours, exact (status)",
                _f(o["unanswerable_refused_share"]["share"]),
                f"{o['unanswerable_refused_share']['questions']} questions",
            )
        )
    if o["adversarial"]["questions"]:
        a = o["adversarial"]
        rows.append(
            (
                "Adversarial passed (G9)",
                "ours, rules",
                _f(a["passed"] / a["questions"]),
                f"{a['questions']} questions",
            )
        )
    for name, m in ragas["metrics"].items():
        extra = f", {m['errors']} errors" if m["errors"] else ""
        kind = "RAGAS, LLM + embeddings" if name == "answer_relevancy" else "RAGAS, LLM"
        rows.append((LABELS[name], kind, _f(m["mean"]), f"{m['scored']} questions{extra}"))
    lines = [
        "| Metric | Kind | Score | Over |",
        "| --- | --- | --- | --- |",
        *(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows),
    ]
    r = ragas["refusals"]
    lines += [
        "",
        f"Refused: {r['refused']} of {ragas['questions']} development questions "
        f"(refusal rate {_f(r['refusal_rate'])}); {r['no_answer']} ended with no verified "
        "sentence. " + EXCLUSION_NOTE,
    ]
    return "\n".join(lines) + "\n"


# --- wiring (the LLM, embeddings and metrics) -----------------------------------------


def make_llm(settings: Any, clock: Any = None) -> tuple[Any, Any]:
    """The judge model as a ragas LLM, paced like the G2 judge; and its rate limiter."""
    from nyaya.answer.llm_client import LLMClient
    from nyaya.eval.judge import JudgeNotConfigured
    from nyaya.eval.ragas_adapters import CompleterLLM
    from nyaya.eval.ratelimit import PausableClock, RateLimitedCompleter

    s = settings.secrets
    missing = [
        name
        for name, value in (
            ("JUDGE_BASE_URL", s.judge_base_url),
            ("JUDGE_API_KEY", s.judge_api_key),
            ("JUDGE_MODEL", s.judge_model),
        )
        if not value
    ]
    if missing:
        raise JudgeNotConfigured(f"judge not configured: {', '.join(missing)} missing")
    a = settings.answers
    client = LLMClient(
        model=s.judge_model,
        max_output_tokens=settings.ragas.max_output_tokens,
        reasoning_effort=settings.judge.reasoning_effort,
        base_url=s.judge_base_url,
        api_key=s.judge_api_key.get_secret_value(),
    )
    limited = RateLimitedCompleter(
        client,
        clock=clock or PausableClock(),
        rpm=settings.judge.rpm,
        retries=a.rate_limit_retries,
        backoff_base_s=a.backoff_base_s,
        backoff_max_s=a.backoff_max_s,
    )
    return CompleterLLM(limited, timeout_s=settings.judge.timeout_s), limited


def make_embeddings(settings: Any) -> Any:
    from nyaya.eval.ragas_adapters import cpu_embeddings

    r = settings.ragas
    return cpu_embeddings(r.embed_model, r.embed_device)


def make_metrics(llm: Any, embeddings: Any | None, strictness: int) -> dict[str, Any]:
    """The four ragas collection metrics (answer relevancy only with `embeddings`)."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    metrics: dict[str, Any] = {"faithfulness": Faithfulness(llm=llm)}
    if embeddings is not None:
        metrics["answer_relevancy"] = AnswerRelevancy(
            llm=llm, embeddings=embeddings, strictness=strictness
        )
    metrics["context_precision"] = ContextPrecisionWithReference(llm=llm)
    metrics["context_recall"] = ContextRecall(llm=llm)
    return metrics


def ragas_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("ragas")
    except PackageNotFoundError:
        return None


class DbContexts:
    """Passage text by chunk id from Postgres, for records without `contexts`.

    Connects on first use only.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._conn: Any = None
        self._engine: Any = None

    def __call__(self, ids: Sequence[str]) -> list[str]:
        from nyaya.models.encoders import document_text

        if self._engine is None:
            from nyaya.search.postgres import PostgresSearch
            from nyaya.store.db import connect

            self._conn = connect(self.settings, autocommit=True)
            self._engine = PostgresSearch(settings=self.settings, conn=self._conn)
        chunks = {c.chunk_id: c for c in self._engine.get_chunks(list(ids))}
        return [document_text(chunks[i].header, chunks[i].text) for i in ids if i in chunks]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
