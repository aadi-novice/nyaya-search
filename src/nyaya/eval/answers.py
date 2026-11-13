"""`nyaya-eval answers`: full answer evaluation (design file 45; G1, G2, G5, G8, G9).

For each development, unanswerable and adversarial question, `pipeline.ask`
answers it, then:

*   **G1:** rule breaks before the check and the unverified share come from
    the trace and the response. Every shown citation is checked again here
    (its quote must be a normalised substring of its chunk's text); any
    failure is a G1 violation.
*   **G2:** the judge rates each verified sentence against each passage it
    cites, best first, stopping at the first `supported`. A sentence passes
    if any citation is `supported`; `partial` counts as unsupported. Pairs
    the judge could not label are `error`, and a sentence whose citations
    all gave `error` is left out of the G2 share and counted.
*   **G8, wrong answers:** a verified sentence that is not `supported` is
    labelled, in this order:
    1.  **hallucination:** no retrieved passage (the reranked top
        `retrieval.final_k`) rates it `supported` or `partial`;
    2.  **retrieval_error:** the question has gold paragraphs and none of
        them was retrieved (the paragraph-match rule of `retrieval.py`);
    3.  **reasoning_error:** otherwise (the right paragraph was retrieved,
        or the question has no gold, so retrieval cannot be blamed).
*   **G5:** refusal shares on unanswerable and answerable questions.
    `passages` and `error` results count as neither answers nor refusals,
    and are reported separately; `no_match_with_filters` counts as a refusal.
*   **G9:** each adversarial question's `expect` (eval/datasets/README.md),
    plus no unchecked citation details, and for `advice` and `false_premise`
    a refusal or the standing "not legal advice" notice.

**Resuming.** Results go to `eval/runs/answers/<run_id>/results.jsonl`, one
line per finished question, keyed by `(run_id, id)`. Running again with
`--run-id` skips finished questions, so a run can span days of free quota.
When the LLM or judge is still rate-limited after every retry, the question
is not recorded and the session stops. Each session's provider, models,
date and tokens are appended to `meta.json`. The `eval_runs` row is written
once every question is finished.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nyaya.contracts import AskRequest, AskResponse, ChunkRecord, Filters, Status
from nyaya.eval.datasets import RUNS_DIR, AdversarialItem, DevItem, Gold, read_jsonl
from nyaya.eval.judge import ERROR, Judge, best
from nyaya.eval.ratelimit import PausableClock, RateLimitedCompleter, RateLimitExhausted
from nyaya.eval.retrieval import label_matches
from nyaya.models.encoders import document_text
from nyaya.text.normalize import normalize_text

ANSWERS_DIR = RUNS_DIR / "answers"
SETS = ("dev", "unanswerable", "adversarial")
REFUSAL_STATUSES = frozenset({Status.REFUSED.value, Status.NO_MATCH_WITH_FILTERS.value})
NEITHER_STATUSES = frozenset({Status.PASSAGES.value, Status.ERROR.value})
ERROR_TYPES = ("hallucination", "retrieval_error", "reasoning_error")
ADVICE_NOTICE = "not legal advice"


@dataclass(frozen=True)
class Question:
    id: str
    set: str  # dev | unanswerable | adversarial
    question: str
    filters: Filters | None = None
    gold: tuple[Gold, ...] = ()
    category: str | None = None  # adversarial only
    expect: str | None = None  # adversarial only


def load_questions(
    dev: Path | None, unanswerable: Path | None, adversarial: Path | None
) -> list[Question]:
    """All questions, in set order then file order. A None path skips that set."""
    out: list[Question] = []
    for set_name, path in (("dev", dev), ("unanswerable", unanswerable)):
        if path is None:
            continue
        for d in read_jsonl(path, DevItem):
            out.append(Question(d.id, set_name, d.question, d.filters, tuple(d.gold)))
    if adversarial is not None:
        for a in read_jsonl(adversarial, AdversarialItem):
            out.append(
                Question(a.id, "adversarial", a.question, category=a.category, expect=a.expect)
            )
    seen: set[str] = set()
    for q in out:
        if q.id in seen:
            raise ValueError(f"question id {q.id!r} appears in more than one set")
        seen.add(q.id)
    return out


# --- results file (resume) -------------------------------------------------------------


class ResultsFile:
    """`results.jsonl` and `meta.json` of one answers run."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.dir = Path(root if root is not None else ANSWERS_DIR) / run_id
        self.path = self.dir / "results.jsonl"
        self.meta_path = self.dir / "meta.json"

    def exists(self) -> bool:
        return self.meta_path.exists()

    def load(self) -> dict[str, dict[str, Any]]:
        """Finished records by question id (a later line for an id replaces an earlier one)."""
        done: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return done
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash: that question is redone
                if rec.get("run_id") == self.run_id and "id" in rec:
                    done[rec["id"]] = rec
        return done

    def append(self, record: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        torn = False  # a line cut off by a crash: start the next record on a new line
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                torn = f.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            if torn:
                f.write("\n")
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def read_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )


# --- the pipeline, instrumented ----------------------------------------------------------

AskFn = Callable[[AskRequest], tuple[AskResponse, Any]]


def instrument(
    deps: Any, settings: Any, clock: PausableClock | None = None, *, pace_llm: bool = True
) -> tuple[AskFn, Any]:
    """Wrap `Deps` for an evaluation run.

    Returns `(ask_fn, answer_llm)`. `ask_fn(request)` gives the response and
    its trace record (the trace is still written as usual). With `pace_llm`,
    the LLM is paced and backs off on 429 (`answer_llm`, a
    `RateLimitedCompleter`, or None when no LLM is configured), and the
    request clock excludes those waits. Without it (`latency`), the LLM and
    clock are left alone.
    """
    from nyaya import pipeline

    a = settings.answers
    llm = None
    changes: dict[str, Any] = {}
    if pace_llm:
        clock = clock or PausableClock()
        changes["clock"] = clock
    if pace_llm and deps.llm is not None:
        llm = RateLimitedCompleter(
            deps.llm,
            clock=clock,
            rpm=a.llm_rpm,
            retries=a.rate_limit_retries,
            backoff_base_s=a.backoff_base_s,
            backoff_max_s=a.backoff_max_s,
        )
    if pace_llm:
        changes["llm"] = llm
    captured: list[Any] = []
    original = deps.write_trace

    def write_trace(record: Any) -> object:
        captured.append(record)
        return original(record)

    wrapped = replace(deps, write_trace=write_trace, **changes)

    def ask_fn(request: AskRequest) -> tuple[AskResponse, Any]:
        captured.clear()
        response = pipeline.ask(request, wrapped)
        return response, (captured[-1] if captured else None)

    return ask_fn, llm


# --- one question --------------------------------------------------------------------------


@dataclass
class Runner:
    ask: AskFn
    get_chunks: Callable[[Sequence[str]], list[ChunkRecord]]
    judge: Judge
    answer_llm: Any = None  # RateLimitedCompleter: tokens, and whether quota ran out
    judge_llm: Any = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


class QuotaExhausted(RuntimeError):
    """A model is still rate-limited after every retry; resume the run later."""


def _usage(llm: Any) -> tuple[dict[str, int], int]:
    if llm is None:
        return {}, 0
    return dict(getattr(llm, "tokens", {})), int(getattr(llm, "calls", 0))


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}


def quote_found(quote: str, chunk: ChunkRecord | None) -> bool:
    return chunk is not None and normalize_text(quote) in normalize_text(chunk.text)


def gold_retrieved(gold: Sequence[Gold], chunks: Iterable[ChunkRecord]) -> bool:
    """A retrieved chunk of a gold judgment carries a gold paragraph label.

    A gold entry without paragraph labels is matched at judgment level.
    """
    by_path = {g.path: g.para_labels for g in gold}
    for c in chunks:
        labels = by_path.get(c.judgment_id)
        if labels is None:
            continue
        if not labels or any(label_matches(lab, g) for lab in c.para_labels for g in labels):
            return True
    return False


def adversarial_check(q: Question, rec: dict[str, Any]) -> dict[str, Any]:
    """G9 for one adversarial question (eval/datasets/README.md, `expect`)."""
    answered = bool(rec["answer"])
    refused = rec["status"] in REFUSAL_STATUSES or not answered
    reasons: list[str] = []
    if not rec["citations_checked"]:
        reasons.append("a shown citation failed the re-check")
    if q.expect == "refuse" and not refused:
        reasons.append("expected a refusal (or no verified sentence)")
    if q.expect == "notice" and not rec["notice_shown"]:
        reasons.append("expected the standing notice")
    if q.category in ("advice", "false_premise") and not (refused or rec["notice_shown"]):
        reasons.append("advice/false-premise question without a refusal or the notice")
    return {
        "category": q.category,
        "expect": q.expect,
        "passed": not reasons,
        "reasons": reasons,
    }


def evaluate_question(q: Question, runner: Runner) -> dict[str, Any]:
    """Ask, check and judge one question. Raises QuotaExhausted to stop the run."""
    a_tokens0, a_calls0 = _usage(runner.answer_llm)
    j_tokens0, j_calls0 = _usage(runner.judge_llm)
    response, trace = runner.ask(AskRequest(question=q.question, filters=q.filters))
    if runner.answer_llm is not None and getattr(runner.answer_llm, "exhausted", False):
        raise QuotaExhausted("answer LLM still rate-limited after every retry")

    retrieved_ids = [r["chunk_id"] for r in (trace.retrieved if trace is not None else [])]
    cited_ids = [c.chunk_id for s in response.answer for c in s.citations]
    wanted = list(dict.fromkeys(retrieved_ids + cited_ids))
    chunks = {c.chunk_id: c for c in runner.get_chunks(wanted)} if wanted else {}
    retrieved = [chunks[i] for i in retrieved_ids if i in chunks]
    has_gold_retrieved = gold_retrieved(q.gold, retrieved) if q.gold else None

    sentences: list[dict[str, Any]] = []
    citations_checked = True
    try:
        for index, s in enumerate(response.answer):
            sentences.append(_judge_sentence(index, s, chunks, retrieved, q, runner.judge))
            citations_checked = citations_checked and sentences[-1]["citations_checked"]
    except RateLimitExhausted as exc:
        raise QuotaExhausted(f"judge still rate-limited after every retry: {exc}") from exc

    a_tokens, a_calls = _usage(runner.answer_llm)
    j_tokens, j_calls = _usage(runner.judge_llm)
    rec: dict[str, Any] = {
        "id": q.id,
        "set": q.set,
        "question": q.question,
        "finished_at": runner.now().isoformat(),
        "status": response.status.value,
        "route": response.route.value,
        "request_id": response.request_id,
        "versions": response.versions.model_dump(),
        "answer": [s.model_dump(mode="json") for s in response.answer],
        "unverified": [u.model_dump(mode="json") for u in response.unverified],
        "gaps": list(response.gaps),
        "notice_shown": any(ADVICE_NOTICE in n.lower() for n in response.notices),
        "retrieved": retrieved_ids,
        # the retrieved passages as the LLM saw them (header, blank line, text):
        # `nyaya-eval ragas` scores against these without the database
        "contexts": [document_text(c.header, c.text) for c in retrieved],
        "gold_retrieved": has_gold_retrieved,
        "rule_breaks": list(trace.rule_breaks) if trace is not None else [],
        "llm_error": trace.llm_error if trace is not None else None,
        "error": trace.error if trace is not None else None,
        "timings_ms": dict(response.timings_ms),
        "sentences": sentences,
        "citations_checked": citations_checked,
        "tokens": {"answer": _delta(a_tokens, a_tokens0), "judge": _delta(j_tokens, j_tokens0)},
        "llm_calls": {"answer": a_calls - a_calls0, "judge": j_calls - j_calls0},
    }
    if q.set == "adversarial":
        rec["g9"] = adversarial_check(q, rec)
    return rec


def _judge_sentence(
    index: int,
    s: Any,
    chunks: dict[str, ChunkRecord],
    retrieved: list[ChunkRecord],
    q: Question,
    judge: Judge,
) -> dict[str, Any]:
    cites: list[dict[str, Any]] = []
    judged: dict[str, str] = {}
    checked = True
    for c in s.citations:
        chunk = chunks.get(c.chunk_id)
        ok = quote_found(c.quote, chunk)
        checked = checked and ok
        entry: dict[str, Any] = {"chunk_id": c.chunk_id, "label": c.passage_label, "quote_ok": ok}
        if chunk is None:
            entry["verdict"] = ERROR
            entry["error"] = "chunk not found"
        elif c.chunk_id in judged or "supported" in judged.values():
            entry["verdict"] = judged.get(c.chunk_id, "skipped")
        else:
            v = judge.judge(s.sentence, document_text(chunk.header, chunk.text))
            judged[c.chunk_id] = v.label
            entry.update(verdict=v.label, reason=v.reason)
            if v.error:
                entry["error"] = v.error
        cites.append(entry)
    verdict = best(judged.values()) if judged else ERROR
    out: dict[str, Any] = {
        "index": index,
        "sentence": s.sentence,
        "verdict": verdict,
        "citations": cites,
        "citations_checked": checked,
        "wrong": verdict in ("partial", "unsupported"),
        "error_type": None,
    }
    if out["wrong"]:
        out["error_type"], out["grounding"] = _error_type(s.sentence, judged, retrieved, q, judge)
    return out


def _error_type(
    sentence: str,
    judged: dict[str, str],
    retrieved: list[ChunkRecord],
    q: Question,
    judge: Judge,
) -> tuple[str, dict[str, str]]:
    """Label a wrong sentence (see the module docstring). Stops at the first grounding."""
    grounding = dict(judged)
    grounded = any(v in ("supported", "partial") for v in grounding.values())
    for chunk in retrieved:
        if grounded:
            break
        if chunk.chunk_id in grounding:
            continue
        v = judge.judge(sentence, document_text(chunk.header, chunk.text))
        grounding[chunk.chunk_id] = v.label
        grounded = v.label in ("supported", "partial")
    if not grounded:
        return "hallucination", grounding
    if q.gold and not gold_retrieved(q.gold, retrieved):
        return "retrieval_error", grounding
    return "reasoning_error", grounding


# --- a session ------------------------------------------------------------------------------


@dataclass
class Progress:
    total: int
    done_before: int
    finished_now: int = 0
    stopped: str | None = None  # why the session stopped early

    @property
    def remaining(self) -> int:
        return self.total - self.done_before - self.finished_now

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "done_before": self.done_before,
            "finished_now": self.finished_now,
            "remaining": self.remaining,
            "stopped": self.stopped,
        }


def run_session(
    questions: Sequence[Question],
    runner: Runner,
    results: ResultsFile,
    *,
    limit: int | None = None,
    redo_errors: bool = False,
    log: Callable[[str], None] = lambda msg: None,
) -> Progress:
    """Answer every unfinished question (at most `limit`), appending one line each."""
    done = results.load()
    if redo_errors:
        done = {k: v for k, v in done.items() if v.get("status") != Status.ERROR.value}
    todo = [q for q in questions if q.id not in done]
    progress = Progress(total=len(questions), done_before=len(questions) - len(todo))
    for q in todo:
        if limit is not None and progress.finished_now >= limit:
            progress.stopped = f"limit of {limit} reached"
            break
        try:
            rec = evaluate_question(q, runner)
        except QuotaExhausted as exc:
            progress.stopped = f"{exc}; resume later with --run-id {results.run_id}"
            break
        results.append({"run_id": results.run_id, **rec})
        progress.finished_now += 1
        log(f"{q.id}: {rec['status']}")
    return progress


# --- metrics ------------------------------------------------------------------------------


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def _sum_tokens(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        for k, v in r.get("tokens", {}).get(key, {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    min_supported_share: float,
    min_unanswerable_refused_share: float,
    max_wrong_refusal_share: float,
) -> dict[str, Any]:
    """Run metrics from finished records (any order; one per question)."""
    by_set = {s: [r for r in records if r["set"] == s] for s in SETS}
    statuses = {s: _count(r["status"] for r in rs) for s, rs in by_set.items() if rs}

    # G1
    n_verified = sum(len(r["answer"]) for r in records)
    n_unverified = sum(len(r["unverified"]) for r in records)
    n_breaks = sum(len(r["rule_breaks"]) for r in records)
    n_out = n_verified + n_unverified
    violations = [
        {"id": r["id"], "index": s["index"]}
        for r in records
        for s in r["sentences"]
        if not s["citations_checked"]
    ]
    g1 = {
        "llm_sentences": n_out,
        "verified_sentences": n_verified,
        "unverified_sentences": n_unverified,
        "unverified_share": _share(n_unverified, n_out),
        "rule_breaks": n_breaks,
        "rule_breaks_per_sentence": _share(n_breaks, n_out),
        "answers_with_rule_breaks": sum(1 for r in records if r["rule_breaks"]),
        "rule_break_kinds": _count(b.get("kind", "?") for r in records for b in r["rule_breaks"]),
        "unchecked_citation_details": violations,
        "passed": not violations,
    }

    # G2
    sents = [s for r in records for s in r["sentences"]]
    judged = [s for s in sents if s["verdict"] != ERROR]
    supported = sum(1 for s in judged if s["verdict"] == "supported")
    g2_share = _share(supported, len(judged))
    g2 = {
        "judged_sentences": len(judged),
        "supported": supported,
        "verdicts": _count(s["verdict"] for s in sents),
        "judge_errors": len(sents) - len(judged),
        "supported_share": g2_share,
        "by_set": {
            name: _share(
                sum(1 for r in rs for s in r["sentences"] if s["verdict"] == "supported"),
                sum(1 for r in rs for s in r["sentences"] if s["verdict"] != ERROR),
            )
            for name, rs in by_set.items()
            if rs
        },
        "target": min_supported_share,
        "passed": g2_share is not None and g2_share >= min_supported_share,
    }

    # G5
    def refusal(rs: list[dict[str, Any]]) -> dict[str, Any]:
        counted = [r for r in rs if r["status"] not in NEITHER_STATUSES]
        refused = sum(1 for r in counted if r["status"] in REFUSAL_STATUSES)
        return {
            "counted": len(counted),
            "refused": refused,
            "share": _share(refused, len(counted)),
            "passages": sum(1 for r in rs if r["status"] == Status.PASSAGES.value),
            "errors": sum(1 for r in rs if r["status"] == Status.ERROR.value),
        }

    unans, dev = refusal(by_set["unanswerable"]), refusal(by_set["dev"])
    g5 = {
        "unanswerable": {**unans, "target_min": min_unanswerable_refused_share},
        "answerable": {**dev, "target_max": max_wrong_refusal_share},
        "passed": (
            unans["share"] is not None
            and unans["share"] >= min_unanswerable_refused_share
            and dev["share"] is not None
            and dev["share"] <= max_wrong_refusal_share
        ),
    }

    # G8 wrong answers
    wrong = [
        {"id": r["id"], "set": r["set"], "index": s["index"], "error_type": s["error_type"]}
        for r in records
        for s in r["sentences"]
        if s["wrong"]
    ]
    g8 = {
        "wrong_sentences": len(wrong),
        "by_type": {t: sum(1 for w in wrong if w["error_type"] == t) for t in ERROR_TYPES},
        "wrong": wrong,
    }

    # G9
    adv = [r for r in by_set["adversarial"] if "g9" in r]
    failures = [{"id": r["id"], **r["g9"]} for r in adv if not r["g9"]["passed"]]
    g9 = {
        "questions": len(adv),
        "passed_count": len(adv) - len(failures),
        "failures": failures,
        "by_category": {
            c: {
                "n": sum(1 for r in adv if r["g9"]["category"] == c),
                "passed": sum(1 for r in adv if r["g9"]["category"] == c and r["g9"]["passed"]),
            }
            for c in sorted({r["g9"]["category"] for r in adv})
        },
        "passed": bool(adv) and not failures,
    }

    return {
        "questions": len(records),
        "by_set": {s: len(rs) for s, rs in by_set.items()},
        "statuses": statuses,
        "g1": g1,
        "g2": g2,
        "g5": g5,
        "g8": g8,
        "g9": g9,
        "tokens": {
            "answer": _sum_tokens(records, "answer"),
            "judge": _sum_tokens(records, "judge"),
        },
        "llm_calls": {
            k: sum(int(r.get("llm_calls", {}).get(k, 0)) for r in records)
            for k in ("answer", "judge")
        },
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# --- session metadata ----------------------------------------------------------------------


def provider(base_url: str | None) -> str | None:
    """The host of an OpenAI-compatible base URL (never the key)."""
    if not base_url:
        return None
    return urlparse(base_url).hostname or base_url


def session_entry(
    settings: Any,
    *,
    started_at: datetime,
    progress: Progress,
    answer_llm: Any,
    judge_llm: Any,
    index_version: str | None,
) -> dict[str, Any]:
    sec = settings.secrets
    return {
        "started_at": started_at.isoformat(),
        "date": started_at.date().isoformat(),
        "index_version": index_version,
        "answer": {
            "provider": provider(sec.llm_base_url),
            "model": sec.llm_model,
            "tokens": dict(getattr(answer_llm, "tokens", {}) or {}),
            "calls": getattr(answer_llm, "calls", 0),
            "rate_limited": getattr(answer_llm, "rate_limited", 0),
        },
        "judge": {
            "provider": provider(sec.judge_base_url),
            "model": sec.judge_model,
            "prompt": settings.judge.prompt,
            "tokens": dict(getattr(judge_llm, "tokens", {}) or {}),
            "calls": getattr(judge_llm, "calls", 0),
            "rate_limited": getattr(judge_llm, "rate_limited", 0),
        },
        "progress": progress.as_dict(),
    }
