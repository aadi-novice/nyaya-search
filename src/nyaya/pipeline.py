"""`ask(request, deps) -> AskResponse`: the query pipeline (design, "Query pipeline").

| Step                          | citation_lookup | hybrid             | hybrid, LLM failure |
| alias lookup                  | yes             | no                 | no                  |
| search, fusion, expansion     | no              | yes                | yes                 |
| rerank (with fallback), gate  | no              | yes                | yes                 |
| LLM, then citation check      | no              | yes                | the LLM fails       |
| status                        | passages        | answered / refused | passages            |

A citation that does not resolve to a loaded judgment falls back to the hybrid
route, with the question unchanged. No candidates under active filters gives
`no_match_with_filters`; no candidates without filters gives `refused`.
Any unexpected exception gives `error`. Every stage is timed, and one trace
row is written per request; a trace failure never fails the request.

Search, encoders, routing, fusion and expansion are built elsewhere and reach
this module only through `Deps`, so tests use fakes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from nyaya.answer.citation_check import FindCitations, check
from nyaya.answer.llm_client import JSONCompleter, LLMBadJSON, LLMUnavailable, generate_answer
from nyaya.answer.notices import notices
from nyaya.answer.prompt import Passage, build_prompt
from nyaya.config import Settings
from nyaya.contracts import (
    AskRequest,
    AskResponse,
    ChunkRecord,
    Era,
    Filters,
    JudgmentRecord,
    PassageRef,
    Route,
    SearchHit,
    Status,
    Versions,
    Zone,
)
from nyaya.trace import TraceRecord

log = logging.getLogger(__name__)

NOTE_LOOKUP = "judgment found by citation"
NOTE_LLM_UNAVAILABLE = "no summary generated (LLM unavailable)"
NOTE_LLM_INVALID = "no summary generated (LLM reply was invalid)"
NOTE_ERROR = "the request failed; please try again"
# Citation kinds that identify one judgment (design, "Router"); SCC and AIR do not.
LOOKUP_KINDS = frozenset({"scr", "neutral", "insc", "cnr"})


# --- interfaces (search/interface.py, models/encoders.py) -----------------------


class SearchEngine(Protocol):
    """The design's SearchEngine, plus `chunks_for_judgment` for citation lookup."""

    def keyword(self, question: str, filters: Filters | None, k: int) -> list[SearchHit]: ...

    def vector(
        self, qvec: Sequence[float], filters: Filters | None, k: int, exact: bool
    ) -> list[SearchHit]: ...

    def get_chunks(self, ids: Sequence[str]) -> list[ChunkRecord]: ...

    def get_judgment(self, path: str) -> JudgmentRecord | None: ...

    def resolve_alias(self, canonical: str) -> str | None: ...

    def links_from(self, chunk_ids: Sequence[str]) -> dict[str, list[str]]: ...

    def chunks_for_judgment(self, path: str) -> list[ChunkRecord]: ...


class QueryEncoder(Protocol):
    def encode_query(self, text: str) -> Sequence[float]: ...


class Reranker(Protocol):
    model: str

    def score(self, question: str, documents: Sequence[str]) -> list[float]:
        """One relevance score in [0, 1] per document, in input order."""
        ...


@dataclass
class Deps:
    settings: Settings
    engine: SearchEngine
    encoder: QueryEncoder
    reranker: Reranker
    fallback_reranker: Reranker | None
    llm: JSONCompleter | None  # None: not configured, treated as unavailable
    route: Callable[[str], Route]  # search/router.py
    find_citations: FindCitations  # link/aliases.py
    rrf: Callable[[list[list[SearchHit]], int], list[SearchHit]]  # search/fusion.py
    expand: Callable[[list[SearchHit], SearchEngine], list[SearchHit]]  # search/expand.py
    index_version: str
    # read from the active index_versions row, not from config
    threshold_primary: float
    threshold_fallback: float
    write_trace: Callable[[TraceRecord], object] = lambda record: None
    clock: Callable[[], float] = time.monotonic
    new_request_id: Callable[[], str] = field(default=lambda: uuid.uuid4().hex)


class RerankFailed(RuntimeError):
    pass


# --- per-request state ----------------------------------------------------------


@dataclass
class _Run:
    request: AskRequest
    deps: Deps
    request_id: str
    started_at: float
    route: Route = Route.HYBRID
    timings: dict[str, float] = field(default_factory=dict)
    reranker_model: str = ""
    fallback_used: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        t0 = self.deps.clock()
        try:
            yield
        finally:
            self.timings[stage] = round((self.deps.clock() - t0) * 1000, 3)

    def respond(self, status: Status, **fields: Any) -> AskResponse:
        deps, s = self.deps, self.deps.settings
        self.timings["total"] = round((deps.clock() - self.started_at) * 1000, 3)
        versions = Versions(
            index_version=deps.index_version,
            config_version=s.versions.config,
            prompt_version=s.versions.prompt,
            llm_model=deps.llm.model if deps.llm is not None else (s.secrets.llm_model or ""),
            reranker_model=self.reranker_model or deps.reranker.model,
            reranker_fallback_used=self.fallback_used,
            embed_model=s.embed.model,
        )
        response = AskResponse(
            request_id=self.request_id,
            status=status,
            notices=notices(s),
            route=self.route,
            filters_applied=_filters_applied(self.request.filters),
            versions=versions,
            timings_ms=dict(self.timings),
            **fields,
        )
        self._write_trace(response)
        return response

    def _write_trace(self, response: AskResponse) -> None:
        t = self.trace
        record = TraceRecord(
            request_id=self.request_id,
            question=self.request.question,
            route=response.route.value,
            filters=response.filters_applied,
            versions=response.versions.model_dump(),
            retrieved=t.get("retrieved", []),
            rows_returned=t.get("rows_returned", {}),
            top_score=t.get("top_score"),
            threshold=t.get("threshold"),
            status=response.status.value,
            rule_breaks=t.get("rule_breaks", []),
            unverified_count=len(response.unverified),
            gaps_count=len(response.gaps),
            reranker_fallback_used=self.fallback_used,
            llm_error=t.get("llm_error"),
            timings_ms=response.timings_ms,
        )
        try:
            self.deps.write_trace(record)
        except Exception:
            log.exception("trace write failed for request %s", self.request_id)


def _filters_applied(filters: Filters | None) -> dict[str, Any]:
    return filters.model_dump(exclude_none=True) if filters is not None else {}


def _passage_ref(chunk: ChunkRecord, judgment: JudgmentRecord | None) -> PassageRef:
    return PassageRef(
        chunk_id=chunk.chunk_id,
        judgment_id=chunk.judgment_id,
        case_name=judgment.title if judgment else chunk.judgment_id,
        scr_citation=judgment.scr_citation if judgment else "",
        neutral_citation=judgment.neutral_citation if judgment else None,
        paras=list(chunk.para_labels) or None,
        pages=list(chunk.pages),
        zone=chunk.zone,
        opinion_type=chunk.opinion_type,
    )


def _passages(engine: SearchEngine, chunks: Sequence[ChunkRecord]) -> list[Passage]:
    judgments: dict[str, JudgmentRecord | None] = {}
    for c in chunks:
        if c.judgment_id not in judgments:
            judgments[c.judgment_id] = engine.get_judgment(c.judgment_id)
    return [Passage(ref=_passage_ref(c, judgments[c.judgment_id]), chunk=c) for c in chunks]


# --- citation lookup -------------------------------------------------------------


def lookup_chunks(
    judgment: JudgmentRecord | None, chunks: Sequence[ChunkRecord], max_editorial: int
) -> list[ChunkRecord]:
    """Design, "What citation_lookup returns"."""
    issue = [c for c in chunks if c.zone is Zone.ISSUE]
    editorial = [c for c in chunks if c.zone is Zone.EDITORIAL]
    if issue or editorial:
        if judgment is not None and judgment.era is Era.D:
            return issue[:1] + editorial
        return editorial[:max_editorial] if editorial else issue[:1]
    return [c for c in chunks if c.zone is Zone.JUDGMENT][:1]


def _lookup(run: _Run) -> AskResponse | None:
    deps = run.deps
    with run.timed("lookup"):
        path = None
        for _span, kind, canonical in deps.find_citations(run.request.question):
            if kind in LOOKUP_KINDS:
                path = deps.engine.resolve_alias(canonical)
                if path is not None:
                    break
        if path is None:
            return None
        judgment = deps.engine.get_judgment(path)
        chunks = lookup_chunks(
            judgment,
            deps.engine.chunks_for_judgment(path),
            deps.settings.retrieval.lookup_max_editorial,
        )
        refs = [_passage_ref(c, judgment) for c in chunks]
    run.trace["retrieved"] = [{"chunk_id": r.chunk_id} for r in refs]
    return run.respond(Status.PASSAGES, passages=refs, passages_note=NOTE_LOOKUP)


# --- hybrid ---------------------------------------------------------------------


def _rerank(run: _Run, question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
    docs = [f"{c.header}\n{c.text}" for c in chunks]
    deps = run.deps
    try:
        scores = deps.reranker.score(question, docs)
        run.reranker_model = deps.reranker.model
    except Exception as primary:
        log.warning("primary reranker failed: %s", primary)
        if deps.fallback_reranker is None:
            raise RerankFailed(f"reranker failed and no fallback: {primary}") from primary
        run.fallback_used = True
        run.reranker_model = deps.fallback_reranker.model
        try:
            scores = deps.fallback_reranker.score(question, docs)
        except Exception as fallback:
            raise RerankFailed(f"both rerankers failed: {fallback}") from fallback
    if len(scores) != len(docs):
        raise RerankFailed(f"reranker returned {len(scores)} scores for {len(docs)} documents")
    return [float(x) for x in scores]


def _hybrid(run: _Run) -> AskResponse:
    deps, s = run.deps, run.deps.settings
    r = s.retrieval
    question, filters = run.request.question, run.request.filters

    with run.timed("embed"):
        qvec = deps.encoder.encode_query(question)
    with run.timed("keyword"):
        kw = deps.engine.keyword(question, filters, r.keyword_k)
    with run.timed("vector"):
        vec = deps.engine.vector(qvec, filters, r.vector_k, False)
    run.trace["rows_returned"] = {
        "keyword": {"requested": r.keyword_k, "returned": len(kw)},
        "vector": {"requested": r.vector_k, "returned": len(vec)},
    }
    if not kw and not vec:
        if _filters_applied(filters):
            return run.respond(Status.NO_MATCH_WITH_FILTERS)
        return run.respond(Status.REFUSED)

    with run.timed("fusion"):
        fused = deps.rrf([kw, vec], r.rrf_k)
    with run.timed("expand"):
        candidates = sorted(deps.expand(fused, deps.engine), key=lambda h: h.fused_rank)
        candidates = candidates[: r.rerank_candidates]
    with run.timed("fetch"):
        by_id = {c.chunk_id: c for c in deps.engine.get_chunks([h.chunk_id for h in candidates])}
        hits = [h for h in candidates if h.chunk_id in by_id]
        chunks = [by_id[h.chunk_id] for h in hits]
    with run.timed("rerank"):
        scores = _rerank(run, question, chunks)
    ranked = sorted(zip(scores, hits, chunks, strict=True), key=lambda t: -t[0])[: r.final_k]
    top_chunks = [c for _, _, c in ranked]
    run.trace["retrieved"] = [
        {
            "chunk_id": h.chunk_id,
            "keyword_rank": h.keyword_rank,
            "vector_rank": h.vector_rank,
            "fused_rank": h.fused_rank,
            "rerank_score": score,
        }
        for score, h, _ in ranked
    ]
    with run.timed("fetch_judgments"):
        passages = _passages(deps.engine, top_chunks)
    refs = [p.ref for p in passages]
    if not ranked:
        return run.respond(Status.REFUSED)

    threshold = deps.threshold_fallback if run.fallback_used else deps.threshold_primary
    top_score = ranked[0][0]
    run.trace.update(top_score=top_score, threshold=threshold)
    if top_score < threshold:
        return run.respond(Status.REFUSED, nearest_passages=refs)

    with run.timed("prompt"):
        system, user, label_map = build_prompt(question, passages, s.versions.prompt, s)
    with run.timed("llm"):
        try:
            if deps.llm is None:
                raise LLMUnavailable("answer LLM not configured")
            result = generate_answer(
                deps.llm, system, user, started_at=run.started_at, settings=s, clock=deps.clock
            )
        except (LLMUnavailable, LLMBadJSON) as exc:
            run.trace["llm_error"] = f"{type(exc).__name__}: {exc}"
            note = NOTE_LLM_INVALID if isinstance(exc, LLMBadJSON) else NOTE_LLM_UNAVAILABLE
            return run.respond(Status.PASSAGES, passages=refs, passages_note=note)
    with run.timed("check"):
        chunk_map: Mapping[str, ChunkRecord] = {c.chunk_id: c for c in top_chunks}
        verified, unverified, gaps, breaks = check(
            result.output,
            label_map,
            chunk_map,
            s,
            resolve_alias=deps.engine.resolve_alias,
            find_citations=deps.find_citations,
        )
    run.trace["rule_breaks"] = [b.to_dict() for b in breaks]
    if not verified:
        return run.respond(Status.REFUSED, unverified=unverified, gaps=gaps, nearest_passages=refs)
    return run.respond(Status.ANSWERED, answer=verified, unverified=unverified, gaps=gaps)


# --- entry points ---------------------------------------------------------------


def ask(
    request: AskRequest,
    deps: Deps,
    *,
    request_id: str | None = None,
    started_at: float | None = None,
) -> AskResponse:
    """Answer one request. `started_at` (deps.clock time) is when it arrived."""
    run = _Run(
        request=request,
        deps=deps,
        request_id=request_id or deps.new_request_id(),
        started_at=deps.clock() if started_at is None else started_at,
    )
    try:
        with run.timed("route"):
            run.route = deps.route(request.question)
        if run.route is Route.CITATION_LOOKUP:
            response = _lookup(run)
            if response is not None:
                return response
            run.route = Route.HYBRID
        return _hybrid(run)
    except Exception:
        # the traces table has no error column; the log holds the details
        log.exception("request %s failed", run.request_id)
        return run.respond(Status.ERROR, passages_note=NOTE_ERROR)


def error_response(
    request: AskRequest,
    deps: Deps,
    *,
    request_id: str,
    note: str = NOTE_ERROR,
    started_at: float | None = None,
) -> AskResponse:
    """An `error` response, with its trace, for a request that did not finish
    (for example the API's `timeouts.request_s` limit)."""
    run = _Run(
        request=request,
        deps=deps,
        request_id=request_id,
        started_at=deps.clock() if started_at is None else started_at,
    )
    return run.respond(Status.ERROR, passages_note=note)
