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

`search(request, deps) -> SearchResponse` runs the same route and the same
retrieval (`_retrieve`: search, fusion, expansion, rerank) and stops before the
LLM, so the page can show the passages before the written answer. It writes no
trace: the /ask that follows is traced. When `deps.retrieval_cache` is set,
`search` stores its ranked passages there and the next `ask` with the same
question, filters and index version reuses them instead of searching again.

Search, encoders, routing, fusion and expansion are built elsewhere and reach
this module only through `Deps`, so tests use fakes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
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
    SearchPassage,
    SearchResponse,
    Status,
    Versions,
    Zone,
)
from nyaya.models.encoders import document_text
from nyaya.search.postgres import InvalidFilter
from nyaya.trace import TraceRecord

log = logging.getLogger(__name__)

NOTE_LOOKUP = "judgment found by citation"
NOTE_LLM_UNAVAILABLE = "no summary generated (LLM unavailable)"
NOTE_LLM_INVALID = "no summary generated (LLM reply was invalid)"
NOTE_ERROR = "the request failed; please try again"
NOTE_INVALID_FILTER = "invalid filter"
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
    # opens a fresh database connection (the API's error traces use it)
    connect: Callable[[], Any] | None = None
    models_loaded: bool = True
    # filled by `search`, read by `ask` (None: no cache, every request retrieves)
    retrieval_cache: RetrievalCache | None = None

    def versions(self) -> Versions:
        """The versions of this deployment (no reranker fallback assumed)."""
        s = self.settings
        return Versions(
            index_version=self.index_version,
            config_version=s.versions.config,
            prompt_version=s.versions.prompt,
            llm_model=self.llm.model if self.llm is not None else (s.secrets.llm_model or ""),
            reranker_model=self.reranker.model,
            reranker_fallback_used=False,
            embed_model=s.embed.model,
        )

    def health(self) -> dict[str, Any]:
        return {
            "active_version": self.index_version,
            "models_loaded": self.models_loaded,
            "fallback_configured": self.fallback_reranker is not None,
            "llm_configured": self.llm is not None,
        }


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

    def _envelope(self, status: Status) -> dict[str, Any]:
        """The fields AskResponse and SearchResponse share."""
        deps = self.deps
        self.timings["total"] = round((deps.clock() - self.started_at) * 1000, 3)
        versions = deps.versions().model_copy(
            update={
                "reranker_model": self.reranker_model or deps.reranker.model,
                "reranker_fallback_used": self.fallback_used,
            }
        )
        return {
            "request_id": self.request_id,
            "status": status,
            "notices": notices(deps.settings),
            "route": self.route,
            "filters_applied": _filters_applied(self.request.filters),
            "versions": versions,
            "timings_ms": dict(self.timings),
        }

    def respond(self, status: Status, **fields: Any) -> AskResponse:
        response = AskResponse(**self._envelope(status), **fields)
        self._write_trace(response)
        return response

    def respond_search(self, status: Status, **fields: Any) -> SearchResponse:
        return SearchResponse(**self._envelope(status), **fields)

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
            error=t.get("error"),
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


def _judgments(
    engine: SearchEngine, chunks: Sequence[ChunkRecord]
) -> dict[str, JudgmentRecord | None]:
    judgments: dict[str, JudgmentRecord | None] = {}
    for c in chunks:
        if c.judgment_id not in judgments:
            judgments[c.judgment_id] = engine.get_judgment(c.judgment_id)
    return judgments


def _passages(
    chunks: Sequence[ChunkRecord], judgments: Mapping[str, JudgmentRecord | None]
) -> list[Passage]:
    return [Passage(ref=_passage_ref(c, judgments.get(c.judgment_id)), chunk=c) for c in chunks]


def _search_passage(
    rank: int, chunk: ChunkRecord, judgment: JudgmentRecord | None, score: float | None
) -> SearchPassage:
    return SearchPassage(
        **_passage_ref(chunk, judgment).model_dump(),
        rank=rank,
        rerank_score=score,
        text=chunk.text,
        decision_date=judgment.decision_date if judgment else None,
    )


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


def _lookup_chunks(run: _Run) -> tuple[JudgmentRecord | None, list[ChunkRecord]] | None:
    """The cited judgment and its lookup chunks; None if no citation resolves."""
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
    run.trace["retrieved"] = [{"chunk_id": c.chunk_id} for c in chunks]
    return judgment, chunks


def _lookup(run: _Run) -> AskResponse | None:
    found = _lookup_chunks(run)
    if found is None:
        return None
    judgment, chunks = found
    refs = [_passage_ref(c, judgment) for c in chunks]
    return run.respond(Status.PASSAGES, passages=refs, passages_note=NOTE_LOOKUP)


def _search_lookup(run: _Run) -> SearchResponse | None:
    found = _lookup_chunks(run)
    if found is None:
        return None
    judgment, chunks = found
    passages = [_search_passage(i, c, judgment, None) for i, c in enumerate(chunks, 1)]
    return run.respond_search(Status.PASSAGES, passages=passages, passages_note=NOTE_LOOKUP)


# --- hybrid ---------------------------------------------------------------------


def _rerank(run: _Run, question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
    docs = [document_text(c.header, c.text) for c in chunks]
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


@dataclass(frozen=True)
class _Retrieved:
    """The hybrid route's ranked passages: what /ask answers from and /search shows.

    `ranked` holds (rerank score, hit, chunk), best first, at most `final_k`.
    `candidates` is False when keyword and vector search both returned nothing.
    """

    candidates: bool
    ranked: tuple[tuple[float, SearchHit, ChunkRecord], ...] = ()
    judgments: Mapping[str, JudgmentRecord | None] = field(default_factory=dict)
    rows_returned: dict[str, Any] = field(default_factory=dict)
    retrieved: tuple[dict[str, Any], ...] = ()
    reranker_model: str = ""
    fallback_used: bool = False

    @property
    def chunks(self) -> list[ChunkRecord]:
        return [c for _, _, c in self.ranked]


CacheKey = tuple[str, str, str]


class RetrievalCache:
    """A small time-limited cache of retrieval results.

    `search` fills it; `ask` reads it, so the /ask the page sends right after a
    /search for the same question, filters and index version does not search
    and rerank a second time. Thread-safe; entries live `ttl_s` seconds.
    """

    def __init__(
        self,
        max_entries: int = 64,
        ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_entries
        self._ttl = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._items: OrderedDict[CacheKey, tuple[float, _Retrieved]] = OrderedDict()

    @staticmethod
    def key(index_version: str, request: AskRequest) -> CacheKey:
        filters = json.dumps(_filters_applied(request.filters), sort_keys=True)
        return (index_version, " ".join(request.question.split()), filters)

    def get(self, key: CacheKey) -> _Retrieved | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored_at, value = item
            if self._clock() - stored_at > self._ttl:
                del self._items[key]
                return None
            return value

    def put(self, key: CacheKey, value: _Retrieved) -> None:
        with self._lock:
            self._items[key] = (self._clock(), value)
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _search_and_rerank(run: _Run) -> _Retrieved:
    """keyword + vector -> RRF -> link expansion -> fetch -> rerank -> top `final_k`."""
    deps, r = run.deps, run.deps.settings.retrieval
    question, filters = run.request.question, run.request.filters

    with run.timed("embed"):
        qvec = deps.encoder.encode_query(question)
    with run.timed("keyword"):
        kw = deps.engine.keyword(question, filters, r.keyword_k)
    with run.timed("vector"):
        vec = deps.engine.vector(qvec, filters, r.vector_k, False)
    rows_returned = {
        "keyword": {"requested": r.keyword_k, "returned": len(kw)},
        "vector": {"requested": r.vector_k, "returned": len(vec)},
    }
    if not kw and not vec:
        return _Retrieved(candidates=False, rows_returned=rows_returned)

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
    with run.timed("fetch_judgments"):
        judgments = _judgments(deps.engine, [c for _, _, c in ranked])
    retrieved = tuple(
        {
            "chunk_id": h.chunk_id,
            "keyword_rank": h.keyword_rank,
            "vector_rank": h.vector_rank,
            "fused_rank": h.fused_rank,
            "rerank_score": score,
        }
        for score, h, _ in ranked
    )
    return _Retrieved(
        candidates=True,
        ranked=tuple(ranked),
        judgments=judgments,
        rows_returned=rows_returned,
        retrieved=retrieved,
        reranker_model=run.reranker_model,
        fallback_used=run.fallback_used,
    )


def _retrieve(run: _Run, *, fill_cache: bool) -> _Retrieved:
    """The hybrid retrieval shared by `ask` (reads the cache) and `search` (fills it)."""
    cache = run.deps.retrieval_cache
    key = RetrievalCache.key(run.deps.index_version, run.request)
    found = None
    if cache is not None and not fill_cache:
        with run.timed("retrieval_cache"):
            found = cache.get(key)
    if found is None:
        found = _search_and_rerank(run)
        if cache is not None and fill_cache:
            cache.put(key, found)
    else:
        run.reranker_model = found.reranker_model
        run.fallback_used = found.fallback_used
    run.trace["rows_returned"] = found.rows_returned
    run.trace["retrieved"] = list(found.retrieved)
    return found


def _threshold(run: _Run) -> float:
    deps = run.deps
    return deps.threshold_fallback if run.fallback_used else deps.threshold_primary


def _hybrid(run: _Run) -> AskResponse:
    deps, s = run.deps, run.deps.settings
    question = run.request.question
    got = _retrieve(run, fill_cache=False)
    if not got.candidates:
        if _filters_applied(run.request.filters):
            return run.respond(Status.NO_MATCH_WITH_FILTERS)
        return run.respond(Status.REFUSED)
    if not got.ranked:
        return run.respond(Status.REFUSED)
    passages = _passages(got.chunks, got.judgments)
    refs = [p.ref for p in passages]

    threshold = _threshold(run)
    top_score = got.ranked[0][0]
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
        chunk_map: Mapping[str, ChunkRecord] = {c.chunk_id: c for c in got.chunks}
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


def _search_hybrid(run: _Run) -> SearchResponse:
    got = _retrieve(run, fill_cache=True)
    if not got.candidates:
        if _filters_applied(run.request.filters):
            return run.respond_search(Status.NO_MATCH_WITH_FILTERS)
        return run.respond_search(Status.REFUSED)
    passages = [
        _search_passage(i, chunk, got.judgments.get(chunk.judgment_id), score)
        for i, (score, _hit, chunk) in enumerate(got.ranked, 1)
    ]
    threshold = _threshold(run)
    top = got.ranked[0][0] if got.ranked else None
    status = Status.PASSAGES if top is not None and top >= threshold else Status.REFUSED
    return run.respond_search(status, passages=passages, top_score=top, threshold=threshold)


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
    except InvalidFilter as exc:
        # a bad filter value is the caller's mistake: a clean error, not a crash
        run.trace["error"] = f"invalid filter: {exc}"
        return run.respond(Status.ERROR, passages_note=f"{NOTE_INVALID_FILTER}: {exc}")
    except Exception as exc:
        log.exception("request %s failed", run.request_id)
        run.trace["error"] = f"{type(exc).__name__}: {exc}"
        return run.respond(Status.ERROR, passages_note=NOTE_ERROR)


def search(
    request: AskRequest,
    deps: Deps,
    *,
    request_id: str | None = None,
) -> SearchResponse:
    """The ranked passages for one request: `ask`'s route and retrieval, no LLM.

    Status: `passages` for a citation lookup, or when the top rerank score
    reaches the refusal threshold; `refused` when it does not (the passages are
    then the nearest ones) or when there are no candidates without filters;
    `no_match_with_filters`; `error`. No trace is written.
    """
    run = _Run(
        request=request,
        deps=deps,
        request_id=request_id or deps.new_request_id(),
        started_at=deps.clock(),
    )
    try:
        with run.timed("route"):
            run.route = deps.route(request.question)
        if run.route is Route.CITATION_LOOKUP:
            response = _search_lookup(run)
            if response is not None:
                return response
            run.route = Route.HYBRID
        return _search_hybrid(run)
    except InvalidFilter as exc:
        return run.respond_search(Status.ERROR, passages_note=f"{NOTE_INVALID_FILTER}: {exc}")
    except Exception:
        log.exception("search %s failed", run.request_id)
        return run.respond_search(Status.ERROR, passages_note=NOTE_ERROR)


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


# --- wiring the real components (the API's startup) ------------------------------


class _QueryEncoder:
    """`encode_query` over models.encoders.Embedder, one GPU call at a time."""

    def __init__(self, embedder: Any, lock: threading.Lock) -> None:
        self._embedder = embedder
        self._lock = lock

    def encode_query(self, text: str) -> Sequence[float]:
        with self._lock:
            return self._embedder.embed_query(text)


class _ScoreReranker:
    """`score(question, documents)` (input order) over a models.encoders reranker."""

    def __init__(self, reranker: Any, lock: threading.Lock | None = None) -> None:
        self._reranker = reranker
        self._lock = lock
        self.model = reranker.model_name

    def score(self, question: str, documents: Sequence[str]) -> list[float]:
        if self._lock is None:
            return self._score(question, documents)
        with self._lock:
            return self._score(question, documents)

    def _score(self, question: str, documents: Sequence[str]) -> list[float]:
        scores_fn = getattr(self._reranker, "scores", None)
        if callable(scores_fn):
            return [float(x) for x in scores_fn(question, documents)]
        out = [0.0] * len(documents)
        for r in self._reranker.rerank(question, documents, len(documents)):
            out[r.index] = float(r.score)
        return out


class _UnavailableReranker:
    """Stands in for a local reranker that failed to load: every call goes to the fallback."""

    def __init__(self, model: str, error: str) -> None:
        self.model = model
        self._error = error

    def score(self, question: str, documents: Sequence[str]) -> list[float]:
        raise RerankFailed(f"local reranker not loaded: {self._error}")


class PerThreadEngine:
    """One PostgresSearch (and so one connection) per worker thread.

    psycopg connections must not run transactions from several threads at
    once, and the API answers requests in a thread pool, so each thread gets
    its own connection on first use. The pool's thread count bounds them.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()

    def _engine(self) -> Any:
        engine = getattr(self._local, "engine", None)
        if engine is None:
            engine = self._factory()
            self._local.engine = engine
        return engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine(), name)


def build_deps(settings: Settings) -> Deps:  # pragma: no cover - needs the GPU and Postgres
    """Load the models, the engine for the active index version, and the LLM client once."""
    from nyaya import trace
    from nyaya.answer.llm_client import LLMClient
    from nyaya.link.aliases import find_citations
    from nyaya.models.encoders import (
        CohereFallbackReranker,
        Embedder,
        QwenReranker,
        startup_check,
    )
    from nyaya.search.expand import expand
    from nyaya.search.fusion import rrf
    from nyaya.search.postgres import PostgresSearch
    from nyaya.search.router import route
    from nyaya.store.db import connect
    from nyaya.zone.zoner import act_normalizer

    normalise_act = act_normalizer(settings)
    first = PostgresSearch(
        settings=settings,
        conn=connect(settings, autocommit=True),
        act_normalizer=normalise_act,
    )
    version = first.index_version
    thresholds = first.thresholds()
    first.conn.close()

    def make_engine() -> PostgresSearch:
        return PostgresSearch(
            index_version=version,
            settings=settings,
            conn=connect(settings, autocommit=True),
            act_normalizer=normalise_act,
        )

    gpu = threading.Lock()
    embedder = Embedder(settings)
    startup_check(settings, embedder)
    models_loaded = True
    try:
        reranker: Any = _ScoreReranker(QwenReranker(settings), gpu)
    except Exception as exc:
        log.exception("local reranker failed to load; the fallback will be used")
        reranker = _UnavailableReranker(settings.rerank.model, f"{type(exc).__name__}: {exc}")
        models_loaded = False
    try:
        fallback: Any = _ScoreReranker(CohereFallbackReranker.from_settings(settings))
    except Exception as exc:
        log.warning("fallback reranker not configured: %s", exc)
        fallback = None
    try:
        llm: Any = LLMClient.from_settings(settings)
    except LLMUnavailable as exc:
        log.warning("%s; answers will be passages only", exc)
        llm = None

    r = settings.retrieval

    def do_expand(hits: list[SearchHit], engine: Any) -> list[SearchHit]:
        return expand(hits, engine, r.expand_rank_offset, r.rerank_candidates)

    def open_conn() -> Any:
        return connect(settings)

    return Deps(
        settings=settings,
        engine=PerThreadEngine(make_engine),
        encoder=_QueryEncoder(embedder, gpu),
        reranker=reranker,
        fallback_reranker=fallback,
        llm=llm,
        route=lambda q: route(q, finder=find_citations),
        find_citations=find_citations,
        rrf=rrf,
        expand=do_expand,
        index_version=version,
        threshold_primary=float(thresholds["qwen"] if thresholds["qwen"] is not None else 1.0),
        threshold_fallback=float(thresholds["cohere"] if thresholds["cohere"] is not None else 1.0),
        write_trace=lambda record: trace.write_trace(open_conn, record),
        connect=open_conn,
        models_loaded=models_loaded,
    )
