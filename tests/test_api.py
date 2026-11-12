"""API tests (design file 37, tests entry 48) with a fake backend.

The fake stands in for the search engine, the LLM and the pipeline, so every
status path is reachable without a database or a model.
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nyaya.api.app import WEB_DIR, create_app
from nyaya.config import load_settings
from nyaya.contracts import (
    AnswerSentence,
    AskRequest,
    AskResponse,
    ChunkRecord,
    Citation,
    JudgmentRecord,
    OpinionType,
    PassageRef,
    Route,
    SearchPassage,
    SearchResponse,
    Status,
    UnverifiedReason,
    UnverifiedSentence,
    Versions,
    Zone,
)

NOTICES = ["This is not legal advice.", "Later overruling of a judgment is not tracked."]
CITATION_DETAIL_KEYS = {"case_name", "scr_citation", "quote", "chunk_id", "pages", "paras"}


def versions() -> Versions:
    return Versions(
        index_version="v2025",
        config_version="cfg1",
        prompt_version="answer_v1",
        llm_model="fake-llm",
        reranker_model="fake-reranker",
        reranker_fallback_used=False,
        embed_model="fake-embed",
    )


def passage(seq: int = 14, zone: Zone = Zone.JUDGMENT) -> PassageRef:
    return PassageRef(
        chunk_id=f"2025_1_230_238#{zone.value}#{seq:03d}",
        judgment_id="2025_1_230_238",
        case_name="Jit Vinayak Arolkar v. State of Goa",
        scr_citation="[2025] 1 S.C.R. 230",
        neutral_citation="2025 INSC 31",
        paras=["13"],
        pages=[236],
        zone=zone,
        opinion_type=OpinionType.MAJORITY if zone is Zone.JUDGMENT else None,
    )


def citation() -> Citation:
    return Citation(
        **passage().model_dump(),
        passage_label="P1",
        quote="the High Court ought to have quashed the proceedings as the dispute is civil",
    )


def response(status: Status, route: Route = Route.HYBRID, **fields: Any) -> AskResponse:
    base: dict[str, Any] = {
        "request_id": "req-1",
        "status": status,
        "notices": NOTICES,
        "route": route,
        "filters_applied": {},
        "versions": versions(),
        "timings_ms": {"search": 12.0, "rerank": 30.0, "llm": 900.0},
    }
    return AskResponse(**(base | fields))


UNVERIFIED = UnverifiedSentence(
    sentence="The landlord must refund the deposit.",
    labels=["P9"],
    reason=UnverifiedReason.UNKNOWN_LABEL,
)

ANSWERED = response(
    Status.ANSWERED,
    answer=[
        AnswerSentence(
            sentence="A criminal case over a civil dispute may be quashed.",
            citations=[citation()],
        )
    ],
    unverified=[UNVERIFIED],
    gaps=["The passages do not address limitation."],
)
PASSAGES_LOOKUP = response(
    Status.PASSAGES,
    route=Route.CITATION_LOOKUP,
    passages=[passage(1, Zone.ISSUE), passage(2, Zone.EDITORIAL)],
    passages_note="judgment found by citation",
)
PASSAGES_LLM_DOWN = response(
    Status.PASSAGES,
    passages=[passage(14), passage(15)],
    passages_note="no summary generated (LLM unavailable)",
)
REFUSED = response(Status.REFUSED, nearest_passages=[passage(14)], unverified=[UNVERIFIED])
NO_MATCH = response(
    Status.NO_MATCH_WITH_FILTERS,
    filters_applied={"year_from": 2018, "year_to": 2026, "act": "Code of Criminal Procedure"},
)

JUDGMENT = JudgmentRecord(
    path="2025_1_230_238",
    title="Jit Vinayak Arolkar v. State of Goa",
    petitioner="Jit Vinayak Arolkar",
    respondent="State of Goa",
    judges=["Abhay S. Oka", "Augustine George Masih"],
    author_judge="Abhay S. Oka",
    scr_citation="[2025] 1 S.C.R. 230",
    neutral_citation="2025 INSC 31",
    cnr="SCIN010000002025",
    disposal="Appeal(s) allowed",
    outcome="allowed",
    decision_date=date(2025, 1, 6),
    decision_year=2025,
    era=None,
    has_pdf=True,
    pages=9,
    clean_text="Header text. 13. The dispute is civil in nature.",
)
CHUNK = ChunkRecord(
    chunk_id="2025_1_230_238#judgment#014",
    judgment_id="2025_1_230_238",
    zone=Zone.JUDGMENT,
    para_labels=["13"],
    para_refs=[],
    text="13. The dispute is civil in nature.",
    header="Jit Vinayak Arolkar v. State of Goa",
    char_start=13,
    char_end=48,
    pages=[236],
    flags=[],
    clean_text_version="ct1",
    chunker_version="ch1",
)


def search_response(status: Status, **fields: Any) -> SearchResponse:
    base: dict[str, Any] = {
        "request_id": "srch-1",
        "status": status,
        "notices": NOTICES,
        "route": Route.HYBRID,
        "filters_applied": {},
        "versions": versions(),
        "timings_ms": {"embed": 5.0, "rerank": 30.0, "total": 40.0},
    }
    return SearchResponse(**(base | fields))


SEARCHED = search_response(
    Status.PASSAGES,
    passages=[
        SearchPassage(
            **passage(14).model_dump(),
            rank=1,
            rerank_score=0.91,
            text="13. The dispute is civil in nature.",
            decision_date=date(2025, 1, 6),
        )
    ],
    top_score=0.91,
    threshold=0.35,
)


class FakeBackend:
    """Returns canned responses chosen by the question text."""

    def __init__(self) -> None:
        self.by_question: dict[str, AskResponse] = {
            "civil fir": ANSWERED,
            "2025 INSC 31": PASSAGES_LOOKUP,
            "llm down": PASSAGES_LLM_DOWN,
            "unanswerable": REFUSED,
            "filtered": NO_MATCH,
        }
        self.release = threading.Event()
        self.asked: list[AskRequest] = []
        self.searched: list[AskRequest] = []
        self.traces: list[tuple[AskRequest, AskResponse, str | None]] = []
        self.fail_trace = False

    def ask(self, request: AskRequest) -> AskResponse:
        self.asked.append(request)
        if request.question == "slow":
            self.release.wait(5)
            return ANSWERED
        if request.question == "boom":
            raise RuntimeError("database went away")
        return self.by_question[request.question]

    def search(self, request: AskRequest) -> SearchResponse:
        self.searched.append(request)
        if request.question == "slow":
            self.release.wait(5)
        if request.question == "boom":
            raise RuntimeError("database went away")
        if request.question == "filtered":
            filters = request.filters.model_dump(exclude_none=True) if request.filters else {}
            return search_response(Status.NO_MATCH_WITH_FILTERS, filters_applied=filters)
        return SEARCHED

    def get_judgment(self, path: str) -> JudgmentRecord | None:
        return JUDGMENT if path == JUDGMENT.path else None

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        return CHUNK if chunk_id == CHUNK.chunk_id else None

    def health(self) -> dict[str, Any]:
        return {"active_version": "v2025", "models_loaded": True, "fallback_configured": False}

    def versions(self) -> Versions:
        return versions()

    def write_trace(self, request: AskRequest, resp: AskResponse, error: str | None) -> None:
        if self.fail_trace:
            raise OSError("trace table missing")
        self.traces.append((request, resp, error))


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(backend: FakeBackend):
    app = create_app(backend, settings=load_settings(env_file=None), request_timeout_s=0.3)
    with TestClient(app) as c:
        yield c
    backend.release.set()


def ask(client: TestClient, question: str, **body: Any) -> dict[str, Any]:
    r = client.post("/ask", json={"question": question, **body})
    assert r.status_code == 200, r.text
    AskResponse.model_validate(r.json())  # the wire format is the contract
    return r.json()


# --- POST /ask: every status path ---------------------------------------------------


def test_answered(client: TestClient) -> None:
    data = ask(client, "civil fir")
    assert data["status"] == "answered"
    cit = data["answer"][0]["citations"][0]
    assert cit["case_name"] == "Jit Vinayak Arolkar v. State of Goa"
    assert cit["quote"].startswith("the High Court")
    assert data["notices"] == NOTICES


def test_answered_gaps_pass_through_without_changing_status(client: TestClient) -> None:
    data = ask(client, "civil fir")
    assert data["gaps"] == ["The passages do not address limitation."]
    assert data["status"] == "answered"
    assert data["answer"], "gaps never turn a verified answer into a refusal"


def test_unverified_sentences_carry_no_citation_details(client: TestClient) -> None:
    for question in ("civil fir", "unanswerable"):
        data = ask(client, question)
        assert data["unverified"]
        for item in data["unverified"]:
            assert set(item) == {"sentence", "labels", "reason"}
            assert not CITATION_DETAIL_KEYS & set(item)
            assert "citations" not in item


def test_passages_from_citation_lookup(client: TestClient) -> None:
    data = ask(client, "2025 INSC 31")
    assert data["status"] == "passages"
    assert data["route"] == "citation_lookup"
    assert [p["zone"] for p in data["passages"]] == ["issue", "editorial"]
    assert data["passages_note"] == "judgment found by citation"
    assert data["answer"] == []


def test_passages_from_llm_failure(client: TestClient) -> None:
    data = ask(client, "llm down")
    assert data["status"] == "passages"
    assert data["route"] == "hybrid"
    assert data["passages_note"] == "no summary generated (LLM unavailable)"
    assert len(data["passages"]) == 2


def test_refused(client: TestClient) -> None:
    data = ask(client, "unanswerable")
    assert data["status"] == "refused"
    assert data["answer"] == []
    assert data["nearest_passages"][0]["chunk_id"] == "2025_1_230_238#judgment#014"


def test_no_match_with_filters(client: TestClient, backend: FakeBackend) -> None:
    filters = {"year_from": 2018, "year_to": 2026, "act": "Code of Criminal Procedure"}
    data = ask(client, "filtered", filters=filters)
    assert data["status"] == "no_match_with_filters"
    assert data["filters_applied"] == filters
    assert backend.asked[-1].filters is not None
    assert backend.asked[-1].filters.act == "Code of Criminal Procedure"


def test_timeout_returns_error_and_writes_trace(client: TestClient, backend: FakeBackend) -> None:
    data = ask(client, "slow", filters={"judge": "Abhay S. Oka"})
    assert data["status"] == "error"
    assert data["answer"] == [] and data["passages"] == []
    assert data["filters_applied"] == {"judge": "Abhay S. Oka"}
    assert data["notices"][0].startswith("The request took longer than")
    assert data["notices"][1:] == load_settings(env_file=None).notices
    assert "total" in data["timings_ms"]
    assert data["versions"]["index_version"] == "v2025"
    assert len(backend.traces) == 1
    request, resp, error = backend.traces[0]
    assert request.question == "slow"
    assert resp.status is Status.ERROR
    assert resp.request_id == data["request_id"]
    assert error is not None and "timeout" in error


def test_pipeline_exception_returns_error_and_writes_trace(
    client: TestClient, backend: FakeBackend
) -> None:
    data = ask(client, "boom")
    assert data["status"] == "error"
    assert "database went away" not in " ".join(data["notices"])  # no internals on the page
    assert backend.traces[0][2] is not None and "database went away" in backend.traces[0][2]


def test_trace_failure_never_fails_the_request(client: TestClient, backend: FakeBackend) -> None:
    backend.fail_trace = True
    assert ask(client, "boom")["status"] == "error"


def test_successful_requests_leave_tracing_to_the_pipeline(
    client: TestClient, backend: FakeBackend
) -> None:
    ask(client, "civil fir")
    assert backend.traces == []


def test_invalid_request_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "extra": 1}).status_code == 422
    bad_years = {"question": "x", "filters": {"year_from": 2026, "year_to": 2018}}
    assert client.post("/ask", json=bad_years).status_code == 422


# --- POST /search -------------------------------------------------------------------


def search(client: TestClient, question: str, **body: Any) -> dict[str, Any]:
    r = client.post("/search", json={"question": question, **body})
    assert r.status_code == 200, r.text
    SearchResponse.model_validate(r.json())  # the wire format is the contract
    return r.json()


def test_search_returns_ranked_passages(client: TestClient, backend: FakeBackend) -> None:
    data = search(client, "civil fir")
    assert data["status"] == "passages"
    (p,) = data["passages"]
    assert (p["rank"], p["rerank_score"], p["chunk_id"]) == (1, 0.91, CHUNK.chunk_id)
    assert p["text"].startswith("13.") and p["decision_date"] == "2025-01-06"
    assert data["threshold"] == 0.35 and data["notices"] == NOTICES
    assert backend.asked == []  # /search never reaches the answer path
    assert backend.traces == []


def test_search_passes_filters(client: TestClient, backend: FakeBackend) -> None:
    filters = {"year_from": 2018, "year_to": 2026, "act": "Code of Criminal Procedure"}
    data = search(client, "filtered", filters=filters)
    assert data["status"] == "no_match_with_filters"
    assert data["filters_applied"] == filters
    assert backend.searched[-1].filters is not None
    assert backend.searched[-1].filters.year_from == 2018


def test_search_timeout_and_failure_are_clean_errors(
    client: TestClient, backend: FakeBackend
) -> None:
    data = search(client, "slow", filters={"judge": "Abhay S. Oka"})
    assert data["status"] == "error" and data["passages"] == []
    assert data["notices"][0].startswith("The request took longer than")
    assert data["filters_applied"] == {"judge": "Abhay S. Oka"}
    backend.release.set()
    data = search(client, "boom")
    assert data["status"] == "error"
    assert "database went away" not in " ".join(data["notices"])
    assert backend.traces == []  # /search is never traced


def test_search_rejects_invalid_requests(client: TestClient) -> None:
    assert client.post("/search", json={"question": ""}).status_code == 422
    bad_years = {"question": "x", "filters": {"year_from": 2026, "year_to": 2018}}
    assert client.post("/search", json=bad_years).status_code == 422


# --- GET /judgment/{path} ------------------------------------------------------------


def test_judgment_with_highlight(client: TestClient) -> None:
    r = client.get(
        "/judgment/2025_1_230_238",
        params={"highlight": CHUNK.chunk_id},
        headers={"accept": "application/json"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == JUDGMENT.title
    assert data["highlight"] == {"chunk_id": CHUNK.chunk_id, "char_start": 13, "char_end": 48}
    JudgmentRecord.model_validate({k: v for k, v in data.items() if k != "highlight"})


def test_judgment_without_highlight(client: TestClient) -> None:
    data = client.get("/judgment/2025_1_230_238").json()
    assert data["highlight"] is None


def test_judgment_not_found(client: TestClient) -> None:
    assert client.get("/judgment/9999_1_1_1").status_code == 404


def test_judgment_unknown_highlight(client: TestClient) -> None:
    r = client.get("/judgment/2025_1_230_238", params={"highlight": "2025_1_230_238#judgment#999"})
    assert r.status_code == 404


def test_judgment_highlight_from_another_judgment(client: TestClient, backend: FakeBackend) -> None:
    other = CHUNK.model_copy(update={"judgment_id": "2024_1_1_10"})
    backend.get_chunk = lambda chunk_id: other  # type: ignore[method-assign]
    r = client.get("/judgment/2025_1_230_238", params={"highlight": CHUNK.chunk_id})
    assert r.status_code == 404


def test_judgment_link_from_a_browser_gets_the_page(client: TestClient) -> None:
    r = client.get(
        "/judgment/2025_1_230_238",
        params={"highlight": CHUNK.chunk_id},
        headers={"accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "app.js" in r.text


# --- GET /health and the static page --------------------------------------------------


def test_health(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["active_version"] == "v2025"
    assert data["models_loaded"] is True
    assert data["fallback_configured"] is False
    assert data["notices"] == load_settings(env_file=None).notices


def test_static_page(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Nyaya Search" in page.text
    for asset in ("app.js", "style.css"):
        assert asset in page.text
        r = client.get(f"/{asset}")
        assert r.status_code == 200 and r.text


def test_page_has_no_external_resources() -> None:
    for name in ("index.html", "app.js", "style.css"):
        text = (WEB_DIR / name).read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, name


def test_page_never_renders_response_text_as_html() -> None:
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "outerHTML" not in js and "insertAdjacentHTML" not in js


def test_page_renders_unverified_without_citation_fields() -> None:
    """The unverified renderer reads only sentence, labels and reason."""
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    start = js.index("function renderUnverified")
    body = js[start : js.index("\n}\n", start)]
    for field in ("case_name", "scr_citation", "quote", "chunk_id", "pages", "paras"):
        assert field not in body


def test_main_binds_localhost(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    import nyaya.api.app as app_module

    seen: dict[str, Any] = {}
    monkeypatch.setattr(app_module, "build_backend", lambda settings: backend)
    monkeypatch.setattr(app_module.uvicorn, "run", lambda app, **kw: seen.update(kw))
    app_module.main(["--port", "8123"])
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8123


def test_error_response_survives_a_versions_failure(
    client: TestClient, backend: FakeBackend
) -> None:
    def broken() -> Versions:
        raise RuntimeError("no active index")

    backend.versions = broken  # type: ignore[method-assign]
    data = ask(client, "boom")
    assert data["status"] == "error"
    assert data["versions"]["index_version"] == "unknown"
    assert data["versions"]["config_version"] == load_settings(env_file=None).versions.config


def test_pipeline_backend_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """PipelineBackend wires nyaya.pipeline, the engine and nyaya.trace."""
    import sys
    import types

    from nyaya.api.app import PipelineBackend

    calls: list[tuple[str, Any]] = []

    class Engine:
        def get_judgment(self, path: str) -> JudgmentRecord | None:
            return JUDGMENT if path == JUDGMENT.path else None

        def get_chunks(self, ids: list[str]) -> list[ChunkRecord]:
            return [CHUNK] if ids == [CHUNK.chunk_id] else []

    class Deps:
        engine = Engine()
        connect = "connect-fn"

        def health(self) -> dict[str, Any]:
            return {"active_version": "v2025"}

        def versions(self) -> Versions:
            return versions()

    deps = Deps()
    pipeline = types.ModuleType("nyaya.pipeline")
    pipeline.build_deps = lambda settings: deps  # type: ignore[attr-defined]
    pipeline.ask = lambda request, d: (calls.append(("ask", d)), ANSWERED)[1]  # type: ignore[attr-defined]
    pipeline.search = lambda request, d: SEARCHED  # type: ignore[attr-defined]
    pipeline.RetrievalCache = lambda: "cache"  # type: ignore[attr-defined]
    trace = types.ModuleType("nyaya.trace")
    trace.record_from_response = lambda req, resp, error=None: ("record", error)  # type: ignore[attr-defined]
    trace.write_trace = lambda connect, record: calls.append(("trace", connect, record))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyaya.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "nyaya.trace", trace)
    import nyaya

    monkeypatch.setattr(nyaya, "pipeline", pipeline, raising=False)
    monkeypatch.setattr(nyaya, "trace", trace, raising=False)

    b = PipelineBackend(load_settings(env_file=None))
    req = AskRequest(question="q")
    assert b.ask(req) is ANSWERED
    assert b.search(req) is SEARCHED
    assert deps.retrieval_cache == "cache"
    assert b.get_judgment(JUDGMENT.path) is JUDGMENT
    assert b.get_judgment("nope") is None
    assert b.get_chunk(CHUNK.chunk_id) is CHUNK
    assert b.get_chunk("nope") is None
    assert b.health() == {"active_version": "v2025"}
    assert b.versions() == versions()
    b.write_trace(req, ANSWERED, "timeout")
    assert calls == [("ask", deps), ("trace", "connect-fn", ("record", "timeout"))]
