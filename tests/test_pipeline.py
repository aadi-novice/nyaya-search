"""pipeline.ask with a fake engine, encoder, rerankers and LLM: every status path."""

from __future__ import annotations

import re
from datetime import date

import pytest

from nyaya.answer.llm_client import LLMBadJSON, LLMUnavailable
from nyaya.config import load_settings
from nyaya.contracts import (
    AskRequest,
    ChunkRecord,
    Era,
    Filters,
    JudgmentRecord,
    OpinionType,
    Route,
    SearchHit,
    Status,
    UnverifiedSentence,
    Zone,
)
from nyaya.pipeline import Deps, ask, error_response

SETTINGS = load_settings(env_file=None)

_SCR = re.compile(r"\[(\d{4})\]\s*(\d+)\s*S\.?\s*C\.?\s*R\.?\s*(\d+)")


def find_citations(text):
    return [(m.span(), "scr", f"scr:{m[1]}:{m[2]}:{m[3]}") for m in _SCR.finditer(text)]


def route(question):
    return Route.CITATION_LOOKUP if find_citations(question) else Route.HYBRID


QUOTE = "the right to a speedy trial is part of the right to life under Article 21"


def chunk(jid, zone, seq, text, opinion=OpinionType.MAJORITY, paras=("9",)):
    return ChunkRecord(
        chunk_id=f"{jid}#{zone.value}#{seq:03d}",
        judgment_id=jid,
        zone=zone,
        opinion_type=opinion if zone is Zone.JUDGMENT else None,
        para_labels=list(paras) if zone is Zone.JUDGMENT else [],
        para_refs=[],
        text=text,
        header=f"header of {jid}",
        char_start=seq * 100,
        char_end=seq * 100 + len(text),
        pages=[230 + seq],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch1",
    )


def judgment(path, title, era=Era.D):
    return JudgmentRecord(
        path=path,
        title=title,
        petitioner=title.split(" v. ")[0],
        respondent=title.split(" v. ")[1],
        judges=["A, J."],
        scr_citation=f"[2025] 1 S.C.R. {path.split('_')[2]}",
        neutral_citation="2025 INSC 31",
        cnr="ESCR010004242025",
        disposal="Dismissed",
        outcome="dismissed",
        decision_date=date(2025, 1, 2),
        decision_year=2025,
        era=era,
        has_pdf=True,
        pages=9,
    )


J1, J2, JC, JX = "2025_1_230_238", "2025_1_300_310", "2015_2_10_20", "2016_1_1_5"
JUDGMENTS = {
    J1: judgment(J1, "Najeeb v. Union of India"),
    J2: judgment(J2, "Rao v. State of Goa"),
    JC: judgment(JC, "Era C v. State", era=Era.C),
    JX: judgment(JX, "Bare v. State", era=Era.C),
}
CHUNKS = [
    chunk(J1, Zone.JUDGMENT, 1, f"17. It is settled that {QUOTE} of the Constitution."),
    chunk(J1, Zone.JUDGMENT, 2, "18. Bail is the rule and jail the exception."),
    chunk(J1, Zone.ISSUE, 0, "Whether delay in trial justifies bail."),
    chunk(J1, Zone.EDITORIAL, 1, "Held: delay justifies bail. [Para 17]"),
    chunk(J1, Zone.EDITORIAL, 2, "Held: bail is the rule. [Para 18]"),
    chunk(J2, Zone.JUDGMENT, 1, "5. Unrelated text about land.", OpinionType.DISSENTING),
    *[chunk(JC, Zone.EDITORIAL, i, f"HELD: point {i}.") for i in range(1, 5)],
    chunk(JX, Zone.JUDGMENT, 1, "1. Only a judgment chunk."),
    chunk(JX, Zone.JUDGMENT, 2, "2. Another one."),
]
BY_ID = {c.chunk_id: c for c in CHUNKS}
C1, C2, C3 = (f"{J1}#judgment#001", f"{J1}#judgment#002", f"{J2}#judgment#001")


class FakeEngine:
    def __init__(self, keyword=(C1, C3), vector=(C2, C1), links=None):
        self.kw = list(keyword)
        self.vec = list(vector)
        self.links = links or {}
        self.calls: list[tuple] = []

    def keyword(self, question, filters, k):
        self.calls.append(("keyword", question, filters, k))
        return [
            SearchHit(chunk_id=c, keyword_rank=i, fused_rank=i) for i, c in enumerate(self.kw, 1)
        ]

    def vector(self, qvec, filters, k, exact):
        self.calls.append(("vector", tuple(qvec), filters, k, exact))
        return [
            SearchHit(chunk_id=c, vector_rank=i, fused_rank=i) for i, c in enumerate(self.vec, 1)
        ]

    def get_chunks(self, ids):
        return [BY_ID[i] for i in ids if i in BY_ID]

    def get_judgment(self, path):
        return JUDGMENTS.get(path)

    def resolve_alias(self, canonical):
        return {"scr:2025:1:230": J1, "scr:2015:2:10": JC, "scr:2016:1:1": JX}.get(canonical)

    def links_from(self, chunk_ids):
        return {c: self.links[c] for c in chunk_ids if c in self.links}

    def chunks_for_judgment(self, path):
        return sorted((c for c in CHUNKS if c.judgment_id == path), key=lambda c: c.chunk_id)


class FakeEncoder:
    def encode_query(self, text):
        return [0.1, 0.2]


class FakeReranker:
    def __init__(self, model, scores=None, fail=False):
        self.model = model
        self.scores = scores or {}
        self.fail = fail
        self.calls = 0

    def score(self, question, documents):
        self.calls += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        out = []
        for d in documents:
            out.append(next((s for key, s in self.scores.items() if key in d), 0.1))
        return out


class FakeLLM:
    model = "fake-llm"

    def __init__(self, output=None, exc=None):
        self.output = output
        self.exc = exc
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system, user, timeout, history=()):
        self.calls.append((system, user))
        if self.exc is not None:
            raise self.exc
        return self.output


def rrf(lists, k):
    scores: dict[str, float] = {}
    for hits in lists:
        for h in hits:
            r = h.keyword_rank or h.vector_rank
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + r)
    order = sorted(scores, key=lambda c: -scores[c])
    return [SearchHit(chunk_id=c, fused_rank=i) for i, c in enumerate(order, 1)]


def expand(hits, engine):
    out = list(hits)
    links = engine.links_from([h.chunk_id for h in hits])
    for h in hits:
        for to in links.get(h.chunk_id, []):
            out.append(SearchHit(chunk_id=to, fused_rank=h.fused_rank + 0.5))
    return sorted(out, key=lambda h: h.fused_rank)


GOOD_OUTPUT = {
    "sentences": [
        {
            "sentence": "A speedy trial is part of Article 21.",
            "citations": [{"label": "P1", "quote": QUOTE}],
        },
        {
            "sentence": "Bail is always granted.",
            "citations": [
                {"label": "P1", "quote": "bail is always granted in every case whatsoever here"}
            ],
        },
    ],
    "gaps": ["The passages do not address costs."],
}


def make_deps(**kw):
    traces = []
    defaults = {
        "settings": SETTINGS,
        "engine": FakeEngine(),
        "encoder": FakeEncoder(),
        "reranker": FakeReranker("qwen", {"speedy trial": 0.9, "Bail is the rule": 0.5}),
        "fallback_reranker": FakeReranker("cohere", {"speedy trial": 0.3}),
        "llm": FakeLLM(GOOD_OUTPUT),
        "route": route,
        "find_citations": find_citations,
        "rrf": rrf,
        "expand": expand,
        "index_version": "v2025a",
        "threshold_primary": 0.35,
        "threshold_fallback": 0.20,
        "write_trace": traces.append,
    }
    defaults.update(kw)
    deps = Deps(**defaults)
    return deps, traces


def req(question="Is speedy trial part of Article 21?", **filters):
    return AskRequest(question=question, filters=Filters(**filters) if filters else None)


# --- answered -------------------------------------------------------------------


def test_answered_path():
    deps, traces = make_deps()
    resp = ask(req(), deps)
    assert resp.status is Status.ANSWERED
    assert resp.route is Route.HYBRID
    assert [s.sentence for s in resp.answer] == ["A speedy trial is part of Article 21."]
    cit = resp.answer[0].citations[0]
    assert (cit.chunk_id, cit.passage_label, cit.case_name) == (
        C1,
        "P1",
        "Najeeb v. Union of India",
    )
    assert cit.scr_citation == "[2025] 1 S.C.R. 230" and cit.paras == ["9"]
    assert [u.reason.value for u in resp.unverified] == ["quote_not_found"]
    assert resp.gaps == ["The passages do not address costs."]
    assert resp.notices == SETTINGS.notices
    assert resp.passages == [] and resp.nearest_passages == []
    v = resp.versions
    assert (v.index_version, v.llm_model, v.reranker_model) == ("v2025a", "fake-llm", "qwen")
    assert not v.reranker_fallback_used and v.prompt_version == "answer_v1"
    assert {
        "embed",
        "keyword",
        "vector",
        "fusion",
        "expand",
        "rerank",
        "llm",
        "check",
        "total",
    } <= set(resp.timings_ms)
    # the prompt holds the passages in rerank order: C1 (0.9) first
    _system, user = deps.llm.calls[0]
    assert user.index(QUOTE) < user.index("Bail is the rule")
    (t,) = traces
    assert t.request_id == resp.request_id and t.status == "answered"
    assert t.unverified_count == 1 and t.gaps_count == 1
    assert [b["kind"] for b in t.rule_breaks] == ["quote_not_found"]  # gaps never break rules
    assert t.top_score == pytest.approx(0.9) and t.threshold == 0.35
    assert t.rows_returned["keyword"] == {"requested": 50, "returned": 2}
    assert t.retrieved[0]["chunk_id"] == C1


def test_unverified_sentences_carry_no_citation_details():
    deps, _ = make_deps()
    resp = ask(req(), deps)
    dumped = resp.model_dump()
    for u in dumped["unverified"]:
        assert set(u) == set(UnverifiedSentence.model_fields)


def test_filters_are_passed_and_reported():
    deps, _ = make_deps()
    resp = ask(req(year_from=2020, judge="A"), deps)
    assert resp.filters_applied == {"year_from": 2020, "judge": "A"}
    kinds = [c[0] for c in deps.engine.calls]
    assert kinds == ["keyword", "vector"]
    assert all(c[2] == Filters(year_from=2020, judge="A") for c in deps.engine.calls)
    assert deps.engine.calls[1][4] is False  # HNSW, not exact


def test_link_expansion_feeds_the_reranker():
    engine = FakeEngine(keyword=(C3,), vector=(C3,), links={C3: [C1]})
    deps, _ = make_deps(engine=engine)
    resp = ask(req(), deps)
    assert resp.status is Status.ANSWERED
    assert resp.answer[0].citations[0].chunk_id == C1


def test_candidates_cut_to_rerank_candidates(monkeypatch):
    settings = SETTINGS.model_copy(
        update={"retrieval": SETTINGS.retrieval.model_copy(update={"rerank_candidates": 1})}
    )
    rr = FakeReranker("qwen", {"speedy trial": 0.9})
    deps, _ = make_deps(
        settings=settings, reranker=rr, engine=FakeEngine(keyword=(C3, C1), vector=())
    )
    resp = ask(req(), deps)
    # only C3 (rank 1) reaches the reranker; it scores 0.1 < 0.35
    assert resp.status is Status.REFUSED
    assert [p.chunk_id for p in resp.nearest_passages] == [C3]


# --- citation lookup ------------------------------------------------------------


def test_lookup_era_d_issue_then_editorials():
    deps, traces = make_deps()
    resp = ask(req("[2025] 1 S.C.R. 230"), deps)
    assert resp.status is Status.PASSAGES and resp.route is Route.CITATION_LOOKUP
    assert [p.chunk_id for p in resp.passages] == [
        f"{J1}#issue#000",
        f"{J1}#editorial#001",
        f"{J1}#editorial#002",
    ]
    assert resp.passages_note == "judgment found by citation"
    assert deps.llm.calls == [] and deps.reranker.calls == 0
    assert deps.engine.calls == []
    assert traces[0].route == "citation_lookup" and traces[0].status == "passages"


def test_lookup_era_c_first_three_editorials():
    deps, _ = make_deps()
    resp = ask(req("What did [2015] 2 SCR 10 hold?"), deps)
    assert [p.chunk_id for p in resp.passages] == [f"{JC}#editorial#00{i}" for i in (1, 2, 3)]
    assert all(p.zone is Zone.EDITORIAL for p in resp.passages)


def test_lookup_without_issue_or_editorial_gives_first_judgment_chunk():
    deps, _ = make_deps()
    resp = ask(req("[2016] 1 SCR 1"), deps)
    assert [p.chunk_id for p in resp.passages] == [f"{JX}#judgment#001"]


def test_unresolved_citation_falls_back_to_hybrid():
    deps, _ = make_deps()
    resp = ask(req("[1999] 3 S.C.R. 999 speedy trial"), deps)
    assert resp.route is Route.HYBRID
    assert resp.status is Status.ANSWERED
    assert deps.engine.calls[0][1] == "[1999] 3 S.C.R. 999 speedy trial"


# --- LLM failure ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("llm", "note"),
    [
        (FakeLLM(exc=LLMUnavailable("RateLimitError")), "no summary generated (LLM unavailable)"),
        (None, "no summary generated (LLM unavailable)"),
        (
            FakeLLM(exc=LLMBadJSON("invalid JSON", "x")),
            "no summary generated (LLM reply was invalid)",
        ),
    ],
)
def test_llm_failure_returns_passages(llm, note):
    deps, traces = make_deps(llm=llm)
    resp = ask(req(), deps)
    assert resp.status is Status.PASSAGES
    assert resp.passages_note == note
    assert [p.chunk_id for p in resp.passages][:2] == [C1, C2]
    assert resp.answer == [] and resp.unverified == []
    assert traces[0].llm_error


# --- refused --------------------------------------------------------------------


def test_refusal_gate_below_threshold():
    deps, traces = make_deps(reranker=FakeReranker("qwen", {"speedy trial": 0.2}))
    resp = ask(req(), deps)
    assert resp.status is Status.REFUSED
    assert resp.nearest_passages[0].chunk_id == C1
    assert deps.llm.calls == []
    assert traces[0].top_score == pytest.approx(0.2)


def test_refused_when_no_sentence_verifies():
    out = {"sentences": [{"sentence": "No quotes here.", "citations": []}], "gaps": []}
    deps, traces = make_deps(llm=FakeLLM(out))
    resp = ask(req(), deps)
    assert resp.status is Status.REFUSED
    assert resp.answer == []
    assert [u.reason.value for u in resp.unverified] == ["no_citation"]
    assert resp.nearest_passages[0].chunk_id == C1
    assert traces[0].rule_breaks == [
        {"sentence_index": 0, "kind": "no_citation", "label": None, "detail": None}
    ]


def test_no_candidates_without_filters_is_refused():
    deps, _ = make_deps(engine=FakeEngine(keyword=(), vector=()))
    resp = ask(req(), deps)
    assert resp.status is Status.REFUSED and resp.nearest_passages == []


def test_no_match_with_filters():
    deps, traces = make_deps(engine=FakeEngine(keyword=(), vector=()))
    resp = ask(req(year_from=1950, year_to=1951), deps)
    assert resp.status is Status.NO_MATCH_WITH_FILTERS
    assert deps.reranker.calls == 0
    assert traces[0].status == "no_match_with_filters"


# --- reranker fallback and errors -----------------------------------------------


def test_fallback_reranker_and_its_threshold():
    deps, traces = make_deps(reranker=FakeReranker("qwen", fail=True))
    resp = ask(req(), deps)
    # cohere scores the speedy-trial chunk 0.3: above 0.20, so answered
    assert resp.status is Status.ANSWERED
    assert resp.versions.reranker_fallback_used is True
    assert resp.versions.reranker_model == "cohere"
    assert traces[0].reranker_fallback_used is True and traces[0].threshold == 0.20


def test_both_rerankers_failing_is_an_error():
    deps, traces = make_deps(reranker=FakeReranker("qwen", fail=True), fallback_reranker=None)
    resp = ask(req(), deps)
    assert resp.status is Status.ERROR
    assert resp.notices == SETTINGS.notices
    assert traces[0].status == "error"


def test_unexpected_exception_is_an_error_and_traced():
    class Broken(FakeEngine):
        def keyword(self, *a):
            raise RuntimeError("connection reset")

    deps, traces = make_deps(engine=Broken())
    resp = ask(req(), deps)
    assert resp.status is Status.ERROR and resp.passages_note
    assert traces[0].status == "error"


def test_trace_failure_never_fails_the_request():
    def broken_trace(record):
        raise RuntimeError("db down")

    deps, _ = make_deps(write_trace=broken_trace)
    assert ask(req(), deps).status is Status.ANSWERED


def test_error_response_for_timeout_uses_given_request_id():
    deps, traces = make_deps()
    resp = error_response(req(), deps, request_id="rid-1", note="request timed out")
    assert resp.status is Status.ERROR and resp.request_id == "rid-1"
    assert resp.passages_note == "request timed out"
    assert traces[0].request_id == "rid-1"


def test_request_id_and_start_time_can_be_supplied():
    deps, _ = make_deps()
    resp = ask(req(), deps, request_id="abc", started_at=0.0)
    assert resp.request_id == "abc"


def test_fallback_failing_too_is_an_error():
    deps, _ = make_deps(
        reranker=FakeReranker("qwen", fail=True), fallback_reranker=FakeReranker("c", fail=True)
    )
    resp = ask(req(), deps)
    assert resp.status is Status.ERROR and resp.versions.reranker_fallback_used


def test_reranker_score_count_mismatch_is_an_error():
    class Short(FakeReranker):
        def score(self, question, documents):
            return [0.9]

    deps, _ = make_deps(reranker=Short("qwen"))
    assert ask(req(), deps).status is Status.ERROR


def test_hits_whose_chunks_are_missing_are_dropped():
    deps, _ = make_deps(engine=FakeEngine(keyword=("gone#judgment#001",), vector=()))
    resp = ask(req(), deps)
    assert resp.status is Status.REFUSED and resp.nearest_passages == []


def test_lookup_chunks_rules():
    from nyaya.pipeline import lookup_chunks

    issue = chunk(JC, Zone.ISSUE, 0, "Issue.")
    assert lookup_chunks(JUDGMENTS[JC], [issue], 3) == [issue]
    eds = [c for c in CHUNKS if c.judgment_id == JC]
    assert lookup_chunks(None, eds, 2) == eds[:2]


# --- integration additions ------------------------------------------------------


def test_invalid_filter_is_a_clean_error_with_trace():
    from nyaya.search.postgres import InvalidFilter

    class BadFilterEngine(FakeEngine):
        def keyword(self, question, filters, k):
            raise InvalidFilter("unknown outcome filter 'won'")

    deps, traces = make_deps(engine=BadFilterEngine())
    resp = ask(AskRequest(question="bail", filters=Filters(outcome="won")), deps)
    assert resp.status is Status.ERROR
    assert "invalid filter" in (resp.passages_note or "")
    assert traces[-1].error.startswith("invalid filter")


def test_unexpected_exception_records_error_in_trace():
    class Boom(FakeEngine):
        def keyword(self, question, filters, k):
            raise RuntimeError("db gone")

    deps, traces = make_deps(engine=Boom())
    resp = ask(AskRequest(question="bail"), deps)
    assert resp.status is Status.ERROR
    assert traces[-1].error == "RuntimeError: db gone"


def test_deps_health_and_versions():
    deps, _ = make_deps(fallback_reranker=None, llm=None)
    assert deps.health() == {
        "active_version": "v2025a",
        "models_loaded": True,
        "fallback_configured": False,
        "llm_configured": False,
    }
    v = deps.versions()
    assert v.index_version == "v2025a"
    assert v.reranker_model == "qwen"
    assert v.reranker_fallback_used is False


def test_score_reranker_adapter_keeps_input_order():
    import threading

    from nyaya.models.encoders import RerankResult
    from nyaya.pipeline import _ScoreReranker

    class TopN:
        model_name = "cohere/x"

        def rerank(self, q, docs, top_n):
            assert top_n == len(docs)
            return [RerankResult(2, 0.9), RerankResult(0, 0.5), RerankResult(1, 0.1)]

    class Scores:
        model_name = "qwen/x"

        def scores(self, q, docs):
            return [0.3, 0.2]

    a = _ScoreReranker(TopN())
    assert a.model == "cohere/x"
    assert a.score("q", ["a", "b", "c"]) == [0.5, 0.1, 0.9]
    assert _ScoreReranker(Scores(), threading.Lock()).score("q", ["a", "b"]) == [0.3, 0.2]


def test_unavailable_reranker_raises_so_the_fallback_runs():
    from nyaya.pipeline import _UnavailableReranker

    deps, _ = make_deps(reranker=_UnavailableReranker("qwen", "no CUDA"))
    resp = ask(AskRequest(question="speedy trial bail"), deps)
    assert resp.versions.reranker_fallback_used is True


def test_per_thread_engine_gives_each_thread_its_own_engine():
    import threading

    from nyaya.pipeline import PerThreadEngine

    made = []

    def factory():
        e = FakeEngine()
        made.append(e)
        return e

    pte = PerThreadEngine(factory)
    pte.get_judgment(J1)
    pte.get_judgment(J1)
    t = threading.Thread(target=lambda: pte.get_judgment(J1))
    t.start()
    t.join()
    assert len(made) == 2


def test_act_normalizer_maps_aliases():
    from nyaya.zone.zoner import act_normalizer

    norm = act_normalizer(SETTINGS)
    assert norm("Indian Penal Code, 1860") == norm("IPC")
