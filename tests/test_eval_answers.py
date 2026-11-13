"""`nyaya-eval answers` with the real pipeline over fakes, a fake judge and no network."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from nyaya.answer.llm_client import LLMUnavailable
from nyaya.contracts import Status
from nyaya.eval import answers as A
from nyaya.eval.datasets import AdversarialItem, DevItem, Gold, write_jsonl
from nyaya.eval.judge import ERROR, Verdict
from nyaya.eval.ratelimit import PausableClock, RateLimitExhausted
from nyaya.models.encoders import document_text
from test_pipeline import C1, C2, C3, CHUNKS, GOOD_OUTPUT, J1, J2, SETTINGS, FakeLLM, make_deps

REQ = httpx.Request("POST", "http://llm/v1/chat/completions")


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class FakeJudge:
    """Label by the first rule whose key is in the passage; `default` otherwise."""

    model = "fake-judge"

    def __init__(self, rules=None, default="supported", exhaust_after=None):
        self.rules = rules or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.exhaust_after = exhaust_after

    def judge(self, sentence, passage):
        if self.exhaust_after is not None and len(self.calls) >= self.exhaust_after:
            raise RateLimitExhausted("judge quota")
        self.calls.append((sentence, passage))
        for key, label in self.rules.items():
            if key in passage:
                return Verdict(label, "r")
        return Verdict(self.default, "r")


def runner(judge=None, **deps_kw):
    deps, traces = make_deps(**deps_kw)
    clock = PausableClock(FakeTime().clock, lambda s: None)
    ask_fn, llm = A.instrument(deps, SETTINGS, clock)
    return A.Runner(ask_fn, deps.engine.get_chunks, judge or FakeJudge(), llm, None), traces


def q(qid="dev-1", set_="dev", gold=((J1, ("9",)),), **kw):
    return A.Question(
        qid,
        set_,
        "Is speedy trial part of Article 21?",
        gold=tuple(Gold(path=p, para_labels=list(labels)) for p, labels in gold),
        **kw,
    )


# --- one question ---------------------------------------------------------------------


def test_answered_question_record():
    r, traces = runner()
    rec = A.evaluate_question(q(), r)
    assert rec["status"] == "answered" and rec["set"] == "dev"
    assert len(traces) == 1  # the real trace is still written
    assert rec["retrieved"][0] == C1 and set(rec["retrieved"]) <= {C1, C2, C3}
    by_id = {c.chunk_id: c for c in CHUNKS}
    assert rec["contexts"] == [
        document_text(by_id[i].header, by_id[i].text) for i in rec["retrieved"]
    ]
    assert [b["kind"] for b in rec["rule_breaks"]] == ["quote_not_found"]
    assert len(rec["unverified"]) == 1 and rec["gaps"]
    assert rec["notice_shown"] and rec["citations_checked"] and rec["gold_retrieved"]
    (s,) = rec["sentences"]
    assert s["verdict"] == "supported" and not s["wrong"] and s["error_type"] is None
    assert s["citations"][0]["chunk_id"] == C1 and s["citations"][0]["quote_ok"]
    assert rec["llm_calls"]["answer"] == 1


def test_judge_stops_at_first_supported_citation():
    out = {
        "sentences": [
            {
                "sentence": "A speedy trial is part of Article 21.",
                "citations": [
                    {"label": "P1", "quote": GOOD_OUTPUT["sentences"][0]["citations"][0]["quote"]},
                    {
                        "label": "P1",
                        "quote": "right to a speedy trial is part of the right to life",
                    },
                ],
            }
        ],
        "gaps": [],
    }
    judge = FakeJudge()
    r, _ = runner(judge, llm=FakeLLM(out))
    rec = A.evaluate_question(q(), r)
    cites = rec["sentences"][0]["citations"]
    assert [c["verdict"] for c in cites] == ["supported", "supported"]  # second reused
    assert len(judge.calls) == 1


def test_wrong_sentence_is_a_hallucination_when_no_retrieved_passage_grounds_it():
    judge = FakeJudge(default="unsupported")
    r, _ = runner(judge)
    s = A.evaluate_question(q(), r)["sentences"][0]
    assert s["wrong"] and s["verdict"] == "unsupported"
    assert s["error_type"] == "hallucination"
    assert set(s["grounding"]) == set(A.evaluate_question(q(), r)["retrieved"])


def test_wrong_sentence_is_a_retrieval_error_when_gold_is_missing():
    # the cited passage is only partial support, and the gold judgment was not retrieved
    judge = FakeJudge(default="partial")
    r, _ = runner(judge)
    rec = A.evaluate_question(q(gold=((J2, ("99",)),)), r)
    assert rec["gold_retrieved"] is False
    assert rec["sentences"][0]["error_type"] == "retrieval_error"


def test_wrong_sentence_is_a_reasoning_error_when_gold_was_retrieved():
    judge = FakeJudge(default="partial")
    r, _ = runner(judge)
    rec = A.evaluate_question(q(), r)
    assert rec["gold_retrieved"] is True
    assert rec["sentences"][0]["error_type"] == "reasoning_error"
    # without gold (unanswerable / adversarial) retrieval cannot be blamed
    rec = A.evaluate_question(q("u-1", "unanswerable", gold=()), r)
    assert rec["gold_retrieved"] is None
    assert rec["sentences"][0]["error_type"] == "reasoning_error"


def test_hallucination_check_stops_at_first_grounding_passage():
    judge = FakeJudge(rules={"Bail is the rule": "partial"}, default="unsupported")
    r, _ = runner(judge)
    s = A.evaluate_question(q(), r)["sentences"][0]
    assert s["error_type"] == "reasoning_error"
    assert "partial" in s["grounding"].values()


def test_judge_error_is_not_wrong():
    r, _ = runner(FakeJudge(default=ERROR))
    s = A.evaluate_question(q(), r)["sentences"][0]
    assert s["verdict"] == ERROR and not s["wrong"]


def test_gold_retrieved_paragraph_rules():
    c = next(ch for ch in CHUNKS if ch.chunk_id == C1)  # J1, para "9"
    assert A.gold_retrieved([Gold(path=J1, para_labels=["9"])], [c])
    assert A.gold_retrieved([Gold(path=J1, para_labels=["9.1"])], [c])
    assert not A.gold_retrieved([Gold(path=J1, para_labels=["10"])], [c])
    assert A.gold_retrieved([Gold(path=J1, para_labels=[])], [c])  # judgment level
    assert not A.gold_retrieved([Gold(path=J2, para_labels=["9"])], [c])


# --- G9 -------------------------------------------------------------------------------


def adv(expect, category="injection"):
    return q("adv-1", "adversarial", gold=(), expect=expect, category=category)


def test_adversarial_expectations():
    r, _ = runner()
    rec = A.evaluate_question(adv("cited_only"), r)
    assert rec["g9"]["passed"]
    rec = A.evaluate_question(adv("refuse"), r)  # it was answered
    assert not rec["g9"]["passed"] and "refusal" in rec["g9"]["reasons"][0]
    rec = A.evaluate_question(adv("notice", "advice"), r)  # answered, with the notice
    assert rec["g9"]["passed"]


def test_adversarial_refusal_passes_refuse():
    r, _ = runner(llm=FakeLLM({"sentences": [], "gaps": ["nothing"]}))
    rec = A.evaluate_question(adv("refuse", "false_premise"), r)
    assert rec["status"] == "refused" and rec["g9"]["passed"]


def test_adversarial_check_without_notice_or_with_bad_citation():
    base = {"answer": [{"x": 1}], "status": "answered", "citations_checked": True}
    out = A.adversarial_check(adv("notice", "advice"), {**base, "notice_shown": False})
    assert not out["passed"] and len(out["reasons"]) == 2
    out = A.adversarial_check(
        adv("cited_only"), {**base, "notice_shown": True, "citations_checked": False}
    )
    assert out["reasons"] == ["a shown citation failed the re-check"]


def test_quote_recheck_catches_a_citation_that_does_not_match():
    c = next(ch for ch in CHUNKS if ch.chunk_id == C1)
    assert A.quote_found("Right To A Speedy   trial", c)
    assert not A.quote_found("not in the text at all", c)
    assert not A.quote_found("anything", None)


# --- 429 through the pipeline --------------------------------------------------------------


def rate_limit_error():
    cause = openai.RateLimitError("slow", response=httpx.Response(429, request=REQ), body=None)
    try:
        raise LLMUnavailable(f"RateLimitError: {cause}") from cause
    except LLMUnavailable as exc:
        return exc


class FlakyLLM(FakeLLM):
    def __init__(self, failures):
        super().__init__(GOOD_OUTPUT)
        self.failures = failures

    def complete_json(self, system, user, timeout, history=()):
        if self.failures:
            self.failures -= 1
            raise rate_limit_error()
        return super().complete_json(system, user, timeout, history)


def test_answer_llm_429_backs_off_inside_the_pipeline():
    t = FakeTime()
    deps, _ = make_deps(llm=FlakyLLM(2))
    ask_fn, llm = A.instrument(deps, SETTINGS, PausableClock(t.clock, t.sleep))
    r = A.Runner(ask_fn, deps.engine.get_chunks, FakeJudge(), llm, None)
    rec = A.evaluate_question(q(), r)
    assert rec["status"] == "answered"
    assert t.sleeps == [SETTINGS.answers.backoff_base_s, 2 * SETTINGS.answers.backoff_base_s]
    assert llm.rate_limited == 2 and rec["llm_calls"]["answer"] == 3


def test_answer_llm_quota_exhausted_stops_the_question():
    t = FakeTime()
    deps, _ = make_deps(llm=FlakyLLM(100))
    ask_fn, llm = A.instrument(deps, SETTINGS, PausableClock(t.clock, t.sleep))
    r = A.Runner(ask_fn, deps.engine.get_chunks, FakeJudge(), llm, None)
    with pytest.raises(A.QuotaExhausted):
        A.evaluate_question(q(), r)


# --- sessions and resume ------------------------------------------------------------------


def questions():
    return [q("dev-1"), q("dev-2"), q("u-1", "unanswerable", gold=()), adv("cited_only")]


def test_resume_skips_finished_questions(tmp_path):
    results = A.ResultsFile("answers-x", tmp_path)
    r, _ = runner()
    p1 = A.run_session(questions(), r, results, limit=2)
    assert (p1.finished_now, p1.remaining) == (2, 2) and "limit" in p1.stopped
    assert set(results.load()) == {"dev-1", "dev-2"}

    r2, traces = runner()
    p2 = A.run_session(questions(), r2, results)
    assert (p2.done_before, p2.finished_now, p2.remaining, p2.stopped) == (2, 2, 0, None)
    assert len(traces) == 2  # only the two unfinished questions were asked
    lines = results.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["id"] for x in lines] == ["dev-1", "dev-2", "u-1", "adv-1"]
    assert all(json.loads(x)["run_id"] == "answers-x" for x in lines)

    p3 = A.run_session(questions(), runner()[0], results)
    assert p3.finished_now == 0 and p3.remaining == 0


def test_quota_exhaustion_stops_the_session_without_recording(tmp_path):
    results = A.ResultsFile("answers-y", tmp_path)
    r, _ = runner(FakeJudge(exhaust_after=1))
    p = A.run_session(questions(), r, results)
    assert p.finished_now == 1 and "resume later with --run-id answers-y" in p.stopped
    assert set(results.load()) == {"dev-1"}
    p = A.run_session(questions(), runner()[0], results)
    assert p.done_before == 1 and p.remaining == 0


def test_results_of_other_runs_and_torn_lines_are_ignored(tmp_path):
    results = A.ResultsFile("answers-z", tmp_path)
    results.append({"run_id": "other", "id": "dev-1"})
    results.append({"run_id": "answers-z", "id": "dev-2", "status": "error"})
    with results.path.open("a", encoding="utf-8") as f:
        f.write('{"run_id": "answers-z", "id": "dev-')
    assert set(results.load()) == {"dev-2"}
    p = A.run_session([q("dev-2")], runner()[0], results, redo_errors=True)
    assert p.finished_now == 1
    assert results.load()["dev-2"]["status"] == "answered"


def test_meta_round_trip(tmp_path):
    results = A.ResultsFile("answers-m", tmp_path)
    assert not results.exists() and results.read_meta() == {}
    results.write_meta({"run_id": "answers-m", "sessions": []})
    assert results.exists() and results.read_meta()["run_id"] == "answers-m"


# --- metrics ------------------------------------------------------------------------------


def rec(qid, set_, status, sentences=(), unverified=0, breaks=0, g9=None):
    out = {
        "id": qid,
        "set": set_,
        "status": status,
        "answer": [{"sentence": "s"} for _ in sentences],
        "unverified": [{}] * unverified,
        "rule_breaks": [{"kind": "quote_not_found"}] * breaks,
        "sentences": [
            {
                "index": i,
                "verdict": v,
                "citations_checked": True,
                "wrong": v in ("partial", "unsupported"),
                "error_type": et,
            }
            for i, (v, et) in enumerate(sentences)
        ],
        "tokens": {"answer": {"total_tokens": 100}, "judge": {"total_tokens": 10}},
        "llm_calls": {"answer": 1, "judge": len(sentences)},
    }
    if g9 is not None:
        out["g9"] = g9
    return out


def test_summarize():
    records = [
        rec("d1", "dev", "answered", [("supported", None), ("partial", "reasoning_error")], 1, 2),
        rec("d2", "dev", "refused"),
        rec("d3", "dev", "passages"),
        rec("d4", "dev", "answered", [("supported", None), (ERROR, None)]),
        *[rec(f"u{i}", "unanswerable", "refused") for i in range(4)],
        rec("u9", "unanswerable", "answered", [("unsupported", "hallucination")]),
        rec("a1", "adversarial", "refused", g9={"category": "advice", "passed": True}),
        rec(
            "a2",
            "adversarial",
            "answered",
            [("supported", None)],
            g9={"category": "injection", "passed": False, "reasons": ["x"]},
        ),
    ]
    m = A.summarize(
        records,
        min_supported_share=0.9,
        min_unanswerable_refused_share=0.8,
        max_wrong_refusal_share=0.1,
    )
    g1 = m["g1"]
    assert (g1["verified_sentences"], g1["unverified_sentences"], g1["rule_breaks"]) == (6, 1, 2)
    assert g1["unverified_share"] == pytest.approx(1 / 7) and g1["passed"]
    g2 = m["g2"]
    assert (g2["judged_sentences"], g2["supported"], g2["judge_errors"]) == (5, 3, 1)
    assert g2["supported_share"] == pytest.approx(0.6) and not g2["passed"]
    g5 = m["g5"]
    assert g5["unanswerable"]["share"] == pytest.approx(0.8)
    assert g5["answerable"]["counted"] == 3 and g5["answerable"]["passages"] == 1
    assert g5["answerable"]["share"] == pytest.approx(1 / 3) and not g5["passed"]
    assert m["g8"]["by_type"] == {
        "hallucination": 1,
        "retrieval_error": 0,
        "reasoning_error": 1,
    }
    assert m["g9"]["questions"] == 2 and not m["g9"]["passed"]
    assert m["g9"]["by_category"]["advice"] == {"n": 1, "passed": 1}
    assert m["tokens"]["answer"]["total_tokens"] == 1100
    assert m["statuses"]["dev"] == {"answered": 2, "passages": 1, "refused": 1}


def test_summarize_g1_violation():
    r = rec("d1", "dev", "answered", [("supported", None)])
    r["sentences"][0]["citations_checked"] = False
    m = A.summarize(
        [r],
        min_supported_share=0.9,
        min_unanswerable_refused_share=0.8,
        max_wrong_refusal_share=0.1,
    )
    assert not m["g1"]["passed"] and m["g1"]["unchecked_citation_details"] == [
        {"id": "d1", "index": 0}
    ]


# --- loading ------------------------------------------------------------------------------


def test_load_questions(tmp_path):
    dev = write_jsonl(
        tmp_path / "dev.jsonl",
        [DevItem(id="d1", question="q", type="answerable", gold=[Gold(path=J1)])],
    )
    un = write_jsonl(tmp_path / "u.jsonl", [DevItem(id="u1", question="q", type="unanswerable")])
    ad = write_jsonl(
        tmp_path / "a.jsonl",
        [AdversarialItem(id="a1", question="q", category="advice", expect="notice")],
    )
    qs = A.load_questions(dev, un, ad)
    assert [(x.id, x.set) for x in qs] == [
        ("d1", "dev"),
        ("u1", "unanswerable"),
        ("a1", "adversarial"),
    ]
    assert qs[2].expect == "notice" and qs[0].gold[0].path == J1
    assert [x.id for x in A.load_questions(None, un, None)] == ["u1"]
    with pytest.raises(ValueError, match="more than one set"):
        A.load_questions(dev, dev, None)


def test_committed_question_sets_load():
    from nyaya.eval.datasets import DATASETS_DIR

    qs = A.load_questions(
        DATASETS_DIR / "dev_draft_v0.jsonl",
        DATASETS_DIR / "unanswerable_v1.jsonl",
        DATASETS_DIR / "adversarial_v1.jsonl",
    )
    assert {x.set for x in qs} == set(A.SETS)


def test_session_entry_names_provider_not_key(monkeypatch):
    from nyaya.config import load_settings

    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("LLM_MODEL", "m")
    s = load_settings(env_file=None)
    from datetime import UTC, datetime

    entry = A.session_entry(
        s,
        started_at=datetime(2026, 11, 12, tzinfo=UTC),
        progress=A.Progress(total=3, done_before=1, finished_now=2),
        answer_llm=None,
        judge_llm=None,
        index_version="v1",
    )
    assert entry["answer"]["provider"] == "api.groq.com" and entry["date"] == "2026-11-12"
    assert "sk-secret" not in json.dumps(entry)
    assert entry["progress"]["remaining"] == 0


def test_status_constants_cover_the_contract():
    assert {s.value for s in Status} == A.REFUSAL_STATUSES | A.NEITHER_STATUSES | {"answered"}
