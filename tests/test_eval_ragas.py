"""`nyaya-eval ragas`: the RAGAS adapters, scoring and report, with a fake LLM.

No network, GPU or database. Skipped when the `ragas` extra is not installed.
"""

from __future__ import annotations

import csv
import json
import math

import pytest

pytest.importorskip("ragas")

from ragas.embeddings.base import BaseRagasEmbedding

from nyaya.answer.llm_client import LLMBadJSON
from nyaya.config import load_settings
from nyaya.eval import cli, runs
from nyaya.eval import ragas_metrics as R
from nyaya.eval.answers import ResultsFile
from nyaya.eval.ragas_adapters import BadReply, CompleterLLM
from nyaya.eval.ratelimit import RateLimitExhausted

# --- fakes --------------------------------------------------------------------------------

REPLIES = {
    # faithfulness, step 1: two claims
    "StatementGeneratorOutput": {"statements": ["claim one", "claim two"]},
    # faithfulness, step 2: one of the two supported -> 0.5
    "NLIStatementOutput": {
        "statements": [
            {"statement": "claim one", "reason": "stated", "verdict": 1},
            {"statement": "claim two", "reason": "not stated", "verdict": 0},
        ]
    },
    # answer relevancy: a committal question (the fake embeddings make it identical)
    "AnswerRelevanceOutput": {"question": "q?", "noncommittal": 0},
    # context precision: every passage useful -> 1.0
    "ContextPrecisionOutput": {"reason": "useful", "verdict": 1},
    # context recall: two of three reference claims attributed -> 2/3
    "ContextRecallOutput": {
        "classifications": [
            {"statement": "a", "reason": "in context", "attributed": 1},
            {"statement": "b", "reason": "in context", "attributed": 1},
            {"statement": "c", "reason": "missing", "attributed": 0},
        ]
    },
}


class FakeCompleter:
    """A JSONCompleter answering each ragas prompt by the output schema it names."""

    model = "fake-judge"

    def __init__(self, replies=None, fail=None):
        self.replies = {**REPLIES, **(replies or {})}
        self.fail = fail  # exception to raise on every call
        self.calls: list[tuple[str, str, list]] = []

    def complete_json(self, system, user, timeout, history=()):
        self.calls.append((system, user, list(history)))
        if self.fail is not None:
            raise self.fail
        for title, reply in self.replies.items():
            if f'"title": "{title}"' in user:
                if isinstance(reply, Exception):
                    raise reply
                return json.loads(json.dumps(reply))
        raise AssertionError(f"unexpected prompt: {user[:200]}")


class FakeEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text, **kw):
        return [1.0, 0.0, 0.0]

    async def aembed_text(self, text, **kw):
        return self.embed_text(text)


def metrics(completer=None, relevancy=True):
    llm = CompleterLLM(completer or FakeCompleter(), timeout_s=5)
    return R.make_metrics(llm, FakeEmbeddings() if relevancy else None, strictness=2)


def sentence(text, quote_ok=True, verdict="supported"):
    return {
        "index": 0,
        "sentence": text,
        "verdict": verdict,
        "citations": [{"chunk_id": "j#judgment#001", "quote_ok": quote_ok}],
        "citations_checked": quote_ok,
        "wrong": verdict in ("partial", "unsupported"),
        "error_type": None,
    }


def record(qid, status="answered", sentences=None, contexts=("passage one", "passage two")):
    sents = (
        sentences
        if sentences is not None
        else ([sentence("The Court held X.")] if status == "answered" else [])
    )
    rec = {
        "run_id": "answers-test",
        "id": qid,
        "set": "dev",
        "question": f"question {qid}?",
        "status": status,
        "answer": [{"sentence": s["sentence"], "citations": []} for s in sents],
        "unverified": [],
        "rule_breaks": [],
        "retrieved": ["j#judgment#001", "j#judgment#002"],
        "sentences": sents,
        "citations_checked": all(s["citations_checked"] for s in sents),
    }
    if contexts is not None:
        rec["contexts"] = list(contexts)
    return rec


REFS = {"dev-1": "Held: X.", "dev-2": "Held: Y.", "dev-3": "Held: Z."}


# --- the LLM adapter ----------------------------------------------------------------------


def test_completer_llm_validates_against_the_ragas_model():
    from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput

    fake = FakeCompleter()
    out = CompleterLLM(fake, timeout_s=5).generate(
        'schema "title": "ContextPrecisionOutput"', ContextPrecisionOutput
    )
    assert out.verdict == 1 and fake.calls[0][2] == []


def test_completer_llm_retries_a_bad_reply_once_with_the_error():
    from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput

    class OnceBad(FakeCompleter):
        def complete_json(self, system, user, timeout, history=()):
            self.calls.append((system, user, list(history)))
            if len(self.calls) == 1:
                return {"verdict": "maybe"}  # wrong type, no reason
            return {"reason": "ok", "verdict": 0}

    fake = OnceBad()
    out = CompleterLLM(fake, timeout_s=5).generate("p", ContextPrecisionOutput)
    assert out.verdict == 0 and len(fake.calls) == 2
    history = fake.calls[1][2]
    assert history[0]["role"] == "assistant" and "maybe" in history[0]["content"]
    assert "not valid" in history[1]["content"]


def test_completer_llm_gives_up_after_two_bad_replies():
    from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput

    fake = FakeCompleter(fail=LLMBadJSON("invalid JSON", "{oops"))
    with pytest.raises(BadReply):
        CompleterLLM(fake, timeout_s=5).generate("p", ContextPrecisionOutput)
    assert len(fake.calls) == 2


def test_completer_llm_lets_quota_exhaustion_through():
    from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput

    fake = FakeCompleter(fail=RateLimitExhausted("429"))
    with pytest.raises(RateLimitExhausted):
        CompleterLLM(fake, timeout_s=5).generate("p", ContextPrecisionOutput)
    assert len(fake.calls) == 1


# --- samples ------------------------------------------------------------------------------


def test_build_samples():
    recs = [
        record("dev-2", status="refused"),
        record("dev-1"),
        record("dev-3", status="passages"),
        {**record("unans-1", status="refused"), "set": "unanswerable"},
        record("dev-4", contexts=None),
    ]
    fetched = []

    def fetch(ids):
        fetched.append(list(ids))
        return [f"text of {i}" for i in ids]

    samples = R.build_samples(recs, REFS, max_contexts=1, fetch_contexts=fetch)
    assert [s.id for s in samples] == ["dev-1", "dev-2", "dev-3", "dev-4"]
    s1, s2, s3, s4 = samples
    assert s1.answered and not s1.refused and s1.response == "The Court held X."
    assert s1.contexts == ("passage one",) and s1.reference == "Held: X."
    assert s2.refused and not s2.answered
    assert not s3.refused and not s3.answered
    assert s4.reference is None and s4.contexts == ("text of j#judgment#001",)
    assert fetched == [["j#judgment#001"]]  # only the record without contexts


def test_read_references(tmp_path):
    p = tmp_path / "x_handcheck.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "held_text", "path", "para_labels", "ok", "comment"])
        w.writerow(["dev-1", "q", " Held: X. ", "p", "1", "", ""])
        w.writerow(["dev-2", "q", "", "p", "1", "", ""])
    assert R.read_references(p) == {"dev-1": "Held: X."}
    with pytest.raises(FileNotFoundError):
        R.read_references(tmp_path / "missing.csv")


def test_committed_dev_draft_has_a_reference_for_every_question():
    from nyaya.eval.datasets import DevItem, read_jsonl

    refs = R.read_references(R.default_references("dev_draft_v0"))
    items = read_jsonl(R.DATASETS_DIR / "dev_draft_v0.jsonl", DevItem)
    assert len(items) == 44 and {i.id for i in items} <= set(refs)


# --- scoring ------------------------------------------------------------------------------


def test_scores_with_the_fake_llm():
    (s,) = R.build_samples([record("dev-1")], REFS, max_contexts=8)
    out = R.score_sample(s, metrics())
    assert out["faithfulness"]["score"] == pytest.approx(0.5)
    assert out["answer_relevancy"]["score"] == pytest.approx(1.0)
    assert out["context_precision"]["score"] == pytest.approx(1.0)
    assert out["context_recall"]["score"] == pytest.approx(2 / 3)
    assert all(e["error"] is None and e["skipped"] is None for e in out.values())


def test_refused_answers_are_left_out_of_faithfulness_and_relevancy():
    fake = FakeCompleter()
    (s,) = R.build_samples([record("dev-2", status="refused")], REFS, max_contexts=8)
    out = R.score_sample(s, metrics(fake))
    assert out["faithfulness"] == {"score": None, "skipped": "refused", "error": None}
    assert out["answer_relevancy"]["skipped"] == "refused"
    # retrieval metrics do not depend on the answer
    assert out["context_precision"]["score"] == pytest.approx(1.0)
    assert out["context_recall"]["score"] == pytest.approx(2 / 3)
    prompts = " ".join(c[1] for c in fake.calls)
    assert "StatementGeneratorOutput" not in prompts and "AnswerRelevanceOutput" not in prompts


def test_skips_without_reference_or_contexts():
    recs = [record("dev-9"), record("dev-1", contexts=())]
    s9, s1 = sorted(R.build_samples(recs, REFS, max_contexts=8), key=lambda s: s.id != "dev-9")
    assert R.skip_reason("context_recall", s9) == "no_reference"
    assert R.skip_reason("faithfulness", s1) == "no_contexts"
    assert R.skip_reason("answer_relevancy", s1) is None
    assert R.skip_reason("context_precision", s1) == "no_contexts"


def test_a_metric_error_is_recorded_and_nan_is_not_a_score():
    fake = FakeCompleter(
        replies={
            "StatementGeneratorOutput": {"statements": []},  # nothing to judge -> NaN
            "ContextRecallOutput": LLMBadJSON("invalid JSON", "{"),
        }
    )
    (s,) = R.build_samples([record("dev-1")], REFS, max_contexts=8)
    out = R.score_sample(s, metrics(fake, relevancy=False))
    assert "answer_relevancy" not in out
    assert out["faithfulness"]["score"] is None and "no score" in out["faithfulness"]["error"]
    assert out["context_recall"]["score"] is None
    assert out["context_recall"]["error"].startswith("BadReply")
    assert out["context_precision"]["score"] == pytest.approx(1.0)


def test_session_resumes_and_stops_on_quota(tmp_path):
    samples = R.build_samples([record("dev-1"), record("dev-2")], REFS, max_contexts=8)
    results = ResultsFile("answers-test", tmp_path)
    p = R.run_session(samples, metrics(), results, limit=1)
    assert p.finished_now == 1 and p.stopped and set(results.load()) == {"dev-1"}

    stuck = metrics(FakeCompleter(fail=RateLimitExhausted("429")))
    p = R.run_session(samples, stuck, results)
    assert p.done_before == 1 and p.finished_now == 0 and "resume" in p.stopped
    assert set(results.load()) == {"dev-1"}  # the stopped question is not recorded

    p = R.run_session(samples, metrics(), results)
    assert p.finished_now == 1 and p.remaining == 0 and set(results.load()) == {"dev-1", "dev-2"}


# --- report -------------------------------------------------------------------------------


def scored_rows():
    recs = [
        record("dev-1"),
        record("dev-2", status="refused"),
        record("dev-3", sentences=[sentence("Wrong.", verdict="unsupported")]),
    ]
    samples = R.build_samples(recs, REFS, max_contexts=8)
    return recs, [
        {
            "id": s.id,
            "refused": s.refused,
            "answered": s.answered,
            "scores": R.score_sample(s, metrics()),
        }
        for s in samples
    ]


def test_aggregate_reports_refusals_separately():
    _, rows = scored_rows()
    agg = R.aggregate(rows, list(R.METRICS))
    assert agg["questions"] == 3
    assert agg["refusals"] == {
        "refused": 1,
        "refusal_rate": pytest.approx(1 / 3),
        "no_answer": 0,
        "answered": 2,
    }
    f = agg["metrics"]["faithfulness"]
    assert f["scored"] == 2 and f["skipped"] == {"refused": 1} and f["mean"] == pytest.approx(0.5)
    assert agg["metrics"]["context_recall"]["scored"] == 3
    assert "left out of faithfulness" in agg["note"]


def test_markdown_table_puts_our_metrics_beside_ragas():
    recs, rows = scored_rows()
    ours = R.our_metrics(
        recs,
        min_supported_share=0.9,
        min_unanswerable_refused_share=0.8,
        max_wrong_refusal_share=0.1,
    )
    assert ours["quote_verification"] == {
        "rate": 1.0,
        "citations": 2,
        "note": ours["quote_verification"]["note"],
    }
    assert ours["citation_support"]["share"] == pytest.approx(0.5)
    assert ours["wrong_refusal_share"]["share"] == pytest.approx(1 / 3)
    md = R.markdown_table(ours, R.aggregate(rows, list(R.METRICS)))
    lines = md.splitlines()
    assert lines[0] == "| Metric | Kind | Score | Over |"
    assert lines[2].startswith("| Quote verification (primary) | ours, exact (code) | 1.000 |")
    assert "| Faithfulness | RAGAS, LLM | 0.500 | 2 questions |" in md
    assert "| Context recall | RAGAS, LLM | 0.667 | 3 questions |" in md
    assert "Refused: 1 of 3" in md and "refusal rate 0.333" in md
    assert not any("Refused, unanswerable" in line for line in lines)  # no such set here


# --- the command --------------------------------------------------------------------------


def test_cmd_ragas_end_to_end(tmp_path, monkeypatch, capsys):
    answers_dir = tmp_path / "answers" / "answers-test"
    answers_dir.mkdir(parents=True)
    results = answers_dir / "results.jsonl"
    recs = [
        record("dev-1"),
        record("dev-2", status="refused"),
        {
            **record("adv-1"),
            "set": "adversarial",
            "g9": {"category": "advice", "expect": "notice", "passed": True, "reasons": []},
        },
    ]
    results.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    refs = tmp_path / "refs.csv"
    with refs.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "held_text"])
        w.writerows(REFS.items())

    real_save = runs.save_run

    def save_run(*a, **kw):
        kw["runs_dir"] = tmp_path / "runs"
        return real_save(*a, **kw)

    monkeypatch.setattr(runs, "save_run", save_run)
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    monkeypatch.setattr(R, "RAGAS_DIR", tmp_path / "ragas")
    monkeypatch.setattr(R, "make_llm", lambda s: (CompleterLLM(FakeCompleter(), 5), None))
    monkeypatch.setattr(R, "make_embeddings", lambda s: FakeEmbeddings())

    argv = ["ragas", "--results", str(results), "--references", str(refs)]
    assert cli.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gate"] is False and out["progress"]["remaining"] == 0
    assert out["refusals"]["refused"] == 1 and out["refusals"]["refusal_rate"] == 0.5
    assert out["means"]["faithfulness"] == pytest.approx(0.5)
    report = json.loads((tmp_path / "runs" / "ragas-answers-test.json").read_text("utf-8"))
    m = report["metrics"]
    assert m["complete"] and [r["id"] for r in m["per_question"]] == ["dev-1", "dev-2"]
    assert m["ours"]["adversarial"]["questions"] == 1
    assert (tmp_path / "runs" / "ragas-answers-test.md").read_text("utf-8") == m["table_md"]
    assert set(ResultsFile("answers-test", tmp_path / "ragas").load()) == {"dev-1", "dev-2"}

    # a second run finds everything done, and refuses other settings
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["progress"]["done_before"] == 2
    assert cli.main([*argv, "--no-relevancy"]) == 2
    assert "move it away" in capsys.readouterr().err


def test_mean_ignores_nan():
    rows = [
        {"refused": False, "answered": True, "scores": {"faithfulness": e}}
        for e in (
            {"score": 1.0, "skipped": None, "error": None},
            {"score": None, "skipped": None, "error": "no score (nothing to judge)"},
        )
    ]
    f = R.aggregate(rows, ["faithfulness"])["metrics"]["faithfulness"]
    assert f["mean"] == 1.0 and f["errors"] == 1 and not math.isnan(f["mean"])
