"""The G2 judge: strict parse, one repair retry, kappa, and judge-validate (no network)."""

from __future__ import annotations

import json

import pytest

from nyaya.answer.llm_client import LLMBadJSON, LLMUnavailable
from nyaya.answer.prompt import load_template
from nyaya.config import load_settings
from nyaya.eval import judge as J
from nyaya.eval.datasets import HandLabel
from nyaya.eval.ratelimit import RateLimitExhausted

SETTINGS = load_settings(env_file=None)
TEMPLATE = load_template("judge_v1")


class ScriptedLLM:
    model = "judge-model"

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete_json(self, system, user, timeout, history=()):
        self.calls.append({"system": system, "user": user, "history": list(history)})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(*script):
    llm = ScriptedLLM(*script)
    return J.Judge(llm, TEMPLATE, timeout_s=10), llm


def test_template_has_both_placeholders_and_fills_them():
    judge, _ = make()
    system, user = judge.prompt("THE SENTENCE", "THE PASSAGE")
    assert "supported" in system and "partial" in system
    assert "THE SENTENCE" in user and "THE PASSAGE" in user and "{{" not in user


@pytest.mark.parametrize(
    "obj",
    [
        [],
        {"label": "maybe"},
        {"reason": "no label"},
        {"label": "supported", "score": 1},
        {"label": "supported", "reason": 3},
        {"label": "Supported"},
    ],
)
def test_parse_verdict_is_strict(obj):
    with pytest.raises(J.BadVerdict):
        J.parse_verdict(obj)


def test_parse_verdict_accepts_label_with_optional_reason():
    assert J.parse_verdict({"label": "partial"}) == ("partial", "")
    assert J.parse_verdict({"label": "supported", "reason": "r"}) == ("supported", "r")


def test_good_reply_first_time():
    judge, llm = make({"label": "supported", "reason": "says so"})
    v = judge.judge("s", "p")
    assert (v.label, v.reason, v.attempts, v.error) == ("supported", "says so", 1, None)
    assert len(llm.calls) == 1


def test_bad_reply_is_retried_once_with_the_error():
    judge, llm = make({"label": "yes"}, {"label": "unsupported", "reason": "r"})
    v = judge.judge("s", "p")
    assert (v.label, v.attempts) == ("unsupported", 2)
    history = llm.calls[1]["history"]
    assert history[0]["role"] == "assistant" and json.loads(history[0]["content"]) == {
        "label": "yes"
    }
    assert "label must be one of" in history[1]["content"]


def test_two_bad_replies_give_error():
    judge, llm = make(LLMBadJSON("invalid JSON: x", "not json"), {"label": "nope"})
    v = judge.judge("s", "p")
    assert v.label == J.ERROR and v.attempts == 2 and "bad reply" in v.error
    assert llm.calls[1]["history"][0]["content"] == "not json"


def test_unavailable_gives_error_but_quota_exhaustion_raises():
    judge, _ = make(LLMUnavailable("ConnectError: down"))
    assert judge.judge("s", "p").label == J.ERROR
    judge, _ = make(RateLimitExhausted("429"))
    with pytest.raises(RateLimitExhausted):
        judge.judge("s", "p")


def test_best():
    assert J.best(["unsupported", "partial"]) == "partial"
    assert J.best(["error", "supported"]) == "supported"
    assert J.best([]) == J.ERROR


def test_cohen_kappa_known_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.7, pe 0.5, kappa 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert J.cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edges():
    assert J.cohen_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert J.cohen_kappa(["a", "a"], ["a", "a"]) == 1.0  # chance agreement 1
    assert J.cohen_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        J.cohen_kappa([], [])
    with pytest.raises(ValueError):
        J.cohen_kappa(["a"], [])


def test_read_hand_labels_missing_or_empty(tmp_path):
    with pytest.raises(J.HandLabelsMissing, match="not found"):
        J.read_hand_labels(tmp_path / "nope.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(J.HandLabelsMissing, match="is empty"):
        J.read_hand_labels(empty)


class LabelJudge:
    """Returns the label written in the passage text."""

    model = "fake-judge"

    def judge(self, sentence, passage):
        return J.Verdict(passage if passage in J.LABELS else J.ERROR, error=None)


def hl(i, label, chunk="c1"):
    return HandLabel(id=f"hl-{i}", sentence="s", chunk_id=chunk, label=label, labeller="author")


def test_validate_reports_kappa_and_pass():
    labels = [hl(1, "supported", "s"), hl(2, "partial", "p"), hl(3, "unsupported", "u")]
    labels.append(hl(4, "supported", "missing"))
    labels.append(hl(5, "partial", "broken"))
    passages = {"s": "supported", "p": "partial", "u": "partial", "broken": "garbage"}
    out = J.validate(labels, passages, LabelJudge(), min_kappa=0.6, min_labels=3)
    assert out["missing_chunks"] == ["hl-4"]
    assert out["judge_errors"] == 1 and out["scored"] == 3
    assert out["agreement"] == pytest.approx(2 / 3)
    assert out["confusion"]["unsupported"]["partial"] == 1
    assert out["kappa"] == pytest.approx(J.cohen_kappa(["s", "p", "u"], ["s", "p", "p"]))
    assert out["enough_labels"] and out["passed"] == (out["kappa"] >= 0.6)


def test_validate_needs_enough_labels():
    labels = [hl(1, "supported", "s"), hl(2, "partial", "p")]
    out = J.validate(
        labels, {"s": "supported", "p": "partial"}, LabelJudge(), min_kappa=0.6, min_labels=30
    )
    assert out["kappa"] == pytest.approx(1.0) and not out["enough_labels"] and not out["passed"]


def test_from_settings_requires_judge_secrets(monkeypatch):
    for k in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(J.JudgeNotConfigured, match="JUDGE_BASE_URL"):
        J.Judge.from_settings(load_settings(env_file=None))


def test_from_settings_builds_and_wraps(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "http://judge/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_MODEL", "judge-m")
    seen = []

    def wrap(llm):
        seen.append(llm)
        return llm

    judge = J.Judge.from_settings(load_settings(env_file=None), wrap=wrap)
    assert judge.model == "judge-m" and seen[0].model == "judge-m"
    assert seen[0].max_output_tokens == SETTINGS.judge.max_output_tokens
    assert judge.timeout_s == SETTINGS.judge.timeout_s
