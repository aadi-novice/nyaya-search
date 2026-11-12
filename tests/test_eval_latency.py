"""`nyaya-eval latency`: percentiles, stage split, pacing and failures (fake pipeline)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from nyaya.contracts import AskRequest, Status
from nyaya.eval import latency as L
from nyaya.eval.datasets import DevItem
from nyaya.eval.ratelimit import PausableClock, Throttle


def test_percentile_nearest_rank():
    vals = [float(x) for x in range(1, 21)]  # 1..20
    assert L.percentile(vals, 50) == 10
    assert L.percentile(vals, 95) == 19
    assert L.percentile(vals, 100) == 20
    assert L.percentile([3.0], 95) == 3.0
    assert L.percentile([], 95) is None
    assert L.percentile([1.0, math.inf], 50) == 1.0
    assert math.isinf(L.percentile([1.0, math.inf], 95))


def test_stage_split_groups_and_other():
    t = {"embed": 100, "keyword": 50, "rerank": 2000, "llm": 6000, "check": 10, "total": 8500}
    s = L.stage_split(t)
    assert s["retrieve"] == pytest.approx(0.15)
    assert s["rerank"] == pytest.approx(2.0) and s["llm"] == pytest.approx(6.0)
    assert s["other"] == pytest.approx(8.5 - 0.15 - 2.0 - 6.0 - 0.01)
    assert L.stage_split({"mystery": 5})["other"] == pytest.approx(0.005)


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def response(status=Status.ANSWERED, timings=None):
    return SimpleNamespace(status=status, timings_ms=timings or {"llm": 1000, "total": 1500})


class FakeAsk:
    """Each question takes `durations[question]` seconds of fake wall time."""

    def __init__(self, t, durations, statuses=None, limited=None):
        self.t = t
        self.durations = durations
        self.statuses = statuses or {}
        self.limited = dict(limited or {})
        self.asked: list[str] = []

    def __call__(self, request):
        qtext = request.question
        self.asked.append(qtext)
        self.t.now += self.durations[qtext]
        if self.limited.get(qtext):
            self.limited[qtext] -= 1
            return response(Status.PASSAGES), SimpleNamespace(
                llm_error="LLMUnavailable: RateLimitError: Error code: 429"
            )
        return response(self.statuses.get(qtext, Status.ANSWERED)), SimpleNamespace(llm_error=None)


def run(t, ask, queries, **kw):
    clock = PausableClock(t.clock, t.sleep)
    args = {
        "warmup": 1,
        "request_limit_s": 30,
        "throttle": Throttle(0, clock),
        "clock": clock,
        "retries": 2,
        "backoff_base_s": 5,
        "backoff_max_s": 60,
    }
    args.update(kw)
    if "rpm" in args:
        args["throttle"] = Throttle(args.pop("rpm"), clock)
    return L.run_latency(queries, ask, **args)


def qs(*names):
    return [(n, AskRequest(question=n)) for n in names]


def test_warm_one_at_a_time_and_summary():
    t = FakeTime()
    durations = {f"q{i}": float(i) for i in range(1, 21)}
    ask = FakeAsk(t, durations)
    samples = run(t, ask, qs(*durations))
    assert ask.asked[0] == "q1" and len(ask.asked) == 21  # one warm-up, not counted
    assert [s.seconds for s in samples] == [float(i) for i in range(1, 21)]
    m = L.summarize(samples, p95_target_s=19)
    assert (m["p50_s"], m["p95_s"], m["passed"], m["failures"]) == (10, 19, True, 0)
    assert m["stages_s"]["llm"]["p95"] == pytest.approx(1.0)
    assert not L.summarize(samples, p95_target_s=18.5)["passed"]


def test_failures_count_as_infinitely_slow():
    t = FakeTime()
    durations = {"a": 1.0, "b": 40.0, "c": 2.0}
    ask = FakeAsk(t, durations, statuses={"c": Status.ERROR})
    samples = run(t, ask, qs("a", "b", "c"), warmup=0)
    assert [s.failed for s in samples] == [False, True, True]  # over request_s; error
    m = L.summarize(samples, p95_target_s=15)
    assert m["failures"] == 2 and m["p95_s"] is None and not m["passed"]
    assert m["p50_s"] is None  # 2 of 3 failed: the median is a failure too


def test_pacing_happens_outside_the_timed_region():
    t = FakeTime()
    ask = FakeAsk(t, {"a": 1.0, "b": 1.0})
    samples = run(t, ask, qs("a", "b"), warmup=0, rpm=20)  # 3 s apart
    assert [s.seconds for s in samples] == [1.0, 1.0]
    assert t.sleeps == [pytest.approx(2.0)]


def test_rate_limited_request_is_retried_not_sampled():
    t = FakeTime()
    ask = FakeAsk(t, {"a": 0.5}, limited={"a": 2})
    (s,) = run(t, ask, qs("a"), warmup=0)
    assert s.status == "answered" and s.seconds == 0.5
    assert t.sleeps == [5, 10] and len(ask.asked) == 3


def test_rate_limited_too_long_aborts():
    t = FakeTime()
    ask = FakeAsk(t, {"a": 0.5}, limited={"a": 10})
    with pytest.raises(L.LatencyAborted):
        run(t, ask, qs("a"), warmup=0)


def test_fixed_query_set_is_first_n_by_id():
    items = [DevItem(id=f"d{i}", question=f"q{i}", type="unanswerable") for i in (3, 1, 2)]
    assert [i for i, _ in L.fixed_query_set(items, 2)] == ["d1", "d2"]
