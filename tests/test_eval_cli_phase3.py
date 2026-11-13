"""The Phase 3 `nyaya-eval` subcommands, wired to fakes (no database, GPU or network)."""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager

import pytest

from nyaya.config import load_settings
from nyaya.eval import answers as A
from nyaya.eval import cli, lawyer, runs
from nyaya.eval.datasets import AdversarialItem, DevItem, Gold, write_jsonl
from test_eval_answers import FakeJudge
from test_pipeline import J1, make_deps


class Conn:
    def __init__(self):
        self.sql = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, q, p=None):
        self.sql.append((q, p))


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Settings without secrets; runs, answers and lawyer output under tmp_path."""
    monkeypatch.setenv("LLM_MODEL", "answer-model")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    monkeypatch.setattr(A, "ANSWERS_DIR", tmp_path / "answers")
    monkeypatch.setattr(lawyer, "LAWYER_DIR", tmp_path / "lawyer")
    real_save = runs.save_run
    conns: list[Conn] = []

    def save_run(*a, **kw):
        kw["runs_dir"] = tmp_path / "runs"
        return real_save(*a, **kw)

    @contextmanager
    def connect(settings):
        conns.append(Conn())
        yield conns[-1]

    monkeypatch.setattr(runs, "save_run", save_run)
    monkeypatch.setattr(cli, "_connect", connect)
    monkeypatch.setattr(cli, "_judge_with_pacing", lambda s, clock: (FakeJudge(), None))
    monkeypatch.setattr("nyaya.pipeline.build_deps", lambda s: make_deps()[0])
    return tmp_path, conns


def datasets(tmp_path):
    q = "Is speedy trial part of Article 21?"
    dev = write_jsonl(
        tmp_path / "dev.jsonl",
        [
            DevItem(
                id=f"d{i}", question=q, type="answerable", gold=[Gold(path=J1, para_labels=["9"])]
            )
            for i in range(3)
        ],
    )
    un = write_jsonl(tmp_path / "un.jsonl", [DevItem(id="u1", question=q, type="unanswerable")])
    adv = write_jsonl(
        tmp_path / "adv.jsonl",
        [AdversarialItem(id="a1", question=q, category="injection", expect="cited_only")],
    )
    return ["--dev", str(dev), "--unanswerable", str(un), "--adversarial", str(adv)]


def test_answers_resume_then_finish_then_lawyer_round_trip(env, capsys):
    tmp_path, conns = env
    args = datasets(tmp_path)
    assert cli.main(["answers", *args, "--limit", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    run_id = out["run_id"]
    assert out["progress"]["remaining"] == 3 and "resume with --run-id" in out["note"]
    assert conns == []  # eval_runs is written only when the run is complete

    assert cli.main(["answers", *args, "--run-id", run_id]) == 1  # G5 fails: all answered
    out = json.loads(capsys.readouterr().out)
    assert out["progress"] == {
        "total": 5,
        "done_before": 2,
        "finished_now": 3,
        "remaining": 0,
        "stopped": None,
    }
    assert out["gates"] == {"g1": True, "g2": True, "g5": False, "g9": True}
    (sql, params) = conns[0].sql[0]
    assert "ON CONFLICT (run_id)" in sql and params[0] == run_id
    meta = A.ResultsFile(run_id).read_meta()
    assert len(meta["sessions"]) == 2 and meta["index_version"] == "v2025a"
    assert meta["sessions"][0]["judge"]["model"] == "judge-model"

    # a resume with different datasets is refused
    assert cli.main(["answers", *args, "--run-id", run_id, "--sets", "dev"]) == 2
    assert "used datasets" in capsys.readouterr().err

    assert cli.main(["lawyer-export"]) == 0  # the latest run
    info = json.loads(capsys.readouterr().out)
    assert info["exported"] == 5 and info["run_id"] == run_id
    path = tmp_path / "lawyer" / run_id / "ratings.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(right_paragraphs="3", law_correct="2", nothing_missing="2", usable="3")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lawyer.COLUMNS)
        w.writeheader()
        w.writerows(rows)

    assert cli.main(["lawyer-import", "--run-id", run_id]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["answers"] == 5 and out["means"]["law_correct"] == 2 and out["gate"] is False
    (sql, params) = conns[-1].sql[0]
    assert "INSERT INTO eval_runs" in sql and params[2] == "v2025a"
    assert json.loads(params[4])["kind"] == "lawyer-review"


def test_answers_needs_the_dev_set(env, capsys):
    tmp_path, _ = env
    assert cli.main(["answers", "--dev", str(tmp_path / "missing.jsonl")]) == 2
    assert "dev_draft_v0" in capsys.readouterr().err


def test_answers_unknown_run_and_bad_sets(env, capsys):
    tmp_path, _ = env
    args = datasets(tmp_path)
    assert cli.main(["answers", *args, "--run-id", "answers-nope"]) == 2
    assert "no answers run" in capsys.readouterr().err
    assert cli.main(["answers", *args, "--sets", "dev,tests"]) == 2
    assert "unknown sets" in capsys.readouterr().err


def test_judge_validate_without_hand_labels(env, capsys):
    tmp_path, _ = env
    assert cli.main(["judge-validate", "--labels", str(tmp_path / "none.jsonl")]) == 2
    assert "hand labels" in capsys.readouterr().err
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert cli.main(["judge-validate", "--labels", str(empty)]) == 2
    assert "is empty" in capsys.readouterr().err


def test_judge_not_configured_is_a_setup_error(monkeypatch, tmp_path, capsys):
    for k in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    monkeypatch.setattr(A, "ANSWERS_DIR", tmp_path / "answers")
    args = datasets(tmp_path)
    assert cli.main(["answers", *args]) == 2
    assert "JUDGE_BASE_URL" in capsys.readouterr().err


def test_lawyer_commands_without_runs(env, capsys):
    assert cli.main(["lawyer-export"]) == 2
    assert "run `answers` first" in capsys.readouterr().err
    assert cli.main(["lawyer-import", "--run-id", "answers-x"]) == 2


def test_latency_command(env, monkeypatch, capsys):
    tmp_path, conns = env
    ds = write_jsonl(
        tmp_path / "lat.jsonl",
        [
            DevItem(id=f"d{i}", question="Is speedy trial part of Article 21?", type="unanswerable")
            for i in range(4)
        ],
    )
    monkeypatch.setattr(
        "nyaya.eval.ratelimit.Throttle.wait", lambda self: None
    )  # no real pacing in tests
    assert cli.main(["latency", "--dataset", str(ds), "--count", "3", "--warmup", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 3 and out["failures"] == 0 and out["passed"]
    assert set(out["stages_s"]) == {"retrieve", "rerank", "llm", "check", "other"}
    assert json.loads(conns[0].sql[0][1][4])["kind"] == "latency"
