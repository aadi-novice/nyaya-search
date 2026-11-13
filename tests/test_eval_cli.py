import argparse
import json
from contextlib import contextmanager

import pytest

from eval_fakes import FakeEngine, chunk
from nyaya.config import load_settings
from nyaya.eval import cli, wiring
from nyaya.eval.datasets import DevItem, Gold, write_jsonl
from nyaya.eval.retrieval import Ranking


def test_parse_years():
    assert cli.parse_years("2025") == [2025]
    assert cli.parse_years("2023-2025") == [2023, 2024, 2025]
    assert cli.parse_years("2025, 2024,2025") == [2024, 2025]
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_years("2026-2024")
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_years(" ")


def test_every_design_subcommand_exists():
    parser = cli.build_parser()
    for name in [
        "build-devset",
        "g6-holdout",
        "retrieval",
        "size-steps",
        "bm25-trigger",
        "tune-threshold",
        "answers",
        "latency",
        "judge-validate",
        "lawyer-export",
        "lawyer-import",
    ]:
        assert parser.parse_args([name]).command == name


def test_not_built_commands_exit_2(capsys):
    assert cli.main(["latency"]) == 2
    assert "Phase 3" in capsys.readouterr().err


def test_unknown_method_is_a_usage_error(monkeypatch, tmp_path, capsys):
    ds = write_jsonl(tmp_path / "d.jsonl", [DevItem(id="u", question="q", type="unanswerable")])
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    assert cli.main(["retrieval", "--dataset", str(ds), "--methods", "bm42"]) == 2
    assert "unknown methods" in capsys.readouterr().err


def test_build_devset_requires_llm_secrets(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert cli.main(["build-devset", "--years", "2025"]) == 2
    assert "LLM_BASE_URL" in capsys.readouterr().err


def test_build_devset_end_to_end(monkeypatch, tmp_path, capsys):
    from nyaya.eval import devset_from_held as dfh

    monkeypatch.setenv("LLM_BASE_URL", "http://x")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    cands = [
        dfh.HeldCandidate(
            path=f"2025_1_{i}_9",
            year=2025,
            held_text="Held: some holding about bail. [Para 3]",
            para_labels=["3"],
            source_held_chunk=f"2025_1_{i}_9#editorial#001",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(dfh, "load_candidates_from_build", lambda d, years: cands)

    class LLM:
        def __init__(self, *a, **k):
            self.n = 0

        def complete_json(self, system, user, timeout):
            self.n += 1
            return {"question": f"Which principle governs pre-arrest relief, case {self.n}?"}

    monkeypatch.setattr(dfh, "OpenAICompatLLM", LLM)
    out = tmp_path / "dev_v1.jsonl"
    assert cli.main(["build-devset", "--years", "2025", "--out", str(out), "--size", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kept"] == 2 and report["llm_calls"] == 2
    assert (tmp_path / "dev_v1_handcheck.csv").exists()


class FakeConn:
    def __init__(self):
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @contextmanager
    def transaction(self):
        yield

    def execute(self, q, p=None):
        self.sql.append((q, p))

        class R:
            def fetchone(self):
                return ("v2025",)

        return R()


def test_retrieval_command_with_fakes(monkeypatch, tmp_path, capsys):
    from nyaya.eval import runs

    eng = FakeEngine([chunk("A#judgment#001", labels=["1"])])

    class R:
        def run(self, method, question, filters):
            return Ranking(["A#judgment#001"], 0.9)

    conn = FakeConn()
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    monkeypatch.setattr(cli, "_connect", lambda s: conn)
    monkeypatch.setattr(wiring, "make_encoders", lambda s, name: (None, None))
    monkeypatch.setattr(wiring, "make_engine_retriever", lambda s, **kw: (R(), eng))
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    real_save = runs.save_run
    monkeypatch.setattr(runs, "save_run", lambda *a, **k: real_save(*a, runs_dir=tmp_path, **k))
    ds = write_jsonl(
        tmp_path / "dev_v1.jsonl",
        [DevItem(id="q1", question="x", type="answerable", gold=[Gold(path="A")])],
    )
    assert cli.main(["retrieval", "--dataset", str(ds), "--methods", "hybrid"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["hybrid"]["judgment"]["recall@5"] == 1.0
    assert any("INSERT INTO eval_runs" in q for q, _ in conn.sql)
    assert list(tmp_path.glob("retrieval-*.json"))


# --- wiring adapters ----------------------------------------------------------------


def test_embed_query_adapters():
    class Single:
        def embed_query(self, q):
            return (1.0, 2.0)

    class Batch:
        def encode_queries(self, qs):
            return [[3.0, 4.0] for _ in qs]

    assert wiring.embed_query_fn(Single())("q") == [1.0, 2.0]
    assert wiring.embed_query_fn(Batch())("q") == [3.0, 4.0]
    with pytest.raises(TypeError):
        wiring.embed_query_fn(object())


def test_rerank_adapter_passes_header_and_text():
    c = chunk("A#judgment#001", text="body", header="head")

    class Scores:
        def scores(self, q, docs):
            return [1.0 if docs[0] == "head\n\nbody" else 0.0]

    class TopN:
        def rerank(self, q, docs, top_n):
            from nyaya.models.encoders import RerankResult

            assert top_n == len(docs)
            return [RerankResult(1, 0.9), RerankResult(0, 0.2)]

    assert wiring.rerank_fn(Scores())("q", [c]) == [1.0]
    assert wiring.rerank_fn(TopN())("q", [c, c]) == [0.2, 0.9]
    with pytest.raises(TypeError):
        wiring.rerank_fn(object())


def test_active_version():
    class C:
        def __init__(self, row):
            self.row = row

        def execute(self, q):
            row = self.row

            class R:
                def fetchone(self):
                    return row

            return R()

    assert wiring.active_version(C(("v1",))) == "v1"
    with pytest.raises(RuntimeError):
        wiring.active_version(C(None))
