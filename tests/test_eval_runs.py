import json
from contextlib import contextmanager

import pytest

from nyaya.eval.runs import new_run_id, save_run


class Conn:
    def __init__(self):
        self.sql = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, q, p=None):
        self.sql.append((q, p))


def test_new_run_id():
    a, b = new_run_id("retrieval"), new_run_id("retrieval")
    assert a.startswith("retrieval-") and a != b


def test_save_run_writes_json_and_row(tmp_path):
    conn = Conn()
    run_id, path = save_run(
        "retrieval",
        {"recall": 0.9},
        config={"methods": ["hybrid"]},
        dataset_version="dev_v1",
        index_version="v2025",
        conn=conn,
        runs_dir=tmp_path,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == run_id and data["metrics"] == {"recall": 0.9}
    q, p = conn.sql[0]
    assert q.startswith("INSERT INTO eval_runs")
    assert p[0] == run_id and p[2] == "v2025" and json.loads(p[4])["kind"] == "retrieval"


def test_save_run_without_db(tmp_path):
    _, path = save_run("x", {}, config={}, dataset_version=None, runs_dir=tmp_path)
    assert path.exists()
    with pytest.raises(ValueError, match="index_version or schema_name"):
        save_run("x", {}, config={}, dataset_version=None, conn=Conn(), runs_dir=tmp_path)
