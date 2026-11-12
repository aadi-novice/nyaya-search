"""Recording evaluation runs: a JSON file under eval/runs/, and an `eval_runs` row."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nyaya.eval.datasets import RUNS_DIR


def new_run_id(kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"


def save_run(
    kind: str,
    metrics: dict[str, Any],
    *,
    config: dict[str, Any],
    dataset_version: str | None,
    index_version: str | None = None,
    schema_name: str | None = None,
    conn: Any = None,
    runs_dir: Path | str = RUNS_DIR,
) -> tuple[str, Path]:
    """Write `<runs_dir>/<run_id>.json`; also insert into `eval_runs` when `conn` is given."""
    if conn is not None and index_version is None and schema_name is None:
        raise ValueError("eval_runs needs index_version or schema_name")
    run_id = new_run_id(kind)
    record = {
        "run_id": run_id,
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset_version,
        "index_version": index_version,
        "schema_name": schema_name,
        "config": config,
        "metrics": metrics,
    }
    path = Path(runs_dir) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    if conn is not None:
        with conn.transaction():
            conn.execute(
                "INSERT INTO eval_runs "
                "(run_id, dataset_version, index_version, schema_name, config, metrics) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)",
                (
                    run_id,
                    dataset_version,
                    index_version,
                    schema_name,
                    json.dumps({"kind": kind, **config}, default=str),
                    json.dumps(metrics, default=str),
                ),
            )
    return run_id, path
