"""Evaluation dataset formats (eval/datasets/README.md) and JSONL helpers.

These models describe files under eval/datasets/. They are evaluation-only
formats, so they live here rather than in contracts.py.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nyaya.contracts import CHUNK_ZONES, Filters, Route, Zone

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"
RUNS_DIR = REPO_ROOT / "eval" / "runs"


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Gold(_Row):
    """One gold judgment and the paragraph labels that answer the question."""

    path: str
    para_labels: list[str] = Field(default_factory=list)


class DevItem(_Row):
    """A development (or test, or unanswerable) question.

    `exclude` lists judgments removed from candidates before scoring, to
    prevent leakage. `scored_zones` lists the zones whose chunks count as hits.
    `source_held_chunk` is the editorial chunk the question was drafted from
    (None for author-written questions, and in drafts built before the index).
    """

    id: str
    question: str = Field(min_length=1)
    type: Literal["answerable", "unanswerable"]
    gold: list[Gold] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    scored_zones: list[Zone] = Field(default_factory=lambda: [Zone.JUDGMENT])
    source_held_chunk: str | None = None
    filters: Filters | None = None

    @model_validator(mode="after")
    def _check(self) -> DevItem:
        if self.type == "answerable" and not self.gold:
            raise ValueError(f"{self.id}: an answerable question needs gold")
        if self.type == "unanswerable" and self.gold:
            raise ValueError(f"{self.id}: an unanswerable question has no gold")
        bad = [z for z in self.scored_zones if z not in CHUNK_ZONES]
        if bad or not self.scored_zones:
            raise ValueError(f"{self.id}: scored_zones must be chunk zones, got {bad}")
        return self

    @property
    def gold_paths(self) -> set[str]:
        return {g.path for g in self.gold}


class SmokeItem(_Row):
    """phase1_smoke.jsonl: the load check and the Phase 1 verification."""

    id: str
    question: str = Field(min_length=1)
    expected_route: Route
    expected_path: str | None
    filters: Filters | None = None


class AdversarialItem(_Row):
    id: str
    question: str = Field(min_length=1)
    category: Literal["injection", "advice", "false_premise"]
    expect: Literal["refuse", "notice", "cited_only"]
    note: str | None = None


class HandLabel(_Row):
    """hand_labels_v1.jsonl: the author's labels for judge-validate (Phase 3)."""

    id: str
    sentence: str
    chunk_id: str
    label: Literal["supported", "partial", "unsupported"]
    labeller: str


R = TypeVar("R", bound=BaseModel)

_DROP_IF_NONE = ("filters", "note")


def read_jsonl[R: BaseModel](path: Path | str, model: type[R]) -> list[R]:
    """Read and validate a JSONL file. Blank lines are skipped; ids must be unique."""
    path = Path(path)
    rows: list[R] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = model.model_validate(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
            rid = getattr(row, "id", None)
            if rid is not None:
                if rid in seen:
                    raise ValueError(f"{path}:{n}: duplicate id {rid!r}")
                seen.add(rid)
            rows.append(row)
    return rows


def write_jsonl(path: Path | str, rows: Iterable[BaseModel]) -> Path:
    """Write rows as JSONL (UTF-8, LF). `filters` and `note` are left out when None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            data = row.model_dump(mode="json")
            for key in _DROP_IF_NONE:
                if key in data and data[key] is None:
                    del data[key]
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    return path
