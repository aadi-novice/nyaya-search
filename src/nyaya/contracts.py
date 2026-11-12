"""Shared contracts (design: "Shared contracts"). Change the design first.

Enum values on the wire and in the database are lower case; member names are
upper case. Every model rejects unknown fields.

`export_json_schema()` writes contracts/schema.json, which the web page uses
for its types: `uv run python -m nyaya.contracts`.
"""

from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import models_json_schema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "contracts" / "schema.json"


# --- enums -------------------------------------------------------------------


class Era(StrEnum):
    C = "c"
    D = "d"
    OUT_OF_SCOPE = "out_of_scope"


class Zone(StrEnum):
    """All five may appear in Zones.missing; only ISSUE, EDITORIAL and JUDGMENT on chunks."""

    HEADER = "header"
    ISSUE = "issue"
    EDITORIAL = "editorial"
    REFERENCES = "references"
    JUDGMENT = "judgment"


CHUNK_ZONES = frozenset({Zone.ISSUE, Zone.EDITORIAL, Zone.JUDGMENT})


class OpinionType(StrEnum):
    MAJORITY = "majority"
    CONCURRING = "concurring"
    DISSENTING = "dissenting"
    UNKNOWN = "unknown"


class Route(StrEnum):
    CITATION_LOOKUP = "citation_lookup"
    HYBRID = "hybrid"


class Status(StrEnum):
    ANSWERED = "answered"
    PASSAGES = "passages"
    REFUSED = "refused"
    NO_MATCH_WITH_FILTERS = "no_match_with_filters"
    ERROR = "error"


class UnverifiedReason(StrEnum):
    UNKNOWN_LABEL = "unknown_label"
    QUOTE_NOT_FOUND = "quote_not_found"
    QUOTE_LENGTH = "quote_length"
    UNCITED_CASE_NAME = "uncited_case_name"
    NO_CITATION = "no_citation"


ChunkFlag = Literal["recursive_split", "unnumbered", "zone_missing"]
Span = tuple[int, int]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_span(start: int, end: int, what: str) -> None:
    if start < 0 or end < start:
        raise ValueError(f"{what}: need 0 <= start <= end, got ({start}, {end})")


# --- parser -> zoner -----------------------------------------------------------


class ParsedDoc(Contract):
    path: str
    era: Era  # c | d only
    clean_text: str
    # (char_offset, page) at the start of each line; page = printed SCR page
    position_map: list[tuple[int, int]]
    pages: int  # number of PDF pages (stored as judgments.pages)
    clean_text_version: str

    @model_validator(mode="after")
    def _era(self) -> ParsedDoc:
        if self.era is Era.OUT_OF_SCOPE:
            raise ValueError("a ParsedDoc is era c or d; out-of-scope PDFs are skipped")
        return self


# --- zoner -> chunker ----------------------------------------------------------


class HeldPoint(Contract):
    text: str
    char_start: int
    char_end: int
    para_refs: list[str]

    @model_validator(mode="after")
    def _span(self) -> HeldPoint:
        _check_span(self.char_start, self.char_end, "HeldPoint")
        return self


class Zones(Contract):
    """All offsets point into ParsedDoc.clean_text."""

    header_span: Span
    issue_span: Span | None = None
    editorial: list[HeldPoint]
    acts: list[str]
    keywords: list[str]
    judgment_span: Span
    missing: list[Zone]


class Opinion(Contract):
    author: str | None = None
    type: OpinionType
    char_start: int
    char_end: int

    @model_validator(mode="after")
    def _span(self) -> Opinion:
        _check_span(self.char_start, self.char_end, "Opinion")
        return self


# --- chunker -> loader -----------------------------------------------------------


class ChunkRecord(Contract):
    chunk_id: str  # "<path>#<zone>#<seq>", e.g. "2025_1_230_238#judgment#014"
    judgment_id: str
    zone: Zone  # issue | editorial | judgment
    opinion_author: str | None = None
    opinion_type: OpinionType | None = None
    para_labels: list[str]
    para_refs: list[str]  # editorial only: paragraphs named by "[Para N]"
    section_label: str | None = None
    text: str  # clean text; the quote check runs on this
    header: str  # embedded and keyword-indexed, never quoted
    char_start: int
    char_end: int
    pages: list[int]
    flags: list[ChunkFlag]
    clean_text_version: str
    chunker_version: str

    @model_validator(mode="after")
    def _check(self) -> ChunkRecord:
        if self.zone not in CHUNK_ZONES:
            raise ValueError(f"zone {self.zone.value!r} cannot appear on a chunk")
        prefix = f"{self.judgment_id}#{self.zone.value}#"
        if not self.chunk_id.startswith(prefix) or len(self.chunk_id) == len(prefix):
            raise ValueError(f"chunk_id must look like '{prefix}<seq>', got {self.chunk_id!r}")
        _check_span(self.char_start, self.char_end, "ChunkRecord")
        return self


# --- judgments -------------------------------------------------------------------


class JudgmentRecord(Contract):
    """Returned by get_judgment, served by GET /judgment/{path}.

    Rows without a PDF (has_pdf = false) have null era, pages, clean_text and
    position_map.
    """

    path: str
    title: str
    petitioner: str
    respondent: str
    judges: list[str]
    author_judge: str | None = None
    scr_citation: str
    neutral_citation: str | None = None
    cnr: str
    disposal: str
    outcome: str
    decision_date: date
    decision_year: int
    era: Era | None
    has_pdf: bool
    pages: int | None
    clean_text: str | None = None
    position_map: list[tuple[int, int]] | None = None


# --- answers ---------------------------------------------------------------------


class PassageRef(Contract):
    """Shown with refusals and in `passages` responses."""

    chunk_id: str
    judgment_id: str
    case_name: str
    scr_citation: str
    neutral_citation: str | None = None
    paras: list[str] | None = None
    pages: list[int]
    zone: Zone
    opinion_type: OpinionType | None = None


class Citation(PassageRef):
    """One verified citation on a sentence: every PassageRef field, plus these."""

    passage_label: str  # e.g. "P3"
    quote: str  # the matched quote


class AnswerSentence(Contract):
    sentence: str
    citations: list[Citation]


class UnverifiedSentence(Contract):
    """Never carries citation details: only the labels the LLM gave."""

    sentence: str
    labels: list[str]
    reason: UnverifiedReason


# --- API -------------------------------------------------------------------------


class Filters(Contract):
    year_from: int | None = None
    year_to: int | None = None
    judge: str | None = None
    act: str | None = None
    outcome: str | None = None

    @model_validator(mode="after")
    def _years(self) -> Filters:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not be after year_to")
        return self


class AskRequest(Contract):
    question: str = Field(min_length=1)
    filters: Filters | None = None


class Versions(Contract):
    index_version: str
    config_version: str
    prompt_version: str
    llm_model: str
    reranker_model: str
    reranker_fallback_used: bool
    embed_model: str


class AskResponse(Contract):
    request_id: str
    status: Status
    answer: list[AnswerSentence] = Field(default_factory=list)  # verified sentences only
    unverified: list[UnverifiedSentence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)  # "the passages do not address ..."
    passages: list[PassageRef] = Field(default_factory=list)  # status = passages
    nearest_passages: list[PassageRef] = Field(default_factory=list)  # status = refused
    passages_note: str | None = None  # e.g. "no summary generated (LLM unavailable)"
    notices: list[str]
    route: Route
    filters_applied: dict[str, Any]
    versions: Versions
    timings_ms: dict[str, float]  # per stage


class SearchPassage(PassageRef):
    """One ranked passage from POST /search: every PassageRef field, plus these."""

    rank: int  # 1 = best
    rerank_score: float | None = None  # None on the citation-lookup route
    text: str  # the chunk's clean text (the text quotes are checked against)
    decision_date: date | None = None


class SearchResponse(Contract):
    """POST /search: the ranked passages /ask would answer from, with no LLM call.

    status: `passages` (top score at or above the refusal threshold, or a
    citation lookup); `refused` (no candidates, or the top score is below the
    threshold: the passages are then the nearest ones); `no_match_with_filters`;
    `error`.
    """

    request_id: str
    status: Status
    passages: list[SearchPassage] = Field(default_factory=list)
    passages_note: str | None = None
    top_score: float | None = None
    threshold: float | None = None  # the refusal threshold in force (hybrid route)
    notices: list[str]
    route: Route
    filters_applied: dict[str, Any]
    versions: Versions
    timings_ms: dict[str, float]


# --- search ----------------------------------------------------------------------


class SearchHit(Contract):
    chunk_id: str
    keyword_rank: int | None = None
    vector_rank: int | None = None
    fused_rank: float  # link-expanded chunks sit at parent rank + 0.5
    rerank_score: float | None = None


# --- JSON Schema -----------------------------------------------------------------

SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    ParsedDoc,
    Zones,
    HeldPoint,
    Opinion,
    ChunkRecord,
    PassageRef,
    Citation,
    JudgmentRecord,
    AnswerSentence,
    UnverifiedSentence,
    AskRequest,
    AskResponse,
    SearchPassage,
    SearchResponse,
    Filters,
    Versions,
    SearchHit,
)


def build_json_schema() -> dict[str, Any]:
    """One JSON Schema document with every contract under `$defs`."""
    _, schema = models_json_schema(
        [(m, "serialization") for m in SCHEMA_MODELS],
        title="Nyaya Search contracts",
    )
    return schema


def export_json_schema(path: Path | str = SCHEMA_PATH) -> Path:
    """Write the JSON Schema to `path` (default contracts/schema.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_json_schema(), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    print(f"wrote {export_json_schema()}")
