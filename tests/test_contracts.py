import json
from datetime import date

import pytest
from pydantic import ValidationError

from nyaya import contracts as c

SCHEMA_PATH = c.SCHEMA_PATH


def test_schema_json_is_current():
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == c.build_json_schema(), (
        "contracts/schema.json is stale: run `uv run python -m nyaya.contracts`"
    )


def test_export_json_schema_writes_file(tmp_path):
    out = c.export_json_schema(tmp_path / "schema.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    for name in (
        "ParsedDoc",
        "Zones",
        "HeldPoint",
        "Opinion",
        "ChunkRecord",
        "PassageRef",
        "Citation",
        "JudgmentRecord",
        "AnswerSentence",
        "UnverifiedSentence",
        "AskRequest",
        "AskResponse",
        "SearchPassage",
        "SearchResponse",
        "Filters",
        "Versions",
        "SearchHit",
    ):
        assert name in data["$defs"], name


@pytest.mark.parametrize(
    ("enum", "values"),
    [
        (c.Era, {"b", "c", "d", "out_of_scope"}),
        (c.Zone, {"header", "issue", "editorial", "references", "judgment"}),
        (c.OpinionType, {"majority", "concurring", "dissenting", "unknown"}),
        (c.Route, {"citation_lookup", "hybrid"}),
        (
            c.Status,
            {"answered", "passages", "refused", "no_match_with_filters", "error"},
        ),
        (
            c.UnverifiedReason,
            {
                "unknown_label",
                "quote_not_found",
                "quote_length",
                "uncited_case_name",
                "no_citation",
            },
        ),
    ],
)
def test_enum_wire_values_are_lower_case(enum, values):
    assert {m.value for m in enum} == values
    for m in enum:
        assert m.name == m.value.upper()


def chunk(**kw):
    base = {
        "chunk_id": "2025_1_230_238#judgment#014",
        "judgment_id": "2025_1_230_238",
        "zone": "judgment",
        "opinion_author": "ABHAY S OKA",
        "opinion_type": "majority",
        "para_labels": ["9", "10"],
        "para_refs": [],
        "section_label": None,
        "text": "9. The power ...",
        "header": "Jit Vinayak Arolkar v. State of Goa (2025) · Judgment",
        "char_start": 100,
        "char_end": 400,
        "pages": [232, 233],
        "flags": [],
        "clean_text_version": "ct1",
        "chunker_version": "ch1",
    }
    base.update(kw)
    return base


def test_chunk_record_round_trip():
    rec = c.ChunkRecord.model_validate(chunk())
    assert rec.zone is c.Zone.JUDGMENT
    dumped = rec.model_dump(mode="json")
    assert dumped["zone"] == "judgment"
    assert dumped["opinion_type"] == "majority"
    assert c.ChunkRecord.model_validate_json(rec.model_dump_json()) == rec


@pytest.mark.parametrize("zone", ["header", "references"])
def test_chunk_zone_limited_to_issue_editorial_judgment(zone):
    with pytest.raises(ValidationError):
        c.ChunkRecord.model_validate(chunk(zone=zone, chunk_id=f"2025_1_230_238#{zone}#001"))


def test_chunk_id_must_match_judgment_and_zone():
    with pytest.raises(ValidationError):
        c.ChunkRecord.model_validate(chunk(chunk_id="2025_1_1_9#judgment#001"))
    with pytest.raises(ValidationError):
        c.ChunkRecord.model_validate(chunk(chunk_id="2025_1_230_238#issue#001"))


def test_chunk_flags_are_known_values():
    c.ChunkRecord.model_validate(chunk(flags=["recursive_split", "unnumbered", "zone_missing"]))
    with pytest.raises(ValidationError):
        c.ChunkRecord.model_validate(chunk(flags=["something_else"]))


def test_chunk_offsets_ordered():
    with pytest.raises(ValidationError):
        c.ChunkRecord.model_validate(chunk(char_start=50, char_end=10))


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        c.ChunkRecord.model_validate(chunk(extra_field=1))


def test_parsed_doc_era_is_c_or_d():
    doc = {
        "path": "2025_1_230_238",
        "era": "d",
        "clean_text": "text",
        "position_map": [[0, 230], [120, 231]],
        "pages": 9,
        "clean_text_version": "ct1",
    }
    parsed = c.ParsedDoc.model_validate(doc)
    assert parsed.position_map[1] == (120, 231)
    with pytest.raises(ValidationError):
        c.ParsedDoc.model_validate({**doc, "era": "out_of_scope"})


def test_zones_and_opinion():
    z = c.Zones.model_validate(
        {
            "header_span": [0, 100],
            "issue_span": None,
            "editorial": [
                {"text": "Held: ...", "char_start": 100, "char_end": 200, "para_refs": ["9"]}
            ],
            "acts": ["penal code, 1860"],
            "keywords": [],
            "judgment_span": [300, 9000],
            "missing": ["issue", "references"],
        }
    )
    assert z.missing == [c.Zone.ISSUE, c.Zone.REFERENCES]
    op = c.Opinion.model_validate(
        {"author": None, "type": "dissenting", "char_start": 300, "char_end": 900}
    )
    assert op.type is c.OpinionType.DISSENTING
    with pytest.raises(ValidationError):
        c.Opinion.model_validate({"author": "X", "type": "partly", "char_start": 0, "char_end": 1})


def test_judgment_record_without_pdf():
    rec = c.JudgmentRecord.model_validate(
        {
            "path": "S_2005_5_75_82",
            "title": "A v. B",
            "petitioner": "A",
            "respondent": "B",
            "judges": ["SURYA KANT"],
            "author_judge": None,
            "scr_citation": "[2005] SUPP. 5 S.C.R. 75",
            "neutral_citation": None,
            "cnr": "ESCR010004242005",
            "disposal": "",
            "outcome": "other",
            "decision_date": "2005-03-01",
            "decision_year": 2005,
            "era": None,
            "has_pdf": False,
            "pages": None,
            "clean_text": None,
            "position_map": None,
        }
    )
    assert rec.decision_date == date(2005, 3, 1)
    assert rec.model_dump(mode="json")["decision_date"] == "2005-03-01"


def passage(**kw):
    base = {
        "chunk_id": "2025_1_230_238#judgment#014",
        "judgment_id": "2025_1_230_238",
        "case_name": "Jit Vinayak Arolkar v. State of Goa",
        "scr_citation": "[2025] 1 S.C.R. 230",
        "neutral_citation": "2025INSC31",
        "paras": ["9", "10"],
        "pages": [232],
        "zone": "judgment",
        "opinion_type": "majority",
    }
    base.update(kw)
    return base


def test_citation_extends_passage_ref():
    cit = c.Citation.model_validate(
        passage(passage_label="P3", quote="must be exercised sparingly and with circumspection")
    )
    assert isinstance(cit, c.PassageRef)
    assert set(c.PassageRef.model_fields) < set(c.Citation.model_fields)


def test_unverified_sentence_carries_no_citation_details():
    u = c.UnverifiedSentence(sentence="x", labels=["P9"], reason="unknown_label")
    assert set(u.model_dump()) == {"sentence", "labels", "reason"}
    with pytest.raises(ValidationError):
        c.UnverifiedSentence.model_validate(
            {"sentence": "x", "labels": [], "reason": "no_citation", "citations": []}
        )


def versions():
    return {
        "index_version": "v2025_01",
        "config_version": "cfg1-abc",
        "prompt_version": "answer_v1",
        "llm_model": "some-model",
        "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
        "reranker_fallback_used": False,
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
    }


def test_ask_request_and_response():
    req = c.AskRequest.model_validate(
        {"question": "anticipatory bail section 438", "filters": {"year_from": 2015}}
    )
    assert req.filters is not None and req.filters.year_from == 2015
    assert c.AskRequest.model_validate({"question": "q"}).filters is None
    with pytest.raises(ValidationError):
        c.AskRequest.model_validate({"question": "q", "filters": {"court": "HC"}})

    resp = c.AskResponse.model_validate(
        {
            "request_id": "r1",
            "status": "answered",
            "answer": [
                {
                    "sentence": "s",
                    "citations": [passage(passage_label="P1", quote="a b c d e f g h")],
                }
            ],
            "notices": ["This is not legal advice."],
            "route": "hybrid",
            "filters_applied": {},
            "versions": versions(),
            "timings_ms": {"search": 12.5},
        }
    )
    assert resp.unverified == [] and resp.gaps == [] and resp.passages == []
    assert resp.nearest_passages == [] and resp.passages_note is None
    wire = resp.model_dump(mode="json")
    assert wire["status"] == "answered" and wire["route"] == "hybrid"


def test_filters_year_order():
    with pytest.raises(ValidationError):
        c.Filters(year_from=2020, year_to=2010)


def test_search_hit():
    hit = c.SearchHit(chunk_id="a#judgment#001", fused_rank=3.5)
    assert hit.keyword_rank is None and hit.vector_rank is None and hit.rerank_score is None
