import json

import pytest

from judgment_fixtures import ERA_C, ERA_D, load, parsed_from_text
from nyaya.chunk.chunker import chunk
from nyaya.chunk.metrics import (
    GATE_EXIT_CODE,
    doc_metrics,
    embedded_text,
    failing_metrics,
    over_model_limit,
    paragraph_integrity,
    size_compliance,
    summarize,
    write_chunk_metrics,
)
from nyaya.config import load_settings
from nyaya.contracts import Era, Opinion, OpinionType, Zones
from nyaya.zone.opinions import split_opinions
from nyaya.zone.zoner import zone


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def build(doc, settings):
    z = zone(doc, settings)
    ops = split_opinions(doc, z, [], settings)
    return z, ops, chunk({"title": "A v. B", "decision_year": 2018}, doc, z, ops, settings)


def words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


def test_fixture_metrics_pass(settings):
    docs = []
    for name in (ERA_C, ERA_D):
        doc = load(name)
        z, ops, chunks = build(doc, settings)
        docs.append(
            doc_metrics(doc, z, ops, chunks, settings, count_tokens=lambda s: len(s.split()))
        )
    s = summarize(docs, settings)
    assert s["paragraph_integrity"] == 1.0
    assert s["over_model_limit"] == 0 and s["tokens_counted"] is True
    assert s["counts"]["paragraphs_total"] >= 9
    assert failing_metrics(s, settings) == [] or failing_metrics(s, settings) == ["size_compliance"]


def test_paragraph_cut_by_boundary_is_counted(settings):
    text = "1. " + words(100, "a") + "\n2. " + words(100, "b") + "\n"
    doc = parsed_from_text("2025_1_1_2", Era.D, text)
    z = Zones(
        header_span=(0, 0),
        editorial=[],
        acts=[],
        keywords=[],
        judgment_span=(0, len(text)),
        missing=[],
    )
    ops = [Opinion(author=None, type=OpinionType.MAJORITY, char_start=0, char_end=len(text))]
    chunks = chunk({"title": "x"}, doc, z, ops, settings)
    assert paragraph_integrity(doc, z, ops, chunks, 700) == (2, 2)
    # a chunk boundary in the middle of paragraph 2
    cut = text.index("b50")
    broken = [
        chunks[0].model_copy(update={"char_end": cut, "text": text[:cut]}),
    ]
    assert paragraph_integrity(doc, z, ops, broken, 700) == (1, 2)
    # an over-long paragraph is left out
    assert paragraph_integrity(doc, z, ops, broken, 50) == (0, 0)
    # no opinions: the judgment span is used
    assert paragraph_integrity(doc, z, [], chunks, 700) == (2, 2)


def test_size_compliance_and_whole_opinion_exemption(settings):
    text = (
        "Title\n[P. Smith and Q. Jones, JJ.]\nJudgment / Order of the Supreme Court\n"
        "P. Smith, J.\n1. " + words(200, "a") + "\n2. " + words(30, "b") + "\n"
        "Q. Jones, J.\n1. I agree.\n"
    )
    doc = parsed_from_text("2025_1_1_2", Era.D, text)
    z, ops, chunks = build(doc, settings)
    size = size_compliance(z, ops, chunks, 80, 700)
    assert size == {
        "in_size": 1,
        "total": 1,
        "in_size_strict": 1,
        "total_strict": 2,
        "whole_opinion_short": 1,
    }


def test_over_model_limit_counts_header_and_text(settings):
    doc = load(ERA_C)
    _, _, chunks = build(doc, settings)
    assert over_model_limit(chunks, lambda s: len(s.split()), limit=10**6) == []
    big = over_model_limit(chunks, lambda s: len(s.split()), limit=50)
    assert big and all(len(embedded_text(c).split()) > 50 for c in chunks if c.chunk_id in big)
    assert embedded_text(chunks[0]).startswith(chunks[0].header + "\n\n")


def test_failing_metrics_and_write(tmp_path, settings):
    s = summarize([], settings)
    assert s["paragraph_integrity"] == 1.0 and s["size_compliance"] == 1.0
    assert s["judgment_chunk_words"] is None and s["tokens_counted"] is False
    assert failing_metrics(s, settings) == []
    bad = dict(s, paragraph_integrity=0.5, size_compliance=0.5, over_model_limit=2)
    assert failing_metrics(bad, settings) == [
        "paragraph_integrity",
        "size_compliance",
        "over_model_limit",
    ]
    assert GATE_EXIT_CODE == 3
    out = write_chunk_metrics(s, tmp_path / "build" / "chunk_metrics.json")
    assert json.loads(out.read_text(encoding="utf-8"))["judgments"] == 0


def test_doc_metrics_without_token_counter(settings):
    doc = load(ERA_D)
    z, ops, chunks = build(doc, settings)
    d = doc_metrics(doc, z, ops, chunks, settings)
    assert d["over_model_limit"] == [] and d["tokens_counted"] is False
    s = summarize([d], settings)
    assert s["judgment_chunk_words"]["min"] <= s["judgment_chunk_words"]["max"]
    assert s["editorial_chunk_words"] is not None
