from __future__ import annotations

import pytest

from nyaya.answer.prompt import Passage, build_prompt, estimate_tokens, load_template
from nyaya.config import load_settings
from nyaya.contracts import ChunkRecord, OpinionType, PassageRef, Zone

SETTINGS = load_settings(env_file=None)


def passage(i: int, *, words: int = 20, zone=Zone.JUDGMENT, opinion=OpinionType.MAJORITY):
    jid = f"2025_1_{i}_9"
    cid = f"{jid}#{zone.value}#{i:03d}"
    text = " ".join(f"w{i}x{n}" for n in range(words))
    chunk = ChunkRecord(
        chunk_id=cid,
        judgment_id=jid,
        zone=zone,
        opinion_type=opinion,
        para_labels=["9", "10"] if zone is Zone.JUDGMENT else [],
        para_refs=[],
        text=text,
        header=f"HEADER-{i}",
        char_start=0,
        char_end=len(text),
        pages=[230],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch1",
    )
    ref = PassageRef(
        chunk_id=cid,
        judgment_id=jid,
        case_name=f"Case {i} v. State",
        scr_citation=f"[2025] 1 S.C.R. {i}",
        paras=chunk.para_labels or None,
        pages=[230],
        zone=zone,
        opinion_type=opinion,
    )
    return Passage(ref=ref, chunk=chunk)


def test_labels_blocks_and_rules():
    ps = [
        passage(1),
        passage(2, zone=Zone.EDITORIAL, opinion=None),
        passage(3, opinion=OpinionType.DISSENTING),
    ]
    system, user, label_map = build_prompt("Is bail a right?", ps, "answer_v1", SETTINGS)
    assert list(label_map) == ["P1", "P2", "P3"]
    assert label_map["P2"] == ps[1].ref
    assert "between 8 and\n   40 words" in system or "between 8 and 40 words" in system
    assert '"sentences"' in system and '"gaps"' in system
    assert "Is bail a right?" in user
    # block order: label, case name, zone, opinion type, paragraphs, then text
    block = user[user.index("[P1]") : user.index("[P2]")]
    order = [block.index(s) for s in ("[P1]", "Case 1 v. State", "Zone:", "Opinion:", "9, 10")]
    assert order == sorted(order)
    assert block.index("9, 10") < block.index("w1x0")
    assert "HEADER-1" not in user  # headers are never shown for quoting
    p2 = user[user.index("[P2]") : user.index("[P3]")]
    assert "editorial" in p2 and "Opinion: none" in p2 and "Paragraphs: none" in p2
    assert "dissenting" in user[user.index("[P3]") :]
    assert "{{" not in system + user


def test_budget_drops_lowest_ranked_passages():
    ps = [passage(i, words=600) for i in range(1, 9)]
    system, user, label_map = build_prompt("q", ps, "answer_v1", SETTINGS)
    assert estimate_tokens(system + user) <= SETTINGS.llm.max_input_tokens
    assert 1 <= len(label_map) < 8
    assert list(label_map) == [f"P{i}" for i in range(1, len(label_map) + 1)]


def test_budget_always_keeps_one_passage():
    ps = [passage(1, words=8000)]
    _, _, label_map = build_prompt("q", ps, "answer_v1", SETTINGS)
    assert list(label_map) == ["P1"]


def test_unknown_version_raises():
    with pytest.raises(FileNotFoundError):
        load_template("no_such_prompt")


def test_template_without_sections_raises(tmp_path):
    (tmp_path / "bad.md").write_text("no headings", encoding="utf-8")
    with pytest.raises(ValueError, match="## system"):
        load_template("bad", prompts_dir=tmp_path)
