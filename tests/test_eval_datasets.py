import json

import pytest

from nyaya.contracts import Route
from nyaya.eval.datasets import (
    DATASETS_DIR,
    AdversarialItem,
    DevItem,
    Gold,
    HandLabel,
    SmokeItem,
    read_jsonl,
    write_jsonl,
)


def test_dev_item_defaults_and_round_trip(tmp_path):
    item = DevItem(
        id="dev-0001",
        question="When can a court refuse bail?",
        type="answerable",
        gold=[Gold(path="2025_1_230_238", para_labels=["9", "10"])],
        source_held_chunk="2025_1_230_238#editorial#002",
    )
    assert [z.value for z in item.scored_zones] == ["judgment"]
    assert item.exclude == []
    path = write_jsonl(tmp_path / "d.jsonl", [item])
    line = json.loads(path.read_text(encoding="utf-8"))
    assert "filters" not in line
    assert line["source_held_chunk"] == "2025_1_230_238#editorial#002"
    assert read_jsonl(path, DevItem) == [item]


def test_dev_item_rules():
    with pytest.raises(ValueError, match="needs gold"):
        DevItem(id="a", question="q", type="answerable")
    with pytest.raises(ValueError, match="no gold"):
        DevItem(id="a", question="q", type="unanswerable", gold=[Gold(path="p")])
    with pytest.raises(ValueError, match="scored_zones"):
        DevItem(id="a", question="q", type="unanswerable", scored_zones=["header"])
    with pytest.raises(ValueError):
        DevItem(id="a", question="q", type="unanswerable", extra=1)


def test_read_jsonl_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "x.jsonl"
    row = {"id": "u1", "question": "q", "type": "unanswerable"}
    p.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        read_jsonl(p, DevItem)


def test_read_jsonl_names_the_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": "u1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"x\.jsonl:1"):
        read_jsonl(p, DevItem)


# --- the committed datasets -----------------------------------------------------


def test_phase1_smoke_dataset():
    rows = read_jsonl(DATASETS_DIR / "phase1_smoke.jsonl", SmokeItem)
    assert len(rows) == 10
    answerable = [r for r in rows if r.expected_path is not None]
    assert len(answerable) == 7
    by_q = {r.question: r for r in rows}
    assert by_q["2025 INSC 31"].expected_route is Route.CITATION_LOOKUP
    assert by_q["2025 INSC 31"].expected_path == "2025_1_230_238"
    assert by_q["anticipatory bail section 438"].expected_route is Route.HYBRID
    assert any(r.filters is not None and r.filters.act for r in rows)
    for r in answerable:
        assert r.expected_path.startswith("2025_")


def test_adversarial_dataset():
    rows = read_jsonl(DATASETS_DIR / "adversarial_v1.jsonl", AdversarialItem)
    assert 18 <= len(rows) <= 25
    assert {r.category for r in rows} == {"injection", "advice", "false_premise"}
    # G9: every advice-seeking or false-premise question expects a refusal or the notice
    for r in rows:
        if r.category in ("advice", "false_premise"):
            assert r.expect in ("refuse", "notice")


def test_unanswerable_dataset(repo_root):
    from nyaya.config import load_settings

    rows = read_jsonl(DATASETS_DIR / "unanswerable_v1.jsonl", DevItem)
    assert len(rows) == load_settings(env_file=None).devset.unanswerable_size
    assert all(r.type == "unanswerable" and not r.gold for r in rows)


def test_hand_labels_file_is_valid():
    rows = read_jsonl(DATASETS_DIR / "hand_labels_v1.jsonl", HandLabel)
    assert isinstance(rows, list)  # filled by the author in Phase 3


def test_dev_draft_is_valid_if_present():
    path = DATASETS_DIR / "dev_draft_v0.jsonl"
    if not path.exists():
        pytest.skip("no draft")
    rows = read_jsonl(path, DevItem)
    assert rows and all(r.type == "answerable" for r in rows)
    assert all(r.gold[0].para_labels for r in rows)
