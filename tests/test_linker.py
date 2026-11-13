import json

from judgment_fixtures import ERA_C, ERA_D, load
from nyaya.chunk.chunker import chunk
from nyaya.config import load_settings
from nyaya.contracts import ChunkRecord, OpinionType, Zone
from nyaya.link.linker import held_links, merge_summaries, resolve_held_links, write_link_summary
from nyaya.zone.opinions import split_opinions
from nyaya.zone.zoner import zone


def rec(path, zone_, seq, labels=(), refs=(), author=None, kind=None):
    return ChunkRecord(
        chunk_id=f"{path}#{zone_.value}#{seq:03d}",
        judgment_id=path,
        zone=zone_,
        opinion_author=author,
        opinion_type=kind,
        para_labels=list(labels),
        para_refs=list(refs),
        text="t",
        header="h",
        char_start=0,
        char_end=1,
        pages=[1],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch1",
    )


P = "2025_1_1_9"
E, J = Zone.EDITORIAL, Zone.JUDGMENT
MAJ, DIS = OpinionType.MAJORITY, OpinionType.DISSENTING


def test_links_by_label_and_sub_paragraph_prefix():
    chunks = [
        rec(P, E, 1, refs=["9", "10"]),
        rec(P, E, 2, refs=["12"]),
        rec(P, J, 1, labels=["9", "10"], kind=MAJ),
        rec(P, J, 2, labels=["12.1", "12.2"], kind=MAJ),
    ]
    assert held_links(chunks) == [
        (f"{P}#editorial#001", f"{P}#judgment#001"),
        (f"{P}#editorial#002", f"{P}#judgment#002"),
    ]


def test_sub_paragraph_ref_falls_back_to_whole_paragraph():
    chunks = [rec(P, E, 1, refs=["12.1"]), rec(P, J, 1, labels=["11", "12"], kind=MAJ)]
    assert held_links(chunks) == [(f"{P}#editorial#001", f"{P}#judgment#001")]


def test_unresolved_counted():
    chunks = [rec(P, E, 1, refs=["9", "40"]), rec(P, E, 2), rec(P, J, 1, labels=["9"], kind=MAJ)]
    links, s = resolve_held_links(chunks)
    assert links == [(f"{P}#editorial#001", f"{P}#judgment#001")]
    assert s["held_points"] == 2 and s["held_points_with_refs"] == 1
    assert (s["para_refs"], s["resolved"], s["unresolved"]) == (2, 1, 1)
    assert s["unresolved_refs"] == [{"chunk_id": f"{P}#editorial#001", "ref": "40"}]
    assert s["resolution_rate"] == 0.5


def test_links_stay_within_judgment():
    q = "2025_1_10_20"
    chunks = [rec(P, E, 1, refs=["9"]), rec(q, J, 1, labels=["9"], kind=MAJ)]
    links, s = resolve_held_links(chunks)
    assert links == [] and s["unresolved"] == 1


def test_label_in_two_opinions_prefers_majority():
    chunks = [
        rec(P, E, 1, refs=["2"]),
        rec(P, J, 1, labels=["1", "2"], author="A, J.", kind=DIS),
        rec(P, J, 2, labels=["1", "2"], author="B, J.", kind=MAJ),
    ]
    links, s = resolve_held_links(chunks)
    assert links == [(f"{P}#editorial#001", f"{P}#judgment#002")]
    assert s["ambiguous"] == 0


def test_label_in_two_unknown_opinions_links_both_and_counts_ambiguous():
    unk = OpinionType.UNKNOWN
    chunks = [
        rec(P, E, 1, refs=["2"]),
        rec(P, J, 1, labels=["2"], author="A, J.", kind=unk),
        rec(P, J, 2, labels=["2"], author="B, J.", kind=unk),
    ]
    links, s = resolve_held_links(chunks)
    assert len(links) == 2 and s["ambiguous"] == 1


def test_duplicate_refs_give_one_link():
    chunks = [rec(P, E, 1, refs=["9", "10"]), rec(P, J, 1, labels=["9", "10"], kind=MAJ)]
    assert len(held_links(chunks)) == 1


def test_no_refs_summary():
    links, s = resolve_held_links([])
    assert links == [] and s["resolution_rate"] is None


def test_merge_and_write(tmp_path):
    _, a = resolve_held_links([rec(P, E, 1, refs=["9"]), rec(P, J, 1, labels=["9"], kind=MAJ)])
    _, b = resolve_held_links([rec(P, E, 1, refs=["7"])])
    m = merge_summaries([a, b])
    assert (m["para_refs"], m["resolved"], m["unresolved"]) == (2, 1, 1)
    assert m["resolution_rate"] == 0.5
    assert merge_summaries([])["resolution_rate"] is None
    out = write_link_summary(m, tmp_path / "build" / "link_summary.json")
    assert json.loads(out.read_text(encoding="utf-8"))["resolved"] == 1


def test_fixture_links_resolve():
    settings = load_settings(env_file=None)
    doc = load(ERA_C)
    z = zone(doc, settings)
    chunks = chunk({"title": "x"}, doc, z, split_opinions(doc, z, [], settings), settings)
    links, s = resolve_held_links(chunks)
    # the excerpt stops after paragraph 9: refs 12, 14 and 15 are beyond it
    assert s["para_refs"] == 3 and s["unresolved"] == 3 and links == []
    doc = load(ERA_D)
    z = zone(doc, settings)
    chunks = chunk({"title": "x"}, doc, z, split_opinions(doc, z, [], settings), settings)
    _, s = resolve_held_links(chunks)
    assert (s["para_refs"], s["resolved"]) == (10, 0)  # excerpt holds paragraphs 1-4 only
