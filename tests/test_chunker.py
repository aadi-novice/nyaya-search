import itertools
import re

import pytest

from judgment_fixtures import ERA_C, ERA_C_TWO_OPINIONS, ERA_D, load, parsed_from_text
from nyaya.chunk.chunker import (
    chunk,
    count_words,
    display_title,
    find_paragraphs,
    make_header,
    pages_for,
    split_sentences,
)
from nyaya.config import load_settings
from nyaya.contracts import Era, Opinion, OpinionType, Zone, Zones
from nyaya.zone.opinions import split_opinions
from nyaya.zone.zoner import zone


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


META = {"title": "A versus B", "decision_year": 2025}


def sentences(n_words: int, tag: str = "w") -> str:
    """Sentences of 20 words each (the last one shorter), n_words in all."""
    words = [f"{tag}{i}" for i in range(n_words)]
    out = []
    for k in range(0, n_words, 20):
        grp = words[k : k + 20]
        grp[0] = grp[0].capitalize()
        out.append(" ".join(grp) + ".")
    return " ".join(out)


def para(label: str, n_words: int) -> str:
    # the label counts as one word
    return f"{label}. {sentences(n_words - 1, 'p' + label.replace('.', '_') + 'x')}"


def make_doc(body: str, era=Era.D, path="2025_1_100_120"):
    text = "Title\n[P. Smith, J.]\nJudgment / Order of the Supreme Court\n" + body
    return parsed_from_text(path, era, text, lines_per_page=3)


def run(doc, settings, meta=META):
    z = zone(doc, settings)
    ops = split_opinions(doc, z, [], settings)
    return z, ops, chunk(meta, doc, z, ops, settings)


def judgment_chunks(chunks):
    return [c for c in chunks if c.zone is Zone.JUDGMENT]


def test_merges_short_paragraphs_up_to_target(settings):
    body = "P. Smith, J.\n" + "\n".join(para(str(i), 150) for i in range(1, 7)) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    assert [c.para_labels for c in jc] == [["1", "2"], ["3", "4"], ["5", "6"]]
    for c in jc:
        assert 80 <= count_words(c.text) <= 400
        assert c.flags == []
        assert c.opinion_author == "P. Smith, J."
        assert c.opinion_type is OpinionType.MAJORITY


def test_text_is_exact_slice_and_ids(settings):
    body = "P. Smith, J.\n" + "\n".join(para(str(i), 150) for i in range(1, 4)) + "\n"
    doc = make_doc(body)
    _, _, chunks = run(doc, settings)
    for i, c in enumerate(judgment_chunks(chunks), start=1):
        assert doc.clean_text[c.char_start : c.char_end] == c.text
        assert c.chunk_id == f"2025_1_100_120#judgment#{i:03d}"
        assert c.chunker_version == settings.versions.chunker
        assert c.clean_text_version == "ct1"


def test_long_paragraph_split_with_one_sentence_overlap(settings):
    body = "P. Smith, J.\n" + para("1", 120) + "\n" + para("2", 1500) + "\n" + para("3", 120) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    pieces = [c for c in jc if "recursive_split" in c.flags]
    assert len(pieces) >= 4
    assert all("2" in c.para_labels for c in pieces)
    for c in jc:
        assert count_words(c.text) <= settings.chunk.max_words
    # consecutive pieces share exactly one sentence
    for a, b in itertools.pairwise(pieces):
        last_sentence = re.findall(r"[A-Z][^.]*\.", a.text)[-1]
        assert b.text.startswith(last_sentence)
        assert a.char_end > b.char_start


def test_sub_paragraphs(settings):
    body = (
        "P. Smith, J.\n"
        + para("12", 90)
        + "\n"
        + para("12.1", 90).replace("12.1.", "12.1", 1)
        + "\n"
        + para("12.2", 90).replace("12.2.", "12.2", 1)
        + "\n"
        + para("13", 300)
        + "\n"
    )
    doc = make_doc(body)
    paras = find_paragraphs(doc.clean_text, 0, len(doc.clean_text))
    assert [p.label for p in paras] == ["12", "12.1", "12.2", "13"]
    _, _, chunks = run(doc, settings)
    assert [c.para_labels for c in judgment_chunks(chunks)] == [["12", "12.1", "12.2"], ["13"]]


def test_two_opinions_never_share_a_chunk(settings):
    body = (
        "P. Smith, J.\n"
        + para("1", 100)
        + "\n"
        + para("2", 50)
        + "\nQ. Jones, J.\n1. I regret I am unable to agree. "
        + sentences(60)
        + "\n"
        + para("2", 100)
        + "\n"
    )
    text = "Title\n[P. Smith and Q. Jones, JJ.]\nJudgment / Order of the Supreme Court\n" + body
    doc = parsed_from_text("2025_1_100_120", Era.D, text)
    _, ops, chunks = run(doc, settings)
    assert [o.type for o in ops] == [OpinionType.UNKNOWN, OpinionType.DISSENTING]
    jc = judgment_chunks(chunks)
    assert [(c.opinion_author, c.para_labels) for c in jc] == [
        ("P. Smith, J.", ["1", "2"]),
        ("Q. Jones, J.", ["1", "2"]),
    ]
    assert "Q. Jones, J. (dissenting)" in jc[1].header


def test_unnumbered_blank_line_paragraphs(settings):
    body = "ORDER\n\n" + sentences(150) + "\n\n" + sentences(150) + "\n\n" + sentences(200) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    assert len(jc) == 2
    assert all(c.flags == ["unnumbered"] and c.para_labels == [] for c in jc)


def test_unnumbered_single_block_sentence_windows(settings):
    body = sentences(1000) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    assert len(jc) == 3
    assert all("unnumbered" in c.flags for c in jc)
    assert all(count_words(c.text) <= settings.chunk.target_words for c in jc)


def test_short_last_chunk_merged_into_neighbour(settings):
    body = "P. Smith, J.\n" + para("1", 350) + "\n" + para("2", 30) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    assert [c.para_labels for c in judgment_chunks(chunks)] == [["1", "2"]]


def test_short_chunk_kept_when_merge_would_exceed_max(settings):
    body = "P. Smith, J.\n" + para("1", 690) + "\n" + para("2", 30) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    assert [c.para_labels for c in jc] == [["1"], ["2"]]


def test_short_first_chunk_merged_forward(settings):
    body = "P. Smith, J.\n" + para("1", 30) + "\n" + para("2", 390) + "\n" + para("3", 300) + "\n"
    _, _, chunks = run(make_doc(body), settings)
    assert [c.para_labels for c in judgment_chunks(chunks)] == [["1", "2"], ["3"]]


def test_section_labels(settings):
    body = (
        "P. Smith, J.\nFACTUAL ASPECT\n"
        + para("1", 300)
        + "\n"
        + para("2", 300)
        + "\nANALYSIS\n"
        + para("3", 300)
        + "\n"
    )
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    assert [(c.para_labels, c.section_label) for c in jc] == [
        (["1"], "FACTUAL ASPECT"),
        (["2"], "FACTUAL ASPECT"),
        (["3"], "ANALYSIS"),
    ]


def test_wrapped_number_without_full_stop_is_not_a_paragraph(settings):
    text = "1. Under Section\n13 of the Act the court may act.\n2. Next paragraph.\n"
    assert [p.label for p in find_paragraphs(text, 0, len(text))] == ["1", "2"]


def test_out_of_order_number_is_not_a_paragraph(settings):
    text = "1. First.\n2. Second quoting a rule:\n1. Short title.\n3. Third.\n40. Too far.\n"
    assert [p.label for p in find_paragraphs(text, 0, len(text))] == ["1", "2", "3"]


def test_paragraph_skip_tolerated(settings):
    text = "1. First.\n2. Second.\n5. Fifth (3 and 4 lost by the parser).\n6. Sixth.\n"
    assert [p.label for p in find_paragraphs(text, 0, len(text))] == ["1", "2", "5", "6"]


def test_header_format():
    h = make_header(
        "Jit Vinayak Arolkar v. State of Goa",
        2025,
        Zone.JUDGMENT,
        "Abhay S. Oka, J.",
        OpinionType.MAJORITY,
        ["9", "10"],
        ["penal code, 1860"],
    )
    assert h == (
        "Jit Vinayak Arolkar v. State of Goa (2025) · Judgment · Abhay S. Oka, J. · "
        "paras 9–10 · penal code, 1860"
    )
    h = make_header("A v. B", 2019, Zone.EDITORIAL, None, None, [], [])
    assert h == "A v. B (2019) · Headnote (editorial)"
    h = make_header("A v. B", None, Zone.ISSUE, None, None, [], ["a", "b", "c", "d"])
    assert h == "A v. B · Issue · a; b; c"
    h = make_header("", 2019, Zone.JUDGMENT, "X, J.", OpinionType.CONCURRING, ["3"], [])
    assert h == "(2019) · Judgment · X, J. (concurring) · paras 3–3"


def test_display_title():
    assert (
        display_title("K.P. MANU versus CHAIRMAN,  SCRUTINY") == "K.P. MANU v. CHAIRMAN, SCRUTINY"
    )
    assert display_title(None) == ""


def test_split_sentences_abbreviations(settings):
    text = (
        "The appeal u/s. 302 was filed by Ors. against the State. "
        "R.F. Nariman, J. agreed with i.e. the view. Next one? Yes! Last"
    )
    spans = split_sentences(text, 0, len(text), settings.chunk.abbreviations)
    got = [text[a:b] for a, b in spans]
    assert got == [
        "The appeal u/s. 302 was filed by Ors. against the State.",
        "R.F. Nariman, J. agreed with i.e. the view.",
        "Next one?",
        "Yes!",
        "Last",
    ]


def test_pages_for():
    pm = [(0, 10), (5, 10), (12, 11), (20, 12)]
    assert pages_for(pm, 0, 4) == [10]
    assert pages_for(pm, 6, 15) == [10, 11]
    assert pages_for(pm, 13, 30) == [11, 12]
    assert pages_for([], 0, 3) == []


def test_issue_and_editorial_chunks_era_d_fixture(settings):
    doc = load(ERA_D)
    meta = {"title": "ARUNKUMAR H SHAH HUF versus AVON ARCADE", "decision_date": "21-04-2025"}
    z, _, chunks = run(doc, settings, meta)
    zones_seq = [c.zone for c in chunks]
    assert zones_seq[0] is Zone.ISSUE
    assert zones_seq[1:4] == [Zone.EDITORIAL] * 3
    assert all(zs is Zone.JUDGMENT for zs in zones_seq[4:])
    issue = chunks[0]
    assert issue.chunk_id == "2025_4_1350_1372#issue#001"
    assert issue.header.startswith("ARUNKUMAR H SHAH HUF v. AVON ARCADE (2025) · Issue · ")
    ed = chunks[1:4]
    assert [c.para_refs for c in ed] == [hp.para_refs for hp in z.editorial]
    assert [c.chunk_id for c in ed] == [f"2025_4_1350_1372#editorial#{i:03d}" for i in (1, 2, 3)]
    assert all(c.para_labels == [] and c.opinion_type is None for c in ed)
    jc = judgment_chunks(chunks)
    assert jc[0].section_label == "FACTUAL DETAILS"
    assert jc[0].para_labels[0] == "1"
    assert "Abhay S. Oka, J. · paras 1–" in jc[0].header
    assert jc[0].header.endswith("registration act, 1908")
    assert all(c.pages for c in chunks)


def test_era_c_fixture_paragraph_labels(settings):
    doc = load(ERA_C)
    _, _, chunks = run(doc, settings, {"title": "U.P.P.S.C. versus RAHUL SINGH"})
    jc = judgment_chunks(chunks)
    labels = [lab for c in jc for lab in c.para_labels]
    assert labels == [str(i) for i in range(1, 10)]
    assert jc[0].text.startswith("DEEPAK GUPTA, J. 1. Applications")
    assert len([c for c in chunks if c.zone is Zone.EDITORIAL]) == 3


def test_era_c_two_opinions_fixture(settings):
    doc = load(ERA_C_TWO_OPINIONS)
    _, _, chunks = run(doc, settings, {"title": "DHAL SINGH DEWANGAN versus STATE"})
    jc = judgment_chunks(chunks)
    authors = [c.opinion_author for c in jc]
    assert authors[0] == "UDAY UMESH LALIT, J." and authors[-1] == "PRAFULLA C. PANT, J."
    pant = [c for c in jc if c.opinion_author == "PRAFULLA C. PANT, J."]
    assert [lab for c in pant for lab in c.para_labels] == ["1", "2"]


def test_zone_missing_flag_and_no_opinions(settings):
    text = "Plain text with no zones.\n1. A numbered paragraph " + sentences(100) + "\n"
    doc = parsed_from_text("2025_1_1_2", Era.D, text)
    z = zone(doc, settings)
    chunks = chunk(META, doc, z, [], settings)
    assert chunks and all(c.flags == ["zone_missing"] for c in chunks)
    assert all(c.opinion_type is None for c in chunks)


def test_empty_opinion_span_gives_no_chunks(settings):
    doc = parsed_from_text("2025_1_1_2", Era.D, "x\n   \n")
    z = Zones(
        header_span=(0, 0), editorial=[], acts=[], keywords=[], judgment_span=(1, 5), missing=[]
    )
    op = Opinion(author=None, type=OpinionType.MAJORITY, char_start=1, char_end=5)
    assert chunk(META, doc, z, [op], settings) == []


def test_huge_single_sentence_hard_split(settings):
    words = " ".join(f"w{i}" for i in range(1600))
    body = "P. Smith, J.\n1. " + words + ".\n"
    _, _, chunks = run(make_doc(body), settings)
    jc = judgment_chunks(chunks)
    assert len(jc) >= 3
    assert all(count_words(c.text) <= settings.chunk.max_words for c in jc)
    assert all("recursive_split" in c.flags for c in jc)


def test_meta_object_and_date(settings):
    class M:
        title = "X versus Y"
        decision_year = None
        decision_date = __import__("datetime").date(2011, 5, 1)

    body = "P. Smith, J.\n" + para("1", 120) + "\n"
    _, _, chunks = run(make_doc(body), settings, M())
    assert chunks[0].header.startswith("X v. Y (2011) · Judgment")
    _, _, chunks = run(make_doc(body), settings, {"title": "X versus Y"})
    assert chunks[0].header.startswith("X v. Y · Judgment")
    _, _, chunks = run(make_doc(body), settings, {"title": "X", "decision_date": "n/a"})
    assert chunks[0].header.startswith("X · Judgment")


def _quoted_para_judgment() -> str:
    """Shaped like 2025_10_631_650: para 12 quotes "22." from an earlier judgment,
    and para 10 quotes "71."; the judgment's own paras 13-25 follow."""
    lines = []
    for n in range(1, 26):
        lines.append(f"{n}. Paragraph {n} of this judgment discusses the point in some words.")
        if n == 10:
            lines.append("71. In Minor Irrigation v. Sahngoo Ram Arya this Court held so.")
        if n == 12:
            lines.append("22. It may be kept in mind that the public order is a State subject.")
    return "\n".join(lines) + "\n"


def test_quoted_numbered_paragraph_does_not_capture_later_labels():
    text = _quoted_para_judgment()
    labels = [p.label for p in find_paragraphs(text, 0, len(text))]
    assert labels == [str(n) for n in range(1, 26)]
    paras = {p.label: p for p in find_paragraphs(text, 0, len(text))}
    # the quoted "22." stays inside para 12; the real para 22 is its own paragraph
    assert "22. It may be kept in mind" in text[paras["12"].start : paras["12"].end]
    assert text[paras["22"].start :].startswith("22. Paragraph 22 of this judgment")
    assert text[paras["17"].start :].startswith("17. Paragraph 17")


def test_a_lost_paragraph_is_still_skipped_over():
    text = "1. One here.\n2. Two here.\n5. Five, after the parser lost 3 and 4.\n6. Six.\n"
    assert [p.label for p in find_paragraphs(text, 0, len(text))] == ["1", "2", "5", "6"]


def test_an_over_long_held_point_is_split_and_keeps_its_links(settings):
    from nyaya.contracts import HeldPoint, Zones

    sentence = "The appellant was rightly convicted on the evidence of the witnesses. "
    body = sentence * 150  # about 1,650 words
    text = "HELD: " + body + "[Para 12]\n12. The judgment paragraph.\n"
    end = text.index("\n12.")
    zones = Zones(
        header_span=(0, 0),
        editorial=[HeldPoint(text=text[6:end], char_start=6, char_end=end, para_refs=["12"])],
        acts=[],
        keywords=[],
        judgment_span=(end + 1, len(text)),
        missing=[],
    )
    doc = parsed_from_text("2015_1_1_9", Era.C, text)
    chunks = chunk({"title": "A v. B", "decision_year": 2015}, doc, zones, [], settings)
    ed = [c for c in chunks if c.zone is Zone.EDITORIAL]
    assert len(ed) > 2
    assert all(len(c.text.split()) <= settings.chunk.max_words for c in ed)
    assert all(c.para_refs == ["12"] and "recursive_split" in c.flags for c in ed)


def test_ocr_specks_and_margin_letters_before_a_number_are_tolerated():
    text = "1. One.\n· 2. Two.\n.. . 3. Three.\n·A 4. Four.\nB 5. Five.\n"
    assert [p.label for p in find_paragraphs(text, 0, len(text))] == ["1", "2", "3", "4", "5"]
