"""Era B (printed SCR, 1964-2009): classification, parsing, zoning, opinions,
paragraphs and chunks.

Word- and line-level cases are synthetic (the shapes measured on the
1970-2009 sample in docs/era_b_findings.md); one small synthetic PDF runs the
whole parser; tests/fixtures/judgments/era_b_1976_3_795_800.txt is the first
104 lines of a real judgment as the Era B parser cleans it.
"""

from __future__ import annotations

from datetime import date

import pymupdf
import pytest

from corpus_helpers import era_c_pdf, plain_pdf
from judgment_fixtures import load, parsed_from_text
from nyaya import build as b
from nyaya.chunk.chunker import chunk, find_paragraphs, paragraph_rules
from nyaya.chunk.metrics import doc_metrics
from nyaya.config import load_settings
from nyaya.contracts import Era, OpinionType, ParsedDoc, Zone
from nyaya.parse.eras import (
    ERA_A_REASON,
    classify,
    classify_pdf,
    count_b_margin_lines,
    out_of_scope_reason,
    out_of_scope_reason_pdf,
)
from nyaya.parse.pdf_layout import (
    ColumnLine,
    Line,
    PageWords,
    Word,
    build_text,
    column_edges,
    era_b_head_ids,
    era_b_margin_ids,
    is_era_b_head,
    is_margin_token,
    paragraph_starts,
    parse,
)
from nyaya.zone.opinions import ocr_name_match, split_opinions
from nyaya.zone.zoner import title_run, zone

ERA_B = "era_b_1976_3_795_800"
W, H = 406.0, 634.0  # a 1970s SCR scan, in points


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


@pytest.fixture(scope="module")
def cfg(settings):
    return settings.era_b


# --- classification ----------------------------------------------------------------

ERA_B_FIRST_PAGES = "\n".join(
    [
        "NARINDRAJIT SINGH & ANR.",
        "v.",
        "STATE OF U.P. & ORS.",
        "October 24, 1972",
        "[A. N. GROVER AND K. K. MATHEW, JJ.]",
        "A",
        "Land Acquisition Act, 1894, sections 4(1) and (2)-",
        "B",
        # PyMuPDF puts most 1990-2009 letters on the body line
        "C  the High Court that the parties being real brothers should settle",
        "the matter among themselves through mediation. Accordingly,",
        "D  requested to be a mediator. On 17th March 2005 it was noted",
        "struction of grid separator, all vehicles would cross over  E",
        "HELB : (i) The law as settled by this Court is that such a notice",
    ]
)


def test_era_b_counts_letters_alone_or_at_a_line_end(settings):
    assert count_b_margin_lines(ERA_B_FIRST_PAGES, settings.era_markers) == 5
    assert classify(ERA_B_FIRST_PAGES, 1972, settings.era_markers) is Era.B
    assert classify(ERA_B_FIRST_PAGES, 2005, settings.era_markers) is Era.B


def test_era_b_needs_no_held_but_needs_the_years(settings):
    text = ERA_B_FIRST_PAGES.replace("HELB :", "It held")
    assert classify(text, 1990, settings.era_markers) is Era.B
    assert classify(text, 2012, settings.era_markers) is Era.OUT_OF_SCOPE


def test_era_a_is_out_of_scope_with_a_reason(settings):
    # 1950-1963: side notes, no margin letters (0-2 matching lines measured)
    era_a = "BHARAT BANK LTD.\nv.\nEMPLOYEES\nA notification was issued.\nHeld, that"
    m = settings.era_markers
    assert classify(era_a, 1955, m) is Era.OUT_OF_SCOPE
    assert out_of_scope_reason(era_a, 1955, m) == ERA_A_REASON
    assert out_of_scope_reason(era_a, 2015, m) == ""  # a 2015 short order: no known layout
    assert out_of_scope_reason_pdf(plain_pdf([era_a.splitlines()]), 1955, m) == ERA_A_REASON


def test_era_c_and_d_still_come_first(settings):
    c_pdf = era_c_pdf([["HELD: 1. The appeal is allowed.", "x y z"], ["more"], ["more"]])
    assert classify_pdf(c_pdf, 2015, settings.era_markers) is Era.C
    assert classify_pdf(c_pdf, 2005, settings.era_markers) is Era.B


# --- margin letters ----------------------------------------------------------------


def body_line(words: list[str], x0: float, y: float, block: int, line: int) -> list[Word]:
    out, x = [], x0
    for t in words:
        w = 5.0 * len(t)
        out.append(Word(x, y, x + w, y + 10, t, block, line))
        x += w + 4
    return out


BODY = [
    "the",
    "appellant",
    "filed",
    "a",
    "petition",
    "under",
    "Article",
    "226",
]  # ends at 72% of the page width


def page_with(extra: list[Word], n_lines: int = 12, x0: float = 55.0) -> PageWords:
    pw = PageWords(0, W, H)
    for i in range(n_lines):
        pw.words += body_line(BODY, x0, 80 + 18 * i, i, 0)
    pw.words += extra
    return pw


@pytest.mark.parametrize(
    "text, margin",
    [("B", True), ("c", True), ("B'", True), ("8", True), ("~-", True), (".", True)],
)
def test_margin_token_shapes(text, margin, cfg):
    letters = frozenset("ABCDEFGH") | frozenset(cfg.ocr_letters)
    assert is_margin_token(text, letters) is margin


@pytest.mark.parametrize("text", ["(c)", "(1)", "Rs.", "the", "Sch", "Q", "J1"])
def test_not_margin_tokens(text, cfg):
    assert not is_margin_token(text, frozenset("ABCDEFGH") | frozenset(cfg.ocr_letters))


def test_letters_left_and_right_of_the_column_are_removed(cfg):
    left = Word(20, 80, 28, 90, "D", 0, 0)  # on the first body line (same PyMuPDF line)
    right = Word(372, 116, 380, 126, "E", 2, 0)
    alone = Word(21, 300, 29, 310, "c", 90, 0)
    speck = Word(8, 400, 14, 404, "~-", 91, 0)
    pw = page_with([left, right, alone, speck])
    assert era_b_margin_ids(pw, cfg) == {id(left), id(right), id(alone), id(speck)}


def test_words_at_the_column_edge_are_kept(cfg):
    article = Word(55, 400, 60, 410, "A", 50, 0)  # "A notification ..." at the edge
    marker = Word(55, 420, 70, 430, "(c)", 51, 0)
    pw = page_with([article, marker])
    assert era_b_margin_ids(pw, cfg) == set()


def test_a_short_page_falls_back_to_the_gap_rule(cfg):
    letter = Word(20, 80, 28, 90, "B", 9, 0)
    speck = Word(10, 120, 14, 124, "~", 10, 0)
    pw = page_with([letter, speck], n_lines=2)  # too few lines to measure a column
    assert era_b_margin_ids(pw, cfg) == {id(letter)}


def test_column_edges_use_the_wider_of_local_and_page():
    # a quotation fills the lower half: its indent must not become the edge
    lines = [ColumnLine(0.1 + 0.04 * i, 0.12, 0.88) for i in range(8)]
    lines += [ColumnLine(0.5 + 0.04 * i, 0.20, 0.80) for i in range(10)]
    left, right = column_edges(lines, 0.8, 0.15, 5)
    assert left == pytest.approx(0.12) and right == pytest.approx(0.88)
    assert column_edges(lines[:3], 0.2, 0.15, 5) is None


# --- running heads -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, page_index, head",
    [
        ("938 SUPREME COURT REPORTS [1985] 2 S.C.R.", 1, True),
        ("SUP!u!l4B COURT REPORTS", 1, True),  # OCR, page number read apart
        ("STATE OF KERALA v. A. LAKSHMIKUTTY (SEN, J.] l43", 2, True),
        ("STATE BANK v. C.I.T. [TULZAPURKAR, J.]", 3, True),
        ("MIS. SETHI AUTO S. STATION & ANR. v. DELHI", 1, True),
        ("[1985] 2 S.C.R. 937", 0, True),
        ("STATE OF HARYANA v. RAJA RAM", 0, False),  # the title block on page 1
        ("of the High Court in Ram v. Shyam, which", 2, False),
    ],
)
def test_era_b_head_shapes(text, page_index, head, cfg, settings):
    printed = 937 + page_index if page_index != 2 else 143
    assert is_era_b_head(text, printed, page_index, cfg, settings.running_head_patterns) is head


def mkline(text: str, x0: float, y: float, block: int) -> Line:
    return Line(body_line(text.split(), x0, y, block, 0))


def test_head_rows_are_the_first_rows_only(cfg, settings):
    num = mkline("699", 340, 20, 1)
    title = mkline("NARINDRAJIT v. U.P. STATE (Grover, J.)", 90, 21, 2)
    body = mkline("the appellant filed a petition under Article 226", 55, 60, 10)
    # head-like text below a body row is never taken
    late = mkline("THE STATE OF X v. Y WAS CITED AT 699", 55, 76, 11)
    foot = mkline("699", 190, 600, 30)
    lines = [num, title, body, late, foot]
    ids = era_b_head_ids(lines, H, 699, 1, cfg, settings.running_head_patterns)
    assert ids == {id(num), id(title), id(foot)}


def test_a_page_without_a_head_keeps_its_first_rows(cfg, settings):
    body = [
        mkline("the appellant filed a petition under Article 226", 55, 40 + 16 * i, i)
        for i in range(3)
    ]
    assert era_b_head_ids(body, H, 699, 1, cfg, settings.running_head_patterns) == set()


def test_heads_on_a_page_placed_mid_sheet(cfg, settings):
    # 1970-1985 PDFs that put the scan in the middle of a letter-size sheet
    head = mkline(".J. K. PRASAD V. BIHAR. (Khanna, I.) 877", 240, 0.212 * 792, 1)
    body = mkline("the appellant filed a petition", 165, 0.25 * 792, 2)
    ids = era_b_head_ids([head, body], 792, 877, 2, cfg, settings.running_head_patterns)
    assert ids == {id(head)}


# --- paragraphs and text -------------------------------------------------------------


def rows_at(xs: list[float | tuple[float, float]]) -> list[tuple[int, Line]]:
    """Rows from x0 (a full line, to x1 = 350) or (x0, x1)."""
    out = []
    for i, x in enumerate(xs):
        x0, x1 = x if isinstance(x, tuple) else (x, 350.0)
        y = 80 + 14 * i
        words = [
            Word(x0, y, x0 + 20, y + 10, "word", i, 0),
            Word(x1 - 20, y, x1, y + 10, "end", i, 0),
        ]
        out.append((0, Line(words)))
    return out


def test_paragraph_starts(cfg):
    # first-line indents at rows 0 and 3; a block quote at rows 6-8 (its
    # first line indented further); the text after the quote at row 9
    xs = [70, 55, 55, 70, 55, 55, (100, 320), (85, 320), (85, 250), 55, 55]
    starts = paragraph_starts(rows_at(xs), [W], cfg)
    assert [i for i, s in enumerate(starts) if s] == [0, 3, 6, 9]


def test_consecutive_one_line_paragraphs(cfg):
    # "1. Leave granted." / "2. Parties to the litigation ..." both indented
    starts = paragraph_starts(rows_at([55, 70, 70, 55, 55]), [W], cfg)
    assert starts == [False, True, True, False, False]


def test_line_after_an_indented_author_line_is_not_a_paragraph(cfg):
    # counsel / "The Judgment of the Court was delivered by" / "GROVER, J.
    # These two appeals" (both indented) / "ment of the Allahabad High Court"
    starts = paragraph_starts(rows_at([55, (80, 250), 80, 55, 55]), [W], cfg)
    assert starts == [False, True, True, False, False]


def test_build_text_puts_a_blank_line_before_paragraphs():
    rows = [(0, ["One", "two."]), (0, ["Three"]), (1, ["four."])]
    text, pmap = build_text(rows, 10, [False, True, False])
    assert text == "One two.\n\nThree\nfour."
    assert pmap == [(0, 10), (10, 10), (16, 11)]
    assert build_text(rows, 10) == ("One two.\nThree\nfour.", [(0, 10), (9, 10), (15, 11)])


def era_b_pdf() -> bytes:
    """Three pages in the shape of a 1985 SCR scan."""
    doc = pymupdf.open()
    pages = [
        [
            (150, "ALPHA LTD."),
            (190, "v."),
            (140, "STATE OF BETA"),
            (100, "[D. A. DESAI AND R. B. MISRA, JJ.]"),
            (70, "Income Tax Act, 1961-Section 10-Exemption-Whether available."),
            (70, "HELD : (1) The assessee is entitled to the exemption. [939B]"),
            (55, "(2) The order of the High Court is set aside and the case is"),
            (55, "remanded to the Tribunal for fresh disposal. [940C]"),
            (70, "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 1980."),
            (55, "S. Rao and P. Iyer for the Appellant and others for them."),
            (70, "The Judgment of the Court was delivered by"),
            (70, "DESAI, J. This appeal by special leave raises a short point"),
            (55, "about the exemption claimed by the assessee under the Act and"),
            (55, "the High Court answered the question against the assessee here."),
        ],
        [
            (70, "The Tribunal had allowed the exemption on the facts found by it"),
            (55, "and the Commissioner then took the matter to the High Court in"),
            (55, "a reference under the Act which was answered against the party."),
            (55, "We have heard learned counsel for both sides on that question."),
            (70, "The only question is whether the income is exempt under the"),
            (55, "section as it then stood and we think that it plainly was so."),
            (55, "The appeal is allowed and the order of the High Court is set aside."),
        ],
        [
            (70, "There will be no order as to costs in the circumstances of the"),
            (55, "case before us since the respondent did not appear to contest it."),
            (55, "Nothing in this order shall affect the other assessment years."),
            (55, "The Tribunal shall dispose of the matter within three months."),
            (55, "Appeal allowed with the directions given above in this order."),
        ],
    ]
    heads = [
        None,
        "938 SUPREME COURT REPORTS [1985] 2 S.C.R.",
        "ALPHA LTD. v. BETA (Desai, J.) 939",
    ]
    for pno, body in enumerate(pages):
        page = doc.new_page(width=W, height=H)
        if heads[pno]:
            page.insert_text((90, 40), heads[pno], fontsize=9)
        else:
            page.insert_text((150, 40), "[1985] 2 S.C.R. 937", fontsize=9)
        letter_x = 372 if pno % 2 == 0 else 22
        for i, (x, line) in enumerate(body):
            y = 80 + 16 * i
            page.insert_text((x, y), line, fontsize=8)
            if i % 2 == 0:
                page.insert_text((letter_x, y), "ABCDEFGH"[i // 2], fontsize=8)
        page.insert_text((190, 620), str(937 + pno), fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


def test_parse_an_era_b_pdf(settings):
    doc = parse(era_b_pdf(), "1985_2_937_939", Era.B, settings)
    assert isinstance(doc, ParsedDoc) and doc.era is Era.B
    text = doc.clean_text
    assert not {"A", "B", "C", "D", "E", "F", "G", "H"} & set(text.split())
    assert "SUPREME COURT REPORTS" not in text and "Desai, J.) 939" not in text
    assert "938" not in text.split() and "939" not in text.split()
    assert "\n\nThe only question is whether" in text  # an indented first line
    assert "\nand the Commissioner then took" in text  # not a paragraph start
    assert doc.position_map[0] == (0, 937)
    assert doc.position_map[-1][1] == 939


# --- zones -----------------------------------------------------------------------------


def test_era_b_fixture_zones(settings):
    doc = load(ERA_B)
    z = zone(doc, settings)
    text = doc.clean_text
    assert z.missing == []  # no "references" zone in Era B
    assert text[: z.header_span[1]].endswith("[A. N. RAY, C.J. AND JASWANT SINGH, JJ.]")
    assert [hp.text[:20] for hp in z.editorial] == ["(I) The pr.incipal q", "(2) The fertilizer m"]
    assert text[z.judgment_span[0] :].startswith("The Judgment of the Court was delivered by")
    # the facts summary between the catchwords and HELD names the Act mid-sentence
    assert z.acts == ["Tamil Nadu General Sales Tax Act, 1959"]
    # the case details and counsel lines are in neither zone
    assert "for the Appellant" not in z.editorial[-1].text


ERA_B_TEXT = """ALPHA LTD.
v.
STATE OF BETA
[D. A. DESAI AND R. B. MISRA, JJ.]
Income Tax Act, 1961-Section 10-Exemption.
Allowing the appeal, the Court
HBLD : 1. The assessee is entitled to the exemption. [939B]
2. The order of the High Court is set aside. [940C]
CtVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 1980.
From the Judgment and Order dated 1.1.1979 of the Bombay High Court.
S. Rao for the Appellant.
{delivered}
{author}This appeal by special leave raises a short point.
The appeal is allowed.
"""


def b_doc(delivered: str, author: str = "DESAI, J. ") -> ParsedDoc:
    text = ERA_B_TEXT.format(delivered=delivered, author=author)
    return parsed_from_text("1985_2_937_939", Era.B, text)


@pytest.mark.parametrize(
    "delivered",
    [
        "The Judgment of the Court was delivered by",
        "1hc Judgment of the Cou:t was delivered by",
        "The following Judgments were deliver~d;",
        "JUDGMENT",
    ],
)
def test_era_b_judgment_starts_at_the_delivered_line(delivered, settings):
    doc = b_doc(delivered)
    z = zone(doc, settings)
    assert doc.clean_text[z.judgment_span[0] :].startswith(delivered)
    assert [hp.text for hp in z.editorial] == [
        "1. The assessee is entitled to the exemption. [939B]",
        "2. The order of the High Court is set aside. [940C]",
    ]
    assert z.acts == ["Income Tax Act, 1961"]


def test_era_b_judgment_starts_at_the_author_without_a_delivered_line(settings):
    doc = b_doc("", "MADON J. ")
    z = zone(doc, settings)
    assert doc.clean_text[z.judgment_span[0] :].startswith("MADON J. This appeal")


def test_era_b_without_a_jurisdiction_line_cuts_the_editorial_at_the_details(settings):
    text = ERA_B_TEXT.replace("CtVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 1980.\n", "")
    doc = parsed_from_text("1985_2_937_939", Era.B, text.format(delivered="", author="DESAI, J. "))
    z = zone(doc, settings)
    assert z.editorial[-1].text.endswith("[940C]")
    assert doc.clean_text[z.judgment_span[0] :].startswith("DESAI, J. This appeal")


def test_era_b_bracketed_held_points(settings):
    text = ERA_B_TEXT.replace("1. The assessee", "(i) The assessee").replace(
        "2. The order", "(ii) The order"
    )
    doc = parsed_from_text("1985_2_937_939", Era.B, text.format(delivered="JUDGMENT", author=""))
    z = zone(doc, settings)
    assert [hp.text[:4] for hp in z.editorial] == ["(i) ", "(ii)"]


@pytest.mark.parametrize(
    "name, act",
    [
        (
            "The appellant is a dealer under the Tamil Nadu Sales Tax Act, 1959",
            "Tamil Nadu Sales Tax Act, 1959",
        ),
        ("Code of Criminal Procedure, 1973", "Code of Criminal Procedure, 1973"),
        ("Land Acquisition (Companies) Rules, 1963", "Land Acquisition (Companies) Rules, 1963"),
        ("I I Customs Act, 1962", "Customs Act, 1962"),
        ("dealer under the Act, 1959", ""),
    ],
)
def test_title_run(name, act):
    assert title_run(name) == act


# --- opinions ----------------------------------------------------------------------------


def test_era_b_fixture_opinion(settings):
    doc = load(ERA_B)
    z = zone(doc, settings)
    (op,) = split_opinions(doc, z, ["A.N. RAY"], settings)
    assert op.author == "JASWANT SINGH, J." and op.type is OpinionType.MAJORITY
    assert doc.clean_text[op.char_start :].startswith("JASWANT SINGH, J.-These appeals")


def test_era_b_two_opinions_with_ocr_names(settings):
    text = (
        ERA_B_TEXT.format(
            delivered="The following Judgments were delivered:", author="BHAGWATl, J.-"
        )
        + "KRISHNA IYER, J.-I regret I am unable to agree with my learned brother.\n"
    )
    text = text.replace(
        "[D. A. DESAI AND R. B. MISRA, JJ.]", "[P. N. BHAGWATI AND V. R. KRISHNA IYER, JJ.]"
    )
    doc = parsed_from_text("1985_2_937_939", Era.B, text)
    z = zone(doc, settings)
    ops = split_opinions(doc, z, [], settings)
    assert [o.author for o in ops] == ["BHAGWATl, J.", "KRISHNA IYER, J."]
    assert ops[1].type is OpinionType.DISSENTING


@pytest.mark.parametrize(
    "name, known, match",
    [
        ("BHAGWATI", "P. N. BHAGWAT!", True),
        ("NANAVATl", "G.T. NANA VAT!", True),
        ("FAzAL ALI", "S. MURTAZA FAZAL ALI", True),
        ("CHANDRACHUD", "Y. V. CHANDRACO:HUD", True),
        ("KHANNA", "P. N. BHAGWATI", False),
        ("SEN", "A. P. SEN", False),  # short names need names_match
    ],
)
def test_ocr_name_match(name, known, match):
    assert ocr_name_match(name, known) is match


def test_c_and_d_opinion_starts_unchanged(settings):
    # Era C keeps its stricter start: an author followed by a sentence is not one
    text = ERA_B_TEXT.format(
        delivered="The Judgment of the Court was delivered by", author="DESAI, J. "
    )
    text = text.replace("HBLD", "HELD").replace("CtVIL", "CIVIL")
    doc = parsed_from_text("2015_2_937_939", Era.C, text)
    z = zone(doc, settings)
    (op,) = split_opinions(doc, z, [], settings)
    assert op.author is None


# --- paragraphs and chunks ---------------------------------------------------------------


@pytest.mark.parametrize(
    "line, label",
    [
        ("4, The High Court has adopted a curious process.", "4"),
        ("8: Shri G.L. Sanghi, learned senior counsel, argued.", "8"),
        ("11.Aggrieved by the same, the appellant filed an appeal.", "11"),
        ("O 12. Mr. Nafade, learned counsel, appeared.", "12"),
        ("SRINIVASAN, J.- 2. The pivotal question is this.", "2"),
    ],
)
def test_era_b_ocr_paragraph_numbers(line, label):
    text = f"1. First paragraph here.\n{line}\n{int(label) + 1}. Next paragraph here.\n"
    text = text.replace("1. First", f"{int(label) - 1}. First")
    paras = find_paragraphs(text, 0, len(text), era_b=True)
    assert [p.label for p in paras] == [str(int(label) - 1), label, str(int(label) + 1)]
    # Era C reads the OCR form as text, not as a number
    assert label not in [p.label for p in find_paragraphs(text, 0, len(text))]


def test_era_b_needs_three_numbered_paragraphs():
    text = "The appeal was heard.\n752.\nIt is dismissed.\n"
    assert find_paragraphs(text, 0, len(text)) != []
    assert find_paragraphs(text, 0, len(text), min_paragraphs=3, era_b=True) == []


def test_paragraph_rules(settings):
    assert paragraph_rules(load(ERA_B), settings) == {"min_paragraphs": 3, "era_b": True}
    c = parsed_from_text("2015_1_1_2", Era.C, "x")
    assert paragraph_rules(c, settings) == {}


def test_era_b_fixture_chunks_follow_blank_line_paragraphs(settings):
    doc = load(ERA_B)
    z = zone(doc, settings)
    ops = split_opinions(doc, z, ["A.N. RAY"], settings)
    chunks = chunk(
        {"title": "Shaw Wallace v. State of Tamil Nadu", "decision_year": 1976},
        doc,
        z,
        ops,
        settings,
    )
    text = doc.clean_text
    ed = [c for c in chunks if c.zone is Zone.EDITORIAL]
    jc = [c for c in chunks if c.zone is Zone.JUDGMENT]
    assert len(ed) == 2 and jc
    for c in chunks:
        assert text[c.char_start : c.char_end] == c.text
    for c in jc:
        assert "unnumbered" in c.flags and c.opinion_author == "JASWANT SINGH, J."
        # every judgment chunk starts at the opinion or right after a blank line
        assert c.char_start == ops[0].char_start or text[c.char_start - 2 : c.char_start] == "\n\n"
    m = doc_metrics(doc, z, ops, chunks, settings)
    assert m["paragraphs_total"] == 0 and m["in_size"] == m["total"]


# --- build ---------------------------------------------------------------------------------


def row(path: str, year: int, tar: str) -> dict:
    return {
        "path": path,
        "title": "ALPHA LTD. versus STATE OF BETA",
        "petitioner": "ALPHA LTD.",
        "respondent": "STATE OF BETA",
        "judges": ["D.A. DESAI", "R.B. MISRA"],
        "author_judge": "D.A. DESAI",
        "scr_citation": f"[{year}] 2 S.C.R. 937",
        "neutral_citation": None,
        "cnr": "",
        "disposal": "Appeal(s) allowed",
        "outcome": "allowed",
        "decision_date": date(year, 2, 20),
        "decision_year": year,
        "folder_year": year,
        "has_pdf": True,
        "tar_path": tar,
        "offset_data": 0,
        "size": 1,
    }


def test_doc_processor_parses_era_b_and_names_era_a_skips(settings):
    pdfs = {
        "b.tar": era_b_pdf(),
        "a.tar": plain_pdf([["BHARAT BANK LTD.", "v.", "EMPLOYEES", "Held, that the award"]]),
    }
    proc = b.DocProcessor(settings, read_pdf=lambda tar, _o, _s: pdfs[tar])
    res = proc(row("1985_2_937_939", 1985, "b.tar"))
    assert res.error is None and res.era == "b"
    assert {c.zone for c in res.chunks} == {Zone.EDITORIAL, Zone.JUDGMENT}
    old = proc(row("1955_1_1_2", 1955, "a.tar"))
    assert old.era == "out_of_scope" and not old.chunks
    assert old.skipped == [("1955_1_1_2", "parse", "era_out_of_scope", ERA_A_REASON)]
