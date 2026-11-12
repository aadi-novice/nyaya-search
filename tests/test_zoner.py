import pytest

from judgment_fixtures import ERA_C, ERA_C_TWO_OPINIONS, ERA_D, load, parsed_from_text
from nyaya.config import load_settings
from nyaya.contracts import Era, Zone
from nyaya.zone.zoner import zone


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def span_text(doc, span):
    return doc.clean_text[span[0] : span[1]]


def test_era_c_zones(settings):
    doc = load(ERA_C)
    z = zone(doc, settings)
    assert z.missing == []
    assert (
        span_text(doc, z.header_span).rstrip().endswith("[UDAY UMESH LALIT AND DEEPAK GUPTA, JJ.]")
    )
    assert z.issue_span is None
    assert span_text(doc, z.judgment_span).startswith("CIVIL APPELLATE JURISDICTION")
    assert [hp.para_refs for hp in z.editorial] == [["12"], ["14"], ["15"]]
    assert z.editorial[0].text.startswith("1.1 The onus is on the candidate")
    assert z.editorial[0].text.endswith("[Para 12][34-C-D]")
    assert z.editorial[1].text.startswith("1.2 In the instant case")
    # the last point stops at its pinpoint, before the list of cases cited
    assert z.editorial[2].text.endswith("[Para 15][35-B-C]")
    for hp in z.editorial:
        assert doc.clean_text[hp.char_start : hp.char_end] == hp.text
    assert z.acts == []  # catchwords name no Act ("Education/Educational Institutions")
    assert z.keywords == []


def test_era_c_two_held_blocks_and_acts(settings):
    doc = load(ERA_C_TWO_OPINIONS)
    z = zone(doc, settings)
    assert z.acts == ["Penal Code, 1860", "Evidence Act, 1872"]
    assert [hp.para_refs for hp in z.editorial] == [["16", "17"], ["9"]]
    assert z.editorial[1].text.startswith("1. In the facts and circumstances")
    assert z.editorial[1].text.endswith("[Para 9)(67-G-H; 68-\nA-B]")
    assert span_text(doc, z.judgment_span).startswith("CRIMINAL APPELLATE JURISDICTION")
    # the "In the Judgment of Prafulla C. Pant, J." line is in the references, not the judgment
    assert "In the Judgment of" not in span_text(doc, z.judgment_span)
    assert z.missing == []


def test_era_d_zones(settings):
    doc = load(ERA_D)
    z = zone(doc, settings)
    assert z.missing == []
    assert span_text(doc, z.header_span).rstrip().endswith("[Abhay S Oka* and Ujjal Bhuyan, JJ.]")
    issue = span_text(doc, z.issue_span)
    assert issue.startswith("Issue arose as regards the correctness")
    assert issue.endswith("first respondent.")
    assert [hp.para_refs for hp in z.editorial] == [
        ["29", "31", "32", "34", "35", "36"],
        ["20", "37"],
        ["22", "23"],
    ]
    assert all(hp.text.startswith("Held:") for hp in z.editorial)
    assert z.editorial[2].text.endswith("[Paras 22, 23]")
    assert z.acts == [
        "Maharashtra Ownership Flats (Regulation of the Promotion of Construction, Sale, "
        "Management and Transfer Act) 1963",
        "Maharashtra Ownership Flats (Regulation of the Promotion of Construction, Sale, "
        "Management and Transfer) Rules, 1964",
        "Registration Act, 1908",
    ]
    assert z.keywords[0] == "Hindu Undivided Family"
    assert z.keywords.count("Deed of partnership") == 1
    assert "Writ jurisdiction" in z.keywords
    assert span_text(doc, z.judgment_span).startswith("Judgment\nAbhay S. Oka, J.")


def test_era_d_missing_everything_falls_back(settings):
    text = "Some PDF text\nwith no labelled sections at all.\n1. A paragraph."
    doc = parsed_from_text("2025_1_1_2", Era.D, text)
    z = zone(doc, settings)
    assert set(z.missing) == {Zone.ISSUE, Zone.EDITORIAL, Zone.REFERENCES, Zone.JUDGMENT}
    assert z.judgment_span == (0, len(text))
    assert z.editorial == [] and z.acts == [] and z.keywords == []


def test_era_d_held_without_link_runs_to_next_held(settings):
    text = (
        "Title\n[A, J.]\nHeadnotes\nPenal Code – s.302:\nHeld: first point without a link.\n"
        "Evidence Act – s.6:\nHeld: second point. [Para 4]\nCase Law Cited\nnone\n"
        "Judgment / Order of the Supreme Court\n1. Text.\n"
    )
    z = zone(parsed_from_text("2025_1_1_2", Era.D, text), settings)
    assert len(z.editorial) == 2
    assert z.editorial[0].para_refs == []
    assert "Evidence Act" in z.editorial[0].text
    assert z.editorial[1].text == "Held: second point. [Para 4]"
    assert z.issue_span is None
    assert z.missing == [Zone.ISSUE]


def test_era_c_missing_bench_and_jurisdiction(settings):
    text = "Some title –\nPenal Code, 1860 – s.302 – Murder.\nHELD: 1. A point. [Para 3]\n"
    doc = parsed_from_text("2015_1_1_2", Era.C, text)
    z = zone(doc, settings)
    assert Zone.HEADER in z.missing and Zone.JUDGMENT in z.missing
    assert Zone.REFERENCES in z.missing
    assert z.judgment_span == (0, len(text))
    assert z.acts == ["Penal Code, 1860"]
    assert [hp.para_refs for hp in z.editorial] == [["3"]]


def test_era_c_held_without_numbers_is_one_point(settings):
    text = (
        "A v. B\n[X, J.]\nService Law: Promotion.\n"
        "HELD: Per Surinder Singh Nijjar, J. The order is bad. Per Y, J. Agreed. [Para 3]\n"
        "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 2011.\nX, J. 1. Text.\n"
    )
    z = zone(parsed_from_text("2011_1_1_2", Era.C, text), settings)
    assert len(z.editorial) == 1
    assert z.editorial[0].text.startswith("Per Surinder Singh Nijjar")
    assert z.acts == []


def test_era_c_point_numbers_must_follow_in_order(settings):
    text = (
        "A v. B\n[X, J.]\nPenal Code, 1860 – s.302.\n"
        "HELD: 1. PWs 1 and 2 saw it under Article 21 The court. [Para 4]\n"
        "1.1 Sub point here. [Para 5]\n2. Second point. [Para 6]\n"
        "7 Wrong number here.\n"
        "CRIMINAL APPELLATE JURISDICTION: Criminal Appeal No. 1 of 2011.\n"
    )
    z = zone(parsed_from_text("2011_1_1_2", Era.C, text), settings)
    assert [hp.para_refs for hp in z.editorial] == [["4"], ["5"], ["6"]]
    assert z.editorial[0].text.startswith("1. PWs 1 and 2 saw it under Article 21 The")


def test_era_c_editorial_missing_when_no_text(settings):
    text = "A v. B\n[X, J.]\nCIVIL APPELLATE JURISDICTION: Civil Appeal No. 1.\nX, J. 1. T.\n"
    z = zone(parsed_from_text("2011_1_1_2", Era.C, text), settings)
    assert Zone.EDITORIAL in z.missing
    assert z.editorial == []


def test_zone_default_settings():
    z = zone(load(ERA_C))
    assert len(z.editorial) == 3
