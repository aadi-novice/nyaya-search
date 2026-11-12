"""pdf_layout.parse: furniture removal by position, hyphens, position map.

Synthetic PDFs cover each rule; four real PDFs (2015, 2018, 2023, 2025) are
checked when docs/data is present.
"""

import re

import pymupdf
import pytest

from corpus_helpers import (
    LEFT_LETTER_X,
    PAGE_H,
    clean_settings,
    era_c_page,
    era_c_pdf,
    tar_for,
)
from nyaya.config import ConfigError
from nyaya.contracts import Era, ParsedDoc
from nyaya.parse.pdf_layout import rejoin_hyphens, repeated_keys, start_page

BASE = clean_settings()
# Bands measured by `calibrate` on the 2010-2023 sample (build/bands.json, 2026-11-12).
MEASURED = {
    "era_c": {"left_x": [0.0672, 0.2718], "right_x": [0.742, 0.9417]},
    "top_y": [0.0536, 0.1666],
    "bottom_y": [0.8365, 0.9519],
}
SETTINGS = BASE.model_copy(update={"bands": BASE.bands.model_validate(MEASURED)})

MARGIN_RESIDUE = re.compile(r"(?<=[a-z,]) [A-H] (?=[a-z])")
WORDS = [
    "appeal",
    "order",
    "court",
    "section",
    "proviso",
    "statute",
    "tribunal",
    "evidence",
    "witness",
    "decree",
]


def parse(data, path="2020_11_786_799", era=Era.C, settings=SETTINGS):
    from nyaya.parse.pdf_layout import parse as _parse

    return _parse(data, path, era, settings)


def varied(i: int) -> list[str]:
    w = WORDS[i % 10 :] + WORDS[: i % 10]
    return [" ".join(w[:7]), " ".join(w[3:10]), " ".join(w[1:6])]


def five_pages(first_lines=None) -> bytes:
    pages = [varied(i) for i in range(5)]
    for i, line in (first_lines or {}).items():
        pages[i] = [line, *pages[i]]
    return era_c_pdf(pages, start_page=786)


# --- path ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, page",
    [("2025_1_230_238", 230), ("S_2005_5_75_82", 75), ("2014_13_1265_1265", 1265)],
)
def test_start_page(path, page):
    assert start_page(path) == page


def test_start_page_rejects_other_ids():
    with pytest.raises(ValueError):
        start_page("2025INSC20")


# --- margin letters ----------------------------------------------------------------


def test_margin_letters_are_removed_on_both_sides():
    doc = parse(five_pages())
    assert isinstance(doc, ParsedDoc)
    lines = doc.clean_text.splitlines()
    assert not any(ln.strip() in set("ABCDEFGH") for ln in lines)
    assert not MARGIN_RESIDUE.search(doc.clean_text)
    assert " H" not in doc.clean_text and "A " not in doc.clean_text


def test_letters_inside_text_are_kept():
    doc = parse(five_pages({0: "A proviso to s. 47 applies, says Schedule H here"}))
    assert "A proviso to s. 47 applies, says Schedule H here" in doc.clean_text


def test_letter_column_outside_the_band_is_still_removed():
    shifted = MEASURED | {"era_c": {"left_x": [0.30, 0.31], "right_x": [0.95, 0.96]}}
    s = BASE.model_copy(update={"bands": BASE.bands.model_validate(shifted)})
    doc = parse(five_pages(), settings=s)
    assert not any(ln.strip() in set("ABCDEFGH") for ln in doc.clean_text.splitlines())


def test_ocr_variant_letters_in_the_band_are_removed():
    d = pymupdf.open()
    era_c_page(d, 0, varied(0), printed_page=786, letters=False)
    page = d[0]
    for i, ch in enumerate("A8cDEFGH"):
        page.insert_text((LEFT_LETTER_X, 158 + i * 72.7), ch, fontsize=10)
    doc = parse(d.tobytes())
    assert "8" not in doc.clean_text.split() and "c" not in doc.clean_text.split()


def test_era_c_without_bands_is_a_config_error():
    with pytest.raises(ConfigError, match=r"bands\.era_c"):
        parse(five_pages(), settings=BASE)


def test_era_d_keeps_lone_letters_and_needs_no_bands():
    doc = parse(five_pages(), era=Era.D, settings=BASE)
    assert {"A", "H"} <= set(doc.clean_text.split())


def test_out_of_scope_is_refused():
    with pytest.raises(ValueError, match="out-of-scope"):
        parse(five_pages(), era=Era.OUT_OF_SCOPE)


# --- running heads and page numbers ------------------------------------------------


def test_running_heads_titles_and_page_numbers_are_removed():
    doc = parse(five_pages())
    text = doc.clean_text
    assert "SUPREME COURT REPORTS" not in text
    assert "S.C.R." not in text
    assert "ALPHA LTD." not in text
    for n in range(786, 791):
        assert str(n) not in text.split()


def test_wrapped_title_second_line_is_removed():
    d = pymupdf.open()
    for pno in range(5):
        era_c_page(d, pno, varied(pno), printed_page=786 + pno)
        if pno % 2:
            d[pno].insert_text((220, 128), f"OF GAMMA [JUDGE {'XYZ'[pno % 3]}, J.]", fontsize=10)
    doc = parse(d.tobytes())
    assert "GAMMA" not in doc.clean_text


def test_body_paragraph_under_the_head_is_kept():
    d = pymupdf.open()
    for pno in range(5):
        era_c_page(d, pno, varied(pno), printed_page=786 + pno)
    d[1].insert_text((149, 128), "3. Aggrieved by the conviction", fontsize=10)
    doc = parse(d.tobytes())
    assert "3. Aggrieved by the conviction" in doc.clean_text


def test_numbers_in_body_and_footnote_markers_are_kept():
    d = pymupdf.open()
    for pno in range(5):
        era_c_page(d, pno, varied(pno), printed_page=786 + pno)
    d[2].insert_text((163, 0.9 * PAGE_H), "3", fontsize=8)  # a footnote marker
    d[2].insert_text((175, 0.9 * PAGE_H), "(2024) 7 SCC 140", fontsize=8)
    doc = parse(d.tobytes())
    assert "3 (2024) 7 SCC 140" in doc.clean_text


def test_first_page_footer_page_number_is_removed():
    d = pymupdf.open()
    for pno in range(5):
        era_c_page(d, pno, varied(pno), printed_page=786 + pno)
    d[0].insert_text((300, 0.85 * PAGE_H), "786", fontsize=10)
    assert "786" not in parse(d.tobytes()).clean_text


def test_repeated_keys_tolerate_ocr_noise():
    k, variant = "unionofindiavdileepkumarsingh", "unionofindiavdileepkumarsinqh"
    pages = [{k}, {k}, {variant}, {"somethingelse"}]
    assert repeated_keys(pages, 3) == {k, variant}
    assert repeated_keys(pages, 4) == set()
    assert repeated_keys([{variant}, {"somethingelse"}], 1) == set()  # needs an exact repeat


# --- lines, hyphens, position map -------------------------------------------------


def test_cells_on_one_baseline_join_into_one_line():
    d = pymupdf.open()
    for pno in range(3):
        era_c_page(d, pno, varied(pno), printed_page=786 + pno)
    page = d[1]
    page.insert_text((163, 400), "[1990] 3 Suppl. SCR 259", fontsize=10)
    page.insert_text((321, 400), "relied on", fontsize=10)
    page.insert_text((400, 400), "Para 41", fontsize=10)
    assert "[1990] 3 Suppl. SCR 259 relied on Para 41" in parse(d.tobytes()).clean_text


def test_rejoin_hyphens():
    rows = [
        (0, ["the", "consi-"]),
        (0, ["deration", "of", "the", "Respondent-complainant", "case"]),
        (0, ["the", "3rd", "Respondent-"]),
        (1, ["complainant,", "said", "Act,", "1995", "-"]),
        (1, ["Proviso", "to", "Anglo-"]),
        (1, ["Indian", "law"]),
        (1, ["10-"]),
        (1, ["only"]),
    ]
    out = [" ".join(t) for _, t in rejoin_hyphens(rows)]
    assert out == [
        "the consideration",
        "of the Respondent-complainant case",
        "the 3rd Respondent-complainant,",
        "said Act, 1995 -",
        "Proviso to Anglo-",
        "Indian law",
        "10-",
        "only",
    ]


def test_rejoin_drops_emptied_lines():
    out = rejoin_hyphens([(0, ["inter-"]), (0, ["alia"]), (0, ["next"])])
    assert out == [(0, ["interalia"]), (0, ["next"])]


def test_hyphen_rejoined_across_pages_in_a_pdf():
    pages = [varied(i) for i in range(5)]
    pages[0] = [*pages[0], "the matter was consi-"]
    pages[1] = ["deration of " + pages[1][0], *pages[1][1:]]
    doc = parse(era_c_pdf(pages))
    assert "the matter was consideration of" in doc.clean_text.replace("\n", " ")


def test_position_map_and_pages():
    doc = parse(five_pages())
    assert doc.pages == 5
    assert doc.clean_text_version == SETTINGS.versions.clean_text
    starts = [0] + [i + 1 for i, ch in enumerate(doc.clean_text) if ch == "\n"]
    assert [o for o, _ in doc.position_map] == starts
    pages = [p for _, p in doc.position_map]
    assert pages == sorted(pages)
    assert pages[0] == 786 and pages[-1] == 790
    assert set(pages) == set(range(786, 791))


# --- real PDFs -------------------------------------------------------------------

REAL = [
    (2015, "2015_2_882_902", {"2.1", "10", "11", "20", "30"}),
    (2018, "2018_14_870_892", {"1", "3.1", "8.9", "12.5", "16"}),
    (2023, "2023_2_587_610", {"1", "15", "26.2", "37"}),
    (2025, "2025_1_105_116", {"10", "11", "12", "13"}),
]


def real_pdf(year: int, path: str) -> bytes:
    from nyaya.corpus.tarstore import read_pdf, scan_offsets

    tar = tar_for(year)
    off, size = scan_offsets(tar)[f"{path}_EN.pdf"]
    return read_pdf(tar, off, size)


@pytest.mark.parametrize("year, path, labels", REAL, ids=[r[1] for r in REAL])
def test_real_pdf(year, path, labels):
    if not tar_for(year).exists():
        pytest.skip("docs/data is absent")
    from nyaya.parse.eras import classify_pdf

    data = real_pdf(year, path)
    era = classify_pdf(data, year, SETTINGS.era_markers)
    assert era is (Era.D if year >= 2024 else Era.C)
    doc = parse(data, path, era)
    assert not MARGIN_RESIDUE.search(doc.clean_text)
    assert "SUPREME COURT REPORTS" not in doc.clean_text
    found = set(re.findall(r"(?m)^(\d{1,2}(?:\.\d{1,2})?)\.?\s", doc.clean_text))
    assert labels <= found
    assert doc.pages == int(path.split("_")[-1]) - start_page(path) + 1
    assert doc.position_map[0][1] == start_page(path)
