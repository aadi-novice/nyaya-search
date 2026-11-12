import pytest

from nyaya.zone.zoner import find_para_links, para_refs, parse_para_list


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[Para 9]", ["9"]),
        ("[Paras 7, 8]", ["7", "8"]),
        ("[Paras 17 and 18]", ["17", "18"]),
        ("[Para 16 and 17]", ["16", "17"]),
        ("[Para 11.1 & 12]", ["11.1", "12"]),
        ("[Para 12.1]", ["12.1"]),
        ("[Paras 10-12]", ["10", "11", "12"]),
        ("[Paras 10–12]", ["10", "11", "12"]),
        ("[Paras 34-36]", ["34", "35", "36"]),
        ("[Paras 9, 10, 12.1]", ["9", "10", "12.1"]),
        ("[Paras 12.1-12.3]", ["12.1", "12.3"]),  # sub-paragraph range: the two ends
        ("[Para 13) [727-E]", ["13"]),  # OCR ")" for "]"
        ("[Para C 10)", ["10"]),  # stray margin letter
        ("[Paras 19 and 20) [366-C-E]", ["19", "20"]),
        ("[ Paras 22, 23 ]", ["22", "23"]),
        ("[Paras 29, 31, 32, 34-36]", ["29", "31", "32", "34", "35", "36"]),
        ("[Para 9.]", ["9"]),
    ],
)
def test_para_refs(text, expected):
    assert para_refs(text) == expected


def test_para_refs_wrapped_across_lines_and_several_links():
    text = "the key answer is wrong. [Para\n12][34-C-D] and later [Paras 14,\n15][34-H; 35-A]"
    assert para_refs(text) == ["12", "14", "15"]


def test_para_refs_deduplicates_in_order():
    assert para_refs("[Para 9] ... [Paras 9, 10]") == ["9", "10"]


def test_para_refs_none():
    assert para_refs("no links here [34-C-D]") == []
    assert para_refs("") == []


def test_find_para_links_spans():
    text = "Held: x. [Para 13) [727-E]"
    links = find_para_links(text)
    assert len(links) == 1
    (start, end), labels = links[0]
    assert text[start:end] == "[Para 13)"
    assert labels == ["13"]


@pytest.mark.parametrize(
    ("inner", "expected"),
    [
        ("12-10", ["12", "10"]),  # reversed range kept as its two ends
        ("1-500", ["1", "500"]),  # implausibly long range kept as its two ends
        ("7,, 8", ["7", "8"]),
        ("A 7 B", ["7"]),
    ],
)
def test_parse_para_list_edge_cases(inner, expected):
    assert parse_para_list(inner) == expected


def test_ocr_forms_from_scanned_years():
    assert para_refs("set aside altogether. [Paras 11 to 15) [259-H;") == [
        "11",
        "12",
        "13",
        "14",
        "15",
    ]
    assert para_refs("constituted by the Union government. (Paras 5, 6] (404-D-GJ") == ["5", "6"]
    assert para_refs("(para 12 of the impugned order)") == []
