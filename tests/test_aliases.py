import pandas as pd
import pytest

from nyaya.link.aliases import (
    alias_collisions,
    build_alias_rows,
    canonical_citation,
    find_citations,
    path_citation,
)

from conftest import FIXTURES


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[2025] 1 S.C.R. 230", "scr:2025:1:230"),
        ("[2025] 1 SCR 230", "scr:2025:1:230"),
        ("[2025]1 S. C. R. 230", "scr:2025:1:230"),
        ("2025 (1) SCR 230", "scr:2025:1:230"),
        ("2025(1)SCR 230", "scr:2025:1:230"),
        ("[2006] SUPP. 6 S.C.R. 72", "scr:2006:s6:72"),
        ("[2006] Supp 6 SCR 72", "scr:2006:s6:72"),
        ("[1996] SUPP. 6 S.C.R. 631", "scr:1996:s6:631"),
        ("2025 INSC 31", "insc:2025:31"),
        ("2025INSC31", "insc:2025:31"),
        ("2025 insc 031", "insc:2025:31"),
        ("(2009) 8 SCC 751", "scc:2009:8:751"),
        ("(2009)8 SCC751", "scc:2009:8:751"),
        ("AIR 1999 SC 3750", "air:1999:3750"),
        ("air 1999 sc 3750", "air:1999:3750"),
        ("ESCR010004242025", "cnr:escr010004242025"),
        ("escr010004242025", "cnr:escr010004242025"),
        ("  2025 INSC 31  ", "insc:2025:31"),
    ],
)
def test_canonical_citation(text, expected):
    assert canonical_citation(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "anticipatory bail section 438",
        "2025 INSC",
        "INSC 31",
        "[2025] S.C.R. 230",
        "ESCR01000424202",  # 11 digits
        "ESCR0100042420251",  # 13 digits
        "12025 INSC 31",
        "2025 INSC 31 and 2025 INSC 32",  # two citations are not one
        "see 2025 INSC 31",  # extra words: not a citation on its own
    ],
)
def test_canonical_citation_rejects(text):
    assert canonical_citation(text) is None


def test_canonical_citation_none_input():
    assert canonical_citation(None) is None  # type: ignore[arg-type]


def test_find_citations_mixed_text():
    text = (
        "Abdul Kuddus v. Union of India [2019] 8 SCR 669 : (2019) 6 SCC 604; see also "
        "[2025] 1 S.C.R. 230 : 2025 INSC 31, AIR 1999 SC 3750 and CNR ESCR010004242025."
    )
    found = find_citations(text)
    assert [(k, c) for _, k, c in found] == [
        ("scr", "scr:2019:8:669"),
        ("scc", "scc:2019:6:604"),
        ("scr", "scr:2025:1:230"),
        ("neutral", "insc:2025:31"),
        ("air", "air:1999:3750"),
        ("cnr", "cnr:escr010004242025"),
    ]
    for (start, end), _, _ in found:
        assert canonical_citation(text[start:end]) is not None


def test_find_citations_second_scr_form_and_supp():
    text = "2011 (8) SCR 774 relied on; [2006] SUPP. 6 S.C.R. 72 referred to"
    assert [c for _, _, c in find_citations(text)] == ["scr:2011:8:774", "scr:2006:s6:72"]


def test_find_citations_none():
    assert find_citations("anticipatory bail under section 438 of the CrPC") == []
    assert find_citations("") == []


def test_find_citations_no_overlap():
    # "(2018) 2 SCC 357" must not also yield an SCR or neutral match
    found = find_citations("(2018) 2 SCC 357")
    assert [k for _, k, _ in found] == ["scc"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("2025_1_230_238", "scr:2025:1:230"),
        ("S_2005_5_75_82", "scr:2005:s5:75"),
        ("not_a_path", None),
        ("2025_1_230", None),
    ],
)
def test_path_citation(path, expected):
    assert path_citation(path) == expected


def _manifest():
    return pd.DataFrame(
        [
            {
                "path": "2025_1_230_238",
                "scr_citation": "[2025] 1 S.C.R. 230",
                "neutral_citation": "2025INSC31",
                "cnr": "ESCR010004242025",
            },
            {
                "path": "S_1996_6_631_656",
                "scr_citation": "[1996] SUPP. 6 S.C.R. 631",
                "neutral_citation": None,
                "cnr": "ESCR010015471996",
            },
            {
                "path": "2018_10_937_952",
                "scr_citation": "[2018] 10 S.C.R. 937",
                "neutral_citation": "",
                "cnr": float("nan"),
            },
        ]
    )


def test_build_alias_rows_only_loaded_paths():
    rows = build_alias_rows(_manifest(), {"2025_1_230_238", "S_1996_6_631_656"})
    assert {r["path"] for r in rows} == {"2025_1_230_238", "S_1996_6_631_656"}
    got = {(r["alias_norm"], r["kind"], r["path"]) for r in rows}
    assert got == {
        ("scr:2025:1:230", "scr", "2025_1_230_238"),
        ("insc:2025:31", "neutral", "2025_1_230_238"),
        ("cnr:escr010004242025", "cnr", "2025_1_230_238"),
        ("scr:1996:s6:631", "scr", "S_1996_6_631_656"),
        ("cnr:escr010015471996", "cnr", "S_1996_6_631_656"),
    }
    assert len(rows) == len(got)  # no duplicate rows


def test_build_alias_rows_empty_loaded_set():
    assert build_alias_rows(_manifest(), set()) == []


def test_build_alias_rows_missing_values_and_raw_columns():
    raw = [
        {
            "path": "2018_10_937_952",
            "citation": "[2018] 10 S.C.R. 937",
            "nc_display": "2018INSC1",
            "cnr": None,
        },
        {"path": "2019_1_1_5", "citation": "garbage", "nc_display": None, "cnr": "bad"},
    ]
    rows = build_alias_rows(raw, ["2018_10_937_952", "2019_1_1_5"])
    got = {(r["alias_norm"], r["kind"], r["path"]) for r in rows}
    assert got == {
        ("scr:2018:10:937", "scr", "2018_10_937_952"),
        ("insc:2018:1", "neutral", "2018_10_937_952"),
        # an unparseable SCR citation falls back to the SCR citation in the path
        ("scr:2019:1:1", "scr", "2019_1_1_5"),
    }


def test_build_alias_rows_citation_and_path_disagree_both_kept():
    raw = [{"path": "2010_11_871_896", "scr_citation": "[2010] 11 S.C.R. 870"}]
    got = {r["alias_norm"] for r in build_alias_rows(raw, {"2010_11_871_896"})}
    assert got == {"scr:2010:11:870", "scr:2010:11:871"}


def test_build_alias_rows_drops_colliding_aliases():
    raw = [
        {"path": "2020_1_1_10", "neutral_citation": "2020INSC5"},
        {"path": "2020_1_11_20", "neutral_citation": "2020 INSC 5"},
    ]
    rows = build_alias_rows(raw, {"2020_1_1_10", "2020_1_11_20"})
    assert "insc:2020:5" not in {r["alias_norm"] for r in rows}
    assert alias_collisions(raw, {"2020_1_1_10", "2020_1_11_20"}) == {
        ("insc:2020:5", "neutral"): ["2020_1_11_20", "2020_1_1_10"]
    }


def test_build_alias_rows_on_fixture_manifest():
    df = pd.read_parquet(FIXTURES / "manifest_small.parquet").drop_duplicates("path")
    df = df.rename(columns={"citation": "scr_citation", "nc_display": "neutral_citation"})
    loaded = set(df["path"])
    rows = build_alias_rows(df, loaded)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"scr", "neutral", "cnr"}
    by_path = {}
    for r in rows:
        by_path.setdefault(r["path"], set()).add(r["alias_norm"])
    assert by_path["S_1959_1_1_10"] >= {"scr:1959:s1:1", "insc:1958:72", "cnr:escr010000811959"}
    assert "scr:2010:11:871" in by_path["2010_11_871_896"]


def test_find_citations_overlap_keeps_first():
    found = find_citations("2019 (8) SCR 2020 INSC 5")
    assert [c for _, _, c in found] == ["scr:2019:8:2020"]


def test_build_alias_rows_wrong_kind_in_column_ignored():
    raw = [
        {
            "path": "2021_1_1_9",
            "scr_citation": "2021 INSC 4",  # not an SCR form
            "neutral_citation": "[2021] 1 S.C.R. 1",  # not a neutral form
            "cnr": "2021 INSC 4",  # not a CNR
        }
    ]
    got = {(r["alias_norm"], r["kind"]) for r in build_alias_rows(raw, {"2021_1_1_9"})}
    assert got == {("scr:2021:1:1", "scr")}
