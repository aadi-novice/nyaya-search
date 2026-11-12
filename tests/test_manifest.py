"""Manifest: column mapping, folder_year, duplicate rule, tar join, outcome_map, dates."""

import json
from datetime import date

import pandas as pd
import pytest

from corpus_helpers import write_tar
from nyaya.config import load_settings
from nyaya.corpus.manifest import (
    MANIFEST_COLUMNS,
    build_manifest,
    folder_year_of,
    load_manifest,
    outcome_for,
    read_metadata,
    split_judges,
)

from conftest import FIXTURES

SETTINGS = load_settings(env_file=None)
DUP = "2010_11_871_896"  # in year=2009 (scraped Oct 2025) and year=2010 (scraped Jun 2025)


def fixture_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "manifest_small.parquet")


def make_root(tmp_path, df: pd.DataFrame, pdfs: dict[int, list[str]]):
    """Write metadata per folder_year and tars holding `<path>_EN.pdf` per year."""
    meta = tmp_path / "meta"
    for y, part in df.groupby("folder_year"):
        d = meta / f"year={y}"
        d.mkdir(parents=True)
        part.drop(columns="folder_year").to_parquet(d / "metadata.parquet", index=False)
    tars = tmp_path / "tars"
    for y, paths in pdfs.items():
        members = [(f"{p}_EN.pdf", f"%PDF {p} {y}".encode()) for p in paths]
        write_tar(tars / f"year={y}" / "english" / "english.tar", members)
    tars.mkdir(exist_ok=True)
    settings = SETTINGS.model_copy(
        update={
            "paths": SETTINGS.paths.model_copy(
                update={
                    "metadata_root": str(meta),
                    "tar_root": str(tars),
                    "build": str(tmp_path / "build"),
                }
            )
        }
    )
    return settings


def all_pdfs_in_own_tar(df: pd.DataFrame) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for p, y in zip(df["path"], df["folder_year"], strict=True):
        if p != "S_1996_6_631_656":  # this one has no PDF anywhere
            out.setdefault(int(y), []).append(p)
    return out


def run(tmp_path, df=None, pdfs=None):
    df = fixture_df() if df is None else df
    pdfs = all_pdfs_in_own_tar(df) if pdfs is None else pdfs
    settings = make_root(tmp_path, df, pdfs)
    out = build_manifest(settings, root=tmp_path)
    summary = json.loads((tmp_path / "build" / "manifest_summary.json").read_text("utf-8"))
    return pd.read_parquet(out), summary, settings


def row(m: pd.DataFrame, path: str) -> pd.Series:
    hit = m[m["path"] == path]
    assert len(hit) == 1
    return hit.iloc[0]


def test_columns_and_one_row_per_path(tmp_path):
    m, summary, _ = run(tmp_path)
    assert list(m.columns) == MANIFEST_COLUMNS
    assert m["path"].is_unique
    assert len(m) == 19
    assert summary["rows_in"] == 20
    assert summary["duplicates_removed"] == 1
    assert summary["judgments"] == 19


def test_column_mapping(tmp_path):
    m, _, _ = run(tmp_path)
    r = row(m, DUP)
    assert r["judges"].tolist() == ["H.S. BEDI", "RAJENDRA MAL LODHA"]
    assert pd.isna(r["author_judge"])
    assert r["scr_citation"] == "[2010] 11 S.C.R. 871"
    assert r["neutral_citation"] == "2009INSC1166"  # stored as given
    assert r["disposal"] == "Appeal(s) allowed"
    assert r["outcome"] == "allowed"
    s = row(m, "S_1996_6_631_656")
    assert pd.isna(s["neutral_citation"])  # blank nc_display
    assert s["judges"].tolist() == ["G.N. RAY"]
    assert isinstance(s["title"], str) and s["cnr"] is not None


def test_dates_and_decision_year(tmp_path):
    m, summary, _ = run(tmp_path)
    r = row(m, DUP)
    assert r["decision_date"] == date(2009, 10, 6)
    assert r["decision_year"] == 2009  # not the `year` column (2010)
    assert row(m, "S_1959_1_1_10")["decision_year"] == 1958
    assert summary["per_decision_year"]["2009"] == {"judgments": 1, "with_pdf": 1}
    assert sum(v["judgments"] for v in summary["per_decision_year"].values()) == 19


def test_outcome_map_including_empty(tmp_path):
    m, _, _ = run(tmp_path)
    got = dict(zip(m["path"], m["outcome"], strict=True))
    assert got["2010_12_854_863"] == "other"  # empty disposal_nature
    assert row(m, "2010_12_854_863")["disposal"] == ""
    assert got["2011_6_744_756"] == "other"  # "Heard" is not listed
    assert got["2010_3_911_1029"] == "dismissed"  # "Dismissed After"
    assert got["2011_6_757_758"] == "disposed"  # "IA disposed"
    assert got["2012_12_947_974"] == "directions"
    assert got["2010_14_647_751"] == "referred"
    assert got["2010_12_551_573"] == "allowed"  # "Case Partly allowed"


def test_outcome_for_ignores_case_and_spacing():
    om = SETTINGS.outcome_map
    assert outcome_for("  appeal(s)   ALLOWED ", om) == "allowed"
    assert outcome_for("", om) == om.default
    assert outcome_for("Something new", om) == om.default


def test_split_judges():
    assert split_judges("A.B. ONE, C. TWO,, ") == ["A.B. ONE", "C. TWO"]
    assert split_judges("") == []


def test_folder_year_comes_from_the_file_path(tmp_path):
    settings = make_root(tmp_path, fixture_df(), {})
    meta = read_metadata(tmp_path / "meta")
    assert set(meta["folder_year"]) == set(fixture_df()["folder_year"])
    assert folder_year_of("a/year=1996/metadata.parquet") == 1996
    with pytest.raises(ValueError):
        folder_year_of("a/b/metadata.parquet")
    assert settings.paths.metadata_root.endswith("meta")


def test_duplicate_prefers_later_scrape_when_both_have_pdfs(tmp_path):
    m, _, _ = run(tmp_path)
    r = row(m, DUP)
    assert r["folder_year"] == 2009  # scraped 2025-10-22, the later copy
    assert r["tar_path"].endswith("year=2009/english/english.tar")


def test_duplicate_prefers_the_copy_whose_pdf_exists(tmp_path):
    pdfs = all_pdfs_in_own_tar(fixture_df())
    pdfs[2009].remove(DUP)  # only the year=2010 tar holds it now
    m, _, _ = run(tmp_path, pdfs=pdfs)
    r = row(m, DUP)
    assert r["folder_year"] == 2010
    assert r["has_pdf"]
    assert r["tar_path"].endswith("year=2010/english/english.tar")


def test_duplicate_tie_goes_to_lowest_tar_path(tmp_path):
    df = fixture_df()
    df.loc[df["path"] == DUP, "scraped_at"] = "2025-06-13T21:14:15.829785"
    m, _, _ = run(tmp_path, df=df)
    assert row(m, DUP)["folder_year"] == 2009  # year=2009 sorts before year=2010


def test_tar_join_falls_back_to_any_tar(tmp_path):
    pdfs = all_pdfs_in_own_tar(fixture_df())
    pdfs[2014].remove("2014_12_875_1036")
    pdfs.setdefault(2026, []).append("2014_12_875_1036")
    m, _, _ = run(tmp_path, pdfs=pdfs)
    r = row(m, "2014_12_875_1036")
    assert r["has_pdf"]
    assert r["tar_path"].endswith("year=2026/english/english.tar")


def test_offsets_point_at_the_pdf(tmp_path):
    from nyaya.corpus.tarstore import read_pdf

    m, _, _ = run(tmp_path)
    r = row(m, "2019_15_258_272")
    data = read_pdf(r["tar_path"], int(r["offset_data"]), int(r["size"]))
    assert data == b"%PDF 2019_15_258_272 2019"


def test_rows_without_pdf_stay_with_null_tar_fields(tmp_path):
    m, summary, settings = run(tmp_path)
    s = row(m, "S_1996_6_631_656")
    assert not s["has_pdf"]
    assert pd.isna(s["tar_path"])
    assert pd.isna(s["offset_data"]) and pd.isna(s["size"])
    assert summary["judgments_without_pdf"] == 1
    assert len(load_manifest(settings, root=tmp_path)) == 19


def test_no_pdf_count_is_after_dedupe(tmp_path):
    m, summary, _ = run(tmp_path, pdfs={})
    assert summary["judgments_without_pdf"] == 19 == (~m["has_pdf"]).sum()


def test_missing_metadata_raises(tmp_path):
    settings = make_root(tmp_path, fixture_df().iloc[:0], {})
    with pytest.raises(FileNotFoundError):
        build_manifest(settings, root=tmp_path)


def test_blank_path_raises(tmp_path):
    df = fixture_df()
    df.loc[3, "path"] = "  "
    settings = make_root(tmp_path, df, {})
    with pytest.raises(ValueError, match="blank path"):
        build_manifest(settings, root=tmp_path)
