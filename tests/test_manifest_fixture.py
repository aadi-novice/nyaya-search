"""Shape of tests/fixtures/manifest_small.parquet (design file 14).

The manifest logic and its test (test_manifest.py) are Phase 1.
"""

import pandas as pd

from conftest import FIXTURES

REAL_COLUMNS = {
    "title", "petitioner", "respondent", "description", "judge", "author_judge", "citation",
    "case_id", "cnr", "decision_date", "disposal_nature", "court", "available_languages",
    "raw_html", "path", "nc_display", "scraped_at", "year",
}  # fmt: skip


def test_manifest_small_fixture():
    df = pd.read_parquet(FIXTURES / "manifest_small.parquet")
    assert 15 <= len(df) <= 25
    assert set(df.columns) == REAL_COLUMNS | {"folder_year"}
    dup = df[df.duplicated("path", keep=False)]
    assert dup["path"].nunique() == 1 and len(dup) == 2
    assert dup["folder_year"].nunique() == 2 and dup["scraped_at"].nunique() == 2
    assert (df["disposal_nature"].fillna("").str.strip() == "").sum() >= 1
    assert df["path"].str.startswith("S_").any()
