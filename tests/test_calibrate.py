"""Calibration of the Era C margin and furniture bands, on synthetic PDFs."""

import json
import shutil

import pandas as pd
import pymupdf
import pytest

from corpus_helpers import (
    FIRST_BODY_Y,
    LEFT_LETTER_X,
    PAGE_H,
    PAGE_W,
    RIGHT_LETTER_X,
    TOP_HEAD_Y,
    clean_settings,
    era_c_page,
    era_c_pdf,
    plain_pdf,
    write_tar,
)
from nyaya.config import load_settings
from nyaya.parse.calibrate import (
    EXIT_CODE,
    NO_BAND_MESSAGE,
    CalibrationError,
    calibrate,
    cluster_band,
    furniture_band,
    measure_bands,
)

from conftest import ROOT

SETTINGS = clean_settings()
PATTERNS = SETTINGS.running_head_patterns
BODY = [
    "HELD: 1. The appeal is allowed and the order is set aside.",
    "A proviso does not travel beyond the provision to which it is",
    "a proviso, and the leading provision prevails over the other.",
]


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


def doc_pages(n: int = 4) -> list[list[str]]:
    """Body text that differs from page to page, as in a real judgment."""
    pages = []
    for i in range(n):
        w = WORDS[i:] + WORDS[:i]
        pages.append([BODY[0] if i == 0 else " ".join(w[:6]), " ".join(w[2:9]), " ".join(w[4:])])
    return pages


def era_c_docs(k: int = 6, start: int = 786) -> list[tuple[str, bytes]]:
    return [
        (
            f"2020_11_{start + 10 * i}_{start + 10 * i + 3}",
            era_c_pdf(doc_pages(), start_page=start + 10 * i),
        )
        for i in range(k)
    ]


def test_measure_bands_finds_margin_and_head_bands():
    bands = measure_bands(era_c_docs(), PATTERNS)
    lo, hi = bands["era_c"]["left_x"]
    assert lo <= LEFT_LETTER_X / PAGE_W and hi >= (LEFT_LETTER_X + 7) / PAGE_W
    assert hi < 163 / PAGE_W + 0.03  # stays near the letters, not across the body
    lo, hi = bands["era_c"]["right_x"]
    assert lo <= RIGHT_LETTER_X / PAGE_W <= hi
    t0, t1 = bands["top_y"]
    assert t0 <= TOP_HEAD_Y / PAGE_H <= t1
    assert t1 < FIRST_BODY_Y / PAGE_H
    assert bands["bottom_y"] == []  # no footers in these pages


def test_bands_json_is_accepted_by_settings():
    bands = measure_bands(era_c_docs(), PATTERNS)
    s = SETTINGS.model_copy(update={"bands": SETTINGS.bands.model_validate(bands)})
    assert s.bands.era_c.left_x is not None and s.bands.bottom_y is None


def test_no_margin_letters_means_no_band():
    docs = [("2020_11_786_789", era_c_pdf(doc_pages(), letters=False))] * 5
    with pytest.raises(CalibrationError) as exc:
        measure_bands(docs, PATTERNS)
    assert exc.value.code == EXIT_CODE == 2
    assert str(exc.value) == NO_BAND_MESSAGE


def test_letters_on_one_side_only_means_no_band():
    doc = pymupdf.open()
    for pno in range(4):  # letters only on even (left-letter) pages
        era_c_page(doc, pno, BODY, printed_page=786 + pno, letters=pno % 2 == 0)
    data = doc.tobytes()
    with pytest.raises(CalibrationError, match=NO_BAND_MESSAGE):
        measure_bands([("2020_11_786_789", data)] * 6, PATTERNS)


def test_cluster_band_ignores_sparse_noise():
    letters = [(0.235, 0.248, 0.2415)] * 30 + [(0.40, 0.41, 0.405), (0.30, 0.31, 0.305)]
    lo, hi = cluster_band(letters)
    assert lo < 0.235 and 0.248 < hi < 0.30
    assert cluster_band(letters[:5]) is None


def test_furniture_band_needs_enough_lines():
    assert furniture_band([(0.1, 0.12)] * 4) is None
    assert furniture_band([(0.1, 0.12)] * 5) == (0.095, 0.125)


# --- the whole step --------------------------------------------------------------


def make_root(tmp_path, docs, years=None, era_d=0):
    """A repo root with config/, a tar of `docs` and a manifest pointing at it."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    members = [(f"{p}_EN.pdf", d) for p, d in docs]
    members += [
        (
            f"2023_1_{i}_{i}_EN.pdf",
            plain_pdf([["Issue for Consideration", "Headnotes", "List of Acts"]]),
        )
        for i in range(era_d)
    ]
    tar = write_tar(tmp_path / "tars" / "year=2020" / "english" / "english.tar", members)
    from nyaya.corpus.tarstore import scan_offsets

    offs = scan_offsets(tar)
    rows = []
    for i, (name, _) in enumerate(members):
        path = name.removesuffix("_EN.pdf")
        o, sz = offs[name]
        year = years[i] if years else 2020
        rows.append(
            {"path": path, "has_pdf": True, "decision_year": year, "tar_path": tar.as_posix(),
             "offset_data": o, "size": sz}
        )  # fmt: skip
    rows.append(
        {"path": "2019_1_1_2", "has_pdf": False, "decision_year": 2019, "tar_path": None,
         "offset_data": None, "size": None}
    )  # fmt: skip
    (tmp_path / "build").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "build" / "manifest.parquet", index=False)
    return tmp_path


def small_settings(sample_size=20, min_era_c=4):
    return SETTINGS.model_copy(
        update={
            "calibrate": SETTINGS.calibrate.model_copy(
                update={"sample_size": sample_size, "min_era_c": min_era_c}
            )
        }
    )


def test_calibrate_writes_bands_json_that_settings_merge(tmp_path):
    root = make_root(tmp_path, era_c_docs(), era_d=2)
    out = calibrate(small_settings(), root=root)
    assert out == root / "build" / "bands.json"
    bands = json.loads(out.read_text("utf-8"))
    assert set(bands) == {"era_c", "top_y", "bottom_y"}
    merged = load_settings(root, env_file=None)
    assert merged.bands.era_c.left_x == tuple(bands["era_c"]["left_x"])
    assert merged.versions.config.startswith(SETTINGS.versions.config + "-")


def test_calibrate_stops_with_too_few_era_c(tmp_path, capsys):
    root = make_root(tmp_path, era_c_docs(3), era_d=3)
    with pytest.raises(CalibrationError) as exc:
        calibrate(small_settings(min_era_c=4), root=root)
    assert exc.value.code == 2
    assert "only 3 era C" in capsys.readouterr().err
    assert not (root / "build" / "bands.json").exists()


def test_calibrate_samples_only_2010_to_2023(tmp_path):
    docs = era_c_docs(6)
    root = make_root(tmp_path, docs, years=[2009, 2024, 2020, 2020, 2020, 2020])
    with pytest.raises(CalibrationError, match="only 4"):
        calibrate(small_settings(min_era_c=5), root=root)


def test_calibrate_reports_missing_band(tmp_path, capsys):
    doc = pymupdf.open()
    for pno in range(4):
        era_c_page(doc, pno, BODY, printed_page=786 + pno, letters=pno % 2 == 0)
    data = doc.tobytes()
    docs = [(f"2020_11_{786 + 10 * i}_{789 + 10 * i}", data) for i in range(6)]
    root = make_root(tmp_path, docs)
    with pytest.raises(CalibrationError) as exc:
        calibrate(small_settings(), root=root)
    assert exc.value.code == 2
    assert NO_BAND_MESSAGE in capsys.readouterr().err
