"""`nyaya-eval g6-holdout`: sampling, the per-PDF review files and the CLI, on synthetic PDFs."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from corpus_helpers import clean_settings, plain_pdf
from nyaya.config import ConfigError
from nyaya.contracts import Era, ParsedDoc
from nyaya.eval import cli
from nyaya.eval.g6_holdout import (
    head_hits,
    holdout_order,
    margin_hits,
    render_pages,
    run_holdout,
    text_by_page,
)
from nyaya.parse.calibrate import calibration_sample

SETTINGS = clean_settings()


def manifest(n: int = 40, years=range(2008, 2027)) -> pd.DataFrame:
    """`n` rows per year; every tenth has no PDF. tar_path carries the path (fake reader)."""
    rows = []
    for year in years:
        for i in range(n):
            path = f"{year}_1_{100 + 10 * i}_{105 + 10 * i}"
            rows.append(
                {
                    "path": path,
                    "has_pdf": i % 10 != 9,
                    "decision_year": year,
                    "tar_path": path,
                    "offset_data": 0,
                    "size": 1,
                }
            )
    return pd.DataFrame(rows)


def settings(holdout_size: int = 5, calib_size: int = 30):
    return SETTINGS.model_copy(
        update={
            "g6": SETTINGS.g6.model_copy(update={"holdout_size": holdout_size}),
            "calibrate": SETTINGS.calibrate.model_copy(update={"sample_size": calib_size}),
        }
    )


# --- sampling ----------------------------------------------------------------------------


def test_order_excludes_calibration_rows_and_respects_years_and_pdfs():
    m = manifest()
    s = settings()
    excluded = set(calibration_sample(m, s)["path"])
    assert len(excluded) == 30
    order = holdout_order(m, range(2010, 2027), excluded, seed=4242)
    assert not excluded & set(order["path"])
    assert order["has_pdf"].all()
    assert order["decision_year"].between(2010, 2026).all()
    eligible = m[m["has_pdf"] & m["decision_year"].between(2010, 2026)]
    assert len(order) == len(eligible) - len(excluded & set(eligible["path"]))


def test_order_is_seeded_and_independent_of_manifest_row_order():
    m = manifest()
    a = holdout_order(m, [2025], [], seed=4242)["path"].tolist()
    b = holdout_order(m.iloc[::-1], [2025], [], seed=4242)["path"].tolist()
    c = holdout_order(m, [2025], [], seed=7)["path"].tolist()
    assert a == b
    assert a != c and sorted(a) == sorted(c)


def test_order_of_an_empty_pool_is_empty():
    assert holdout_order(manifest(), [1999], [], seed=1).empty


# --- hints and page split ------------------------------------------------------------------


def test_margin_and_head_hints():
    text = "the appeal C is allowed\nSUPREME COURT REPORTS [2015] 2 S.C.R.\nclean line"
    (hit,) = margin_hits(text)
    assert hit.startswith("the appeal C is allowed") and "\n" not in hit
    assert head_hits(text, ["Supreme Court Reports"]) == ["SUPREME COURT REPORTS [2015] 2 S.C.R."]
    assert margin_hits("Article 21 of the Constitution") == []


def test_text_by_page_follows_the_position_map():
    text = "first line\nsecond line\nthird line"
    doc = ParsedDoc(
        path="2025_1_230_238",
        era=Era.D,
        clean_text=text,
        position_map=[(0, 230), (11, 230), (23, 231)],
        pages=2,
        clean_text_version="ct1",
    )
    assert text_by_page(doc, 230) == {0: "first line\nsecond line", 1: "third line"}


def test_render_pages_writes_one_png_per_page(tmp_path):
    pdf = plain_pdf([["page one"], ["page two"]])
    assert render_pages(pdf, tmp_path, 40) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["page-001.png", "page-002.png"]


# --- the run -------------------------------------------------------------------------------


def fake_reader(pdfs: dict[str, bytes]):
    def read(tar_path: str, offset: int, size: int) -> bytes:
        return pdfs.get(tar_path, PDF)

    return read


PDF = plain_pdf(
    [
        ["1. The appeal is allowed.", "2. The order of the High Court is set aside."],
        ["3. No order as to costs.", "SUPREME COURT REPORTS"],
    ]
)


def no_render(pdf, out_dir, dpi):
    return 2


def test_run_writes_review_files_index_and_handcheck(tmp_path):
    m = manifest()
    s = settings(holdout_size=4)
    out = tmp_path / "g6"
    result = run_holdout(
        m,
        s,
        range(2010, 2027),
        out,
        read_pdf=fake_reader({}),
        classify_pdf=lambda pdf, year: Era.D,
        render=render_pages,
        dpi=30,
    )
    assert len(result.reviewed) == 4 and result.skipped == []
    first = result.reviewed[0]
    d = out / first.path
    assert (d / "page-001.png").exists() and (d / "page-002.png").exists()
    assert "No order as to costs." in (d / "clean_text.txt").read_text("utf-8")
    review = (d / "review.html").read_text("utf-8")
    assert 'src="page-001.png"' in review and "set aside" in review
    assert first.pdf_pages == 2

    index = (out / "index.html").read_text("utf-8")
    assert all(f"{r.path}/review.html" in index for r in result.reviewed)
    with (out / "handcheck.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["path"] for r in rows] == [r.path for r in result.reviewed]
    assert rows[0]["margin_letters_found"] == "" and rows[0]["era"] == "d"

    sample = json.loads((out / "sample.json").read_text("utf-8"))
    assert sample["seed"] == SETTINGS.g6.holdout_seed and sample["size"] == 4
    assert sample["paths"] == [r.path for r in result.reviewed]
    assert not set(sample["excluded_calibration_paths"]) & set(sample["paths"])
    assert result.summary()["reviewed"] == 4


def test_run_is_reproducible(tmp_path):
    kw = dict(read_pdf=fake_reader({}), classify_pdf=lambda p, y: Era.D, render=no_render)
    a = run_holdout(manifest(), settings(), [2025, 2026], tmp_path / "a", **kw)
    b = run_holdout(manifest(), settings(), [2025, 2026], tmp_path / "b", **kw)
    assert [r.path for r in a.reviewed] == [r.path for r in b.reviewed]


def test_out_of_scope_and_failing_pdfs_are_skipped_and_replaced(tmp_path):
    m = manifest()
    order = holdout_order(m, [2025], calibration_sample(m, settings())["path"], 4242)
    oos, broken = order["path"].iloc[0], order["path"].iloc[1]
    pdfs = {broken: b"not a pdf"}

    def classify(pdf, year):
        return Era.OUT_OF_SCOPE if pdf is OOS else Era.D

    OOS = plain_pdf([["scanned"]])
    pdfs[oos] = OOS
    result = run_holdout(
        m,
        settings(holdout_size=3),
        [2025],
        tmp_path,
        read_pdf=fake_reader(pdfs),
        classify_pdf=classify,
        render=no_render,
    )
    assert len(result.reviewed) == 3
    skipped = {s["path"]: s["reason"] for s in result.skipped}
    assert skipped == {oos: "era_out_of_scope", broken: "error"}
    assert [r.path for r in result.reviewed] == order["path"].iloc[2:5].tolist()


def test_too_few_eligible_pdfs_returns_what_there_is(tmp_path):
    m = manifest(n=3, years=[2025])
    result = run_holdout(
        m,
        settings(holdout_size=50, calib_size=0),
        [2025],
        tmp_path,
        read_pdf=fake_reader({}),
        classify_pdf=lambda p, y: Era.D,
        render=no_render,
    )
    assert len(result.reviewed) == 3  # three rows, all with a PDF


def test_uncalibrated_era_c_stops_the_run(tmp_path):
    s = settings()
    s = s.model_copy(update={"bands": s.bands.model_copy(update={"era_c": type(s.bands.era_c)()})})
    with pytest.raises(ConfigError, match="not calibrated"):
        run_holdout(
            manifest(),
            s,
            [2015],
            tmp_path,
            read_pdf=fake_reader({}),
            classify_pdf=lambda p, y: Era.C,
            render=no_render,
        )


# --- CLI ----------------------------------------------------------------------------------


def test_cli_runs_and_exit_code_reflects_the_sample_size(monkeypatch, tmp_path, capsys):
    import nyaya.corpus.manifest as manifest_mod
    import nyaya.eval.g6_holdout as g6

    seen = {}

    def fake_run(m, s, years, out, *, size, dpi):
        seen.update(years=years, out=out, size=size, dpi=dpi)
        return g6.HoldoutResult(out, [object()] * (size or 0), [], 0, 10)

    monkeypatch.setattr(g6.HoldoutResult, "summary", lambda self: {"reviewed": len(self.reviewed)})
    monkeypatch.setattr(cli, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(manifest_mod, "load_manifest", lambda s: manifest())
    monkeypatch.setattr(g6, "run_holdout", fake_run)
    assert cli.main(["g6-holdout", "--years", "2025", "--size", "2", "--out", str(tmp_path)]) == 0
    assert seen == {"years": [2025], "out": tmp_path, "size": 2, "dpi": g6.DEFAULT_DPI}
    assert json.loads(capsys.readouterr().out) == {"reviewed": 2}
    # fewer than g6.holdout_size reviewed: exit 2
    monkeypatch.setattr(g6, "run_holdout", lambda *a, **k: g6.HoldoutResult(tmp_path, [], [], 0, 0))
    assert cli.main(["g6-holdout", "--out", str(tmp_path)]) == 2
