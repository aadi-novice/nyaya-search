"""nyaya-build on small fixtures: one Era D PDF made from fixture text, one
out-of-scope PDF, one judgment without a PDF. No tar, GPU or database."""

from __future__ import annotations

import csv
import json
from datetime import date

import numpy as np
import pandas as pd
import pymupdf
import pytest

from judgment_fixtures import ERA_D, JUDGMENTS
from nyaya import build as b
from nyaya.config import load_settings
from nyaya.contracts import ChunkRecord
from nyaya.store import loader

DIM = 1024
D_PATH = "2025_4_1350_1372"


def pdf_from_text(text: str, lines_per_page: int = 45) -> bytes:
    doc = pymupdf.open()
    lines = text.splitlines()
    for start in range(0, len(lines), lines_per_page):
        page = doc.new_page(width=612, height=792)
        y = 120.0
        for line in lines[start : start + lines_per_page]:
            page.insert_text((72, y), line, fontsize=9)
            y += 13.0
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="module")
def settings():
    s = load_settings(env_file=None)
    # a single fixture judgment cannot meet corpus-level G6 shares; the gate
    # itself is tested separately below
    g6 = s.g6.model_copy(update={"min_paragraph_integrity": 0.0, "min_size_compliance": 0.0})
    return s.model_copy(update={"g6": g6})


@pytest.fixture(scope="module")
def pdfs():
    era_d = (JUDGMENTS / f"{ERA_D}.txt").read_text(encoding="utf-8")
    return {
        "d.tar": pdf_from_text(era_d),
        "x.tar": pdf_from_text("A short order with no editorial markers.\nDismissed."),
    }


def manifest_rows():
    base = {
        "petitioner": "A",
        "respondent": "B",
        "judges": ["ABHAY S. OKA", "UJJAL BHUYAN"],
        "author_judge": "ABHAY S. OKA",
        "cnr": "",
        "disposal": "Dismissed",
        "outcome": "dismissed",
        "decision_date": date(2025, 4, 21),
        "decision_year": 2025,
        "folder_year": 2025,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "path": D_PATH,
                "title": "Arunkumar H Shah HUF versus Avon Arcade Premises",
                "scr_citation": "[2025] 4 S.C.R. 1350",
                "neutral_citation": "2025INSC530",
                "cnr": "ESCR010004242025",
                "has_pdf": True,
                "tar_path": "d.tar",
                "offset_data": 0,
                "size": 1,
            },
            {
                **base,
                "path": "2025_4_1_5",
                "title": "Out v. Scope",
                "scr_citation": "[2025] 4 S.C.R. 1",
                "neutral_citation": None,
                "has_pdf": True,
                "tar_path": "x.tar",
                "offset_data": 0,
                "size": 1,
            },
            {
                **base,
                "path": "2025_4_9_9",
                "title": "No v. Pdf",
                "scr_citation": "[2025] 4 S.C.R. 9",
                "neutral_citation": None,
                "has_pdf": False,
                "tar_path": None,
                "offset_data": pd.NA,
                "size": pd.NA,
            },
            {
                **base,
                "path": "2024_1_1_2",
                "title": "Other v. Year",
                "scr_citation": "",
                "neutral_citation": None,
                "decision_year": 2024,
                "has_pdf": False,
                "tar_path": None,
                "offset_data": pd.NA,
                "size": pd.NA,
            },
        ]
    ).astype({"offset_data": "Int64", "size": "Int64"})


@pytest.fixture
def built(tmp_path, settings, pdfs):
    paths = b.BuildPaths(tmp_path)
    manifest_rows().to_parquet(paths.manifest, index=False)
    proc = b.DocProcessor(
        settings,
        count_tokens=lambda s: len(s.split()),
        read_pdf=lambda tar, off, size: pdfs[tar],
    )
    reports = b.parse_chunk(settings, [2025], paths=paths, processor=proc)
    return paths, proc, reports


# --- arguments and helpers -----------------------------------------------------------


def test_parse_years():
    assert b.parse_years("2025") == [2025]
    assert b.parse_years("2010-2012") == [2010, 2011, 2012]
    assert b.parse_years("2025, 2024,2025") == [2024, 2025]
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        b.parse_years("2026-2010")
    with pytest.raises(argparse.ArgumentTypeError):
        b.parse_years(" , ")


def test_token_batches_respect_budget_and_cover_everything():
    lengths = [10, 500, 30, 2000, 40, 40, 900]
    batches = b.token_batches(lengths, budget=1000, max_batch=3)
    assert sorted(i for bt in batches for i in bt) == list(range(len(lengths)))
    for bt in batches:
        assert len(bt) <= 3
        assert len(bt) == 1 or len(bt) * max(lengths[i] for i in bt) <= 1000
    assert batches[0] == [3]  # longest first, alone when over budget


def test_opinion_judges_adds_author():
    assert b.opinion_judges({"judges": ["A"], "author_judge": "B"}) == ["A", "B"]
    assert b.opinion_judges({"judges": ["A"], "author_judge": "A"}) == ["A"]
    assert b.opinion_judges({"judges": None, "author_judge": None}) == []


# --- parse-chunk -------------------------------------------------------------------


def test_parse_chunk_writes_the_loader_layout(built, settings):
    paths, _, reports = built
    (rep,) = reports
    assert rep.judgments == 3 and rep.with_pdf == 2
    assert rep.by_era == {"d": 1, "out_of_scope": 1, "no_pdf": 1}
    assert rep.errors == 0 and rep.chunks > 3

    j = pd.read_parquet(paths.judgments(2025))
    assert set(j["path"]) == {D_PATH, "2025_4_1_5", "2025_4_9_9"}
    d = j.set_index("path").loc[D_PATH]
    assert d["era"] == "d" and d["clean_text"] and json.loads(d["position_map"])[0][1] == 1350
    assert d["tar_file"] == "d.tar"
    nopdf = j.set_index("path").loc["2025_4_9_9"]
    assert not nopdf["has_pdf"] and pd.isna(nopdf["clean_text"])

    chunks = loader.read_chunks(paths.root)
    assert {c.judgment_id for c in chunks} == {D_PATH}
    zones = {c.zone.value for c in chunks}
    assert {"issue", "editorial", "judgment"} <= zones
    refs = pd.read_parquet(paths.refs(2025))
    assert (refs["kind"] == "act").any()
    links = pd.read_parquet(paths.links(2025))
    assert list(links.columns) == ["from_chunk_id", "to_chunk_id"]

    aliases = pd.read_parquet(paths.root / "aliases.parquet")
    got = set(zip(aliases["alias_norm"], aliases["path"], strict=True))
    assert ("insc:2025:530", D_PATH) in got
    assert ("scr:2025:4:1350", D_PATH) in got
    assert not any(p == "2024_1_1_2" for _, p in got)  # not built

    metrics = json.loads((paths.root / "chunk_metrics.json").read_text())
    assert metrics["tokens_counted"] is True and metrics["over_model_limit"] == 0
    assert metrics["years"] == [2025]
    assert (paths.root / "link_summary.json").exists()
    assert (paths.root / "act_names_unmapped.csv").read_text().startswith("name,count")

    with (paths.root / "skipped.csv").open(encoding="utf-8") as f:
        skipped = list(csv.DictReader(f))
    reasons = {(r["path"], r["reason"]) for r in skipped}
    assert ("2025_4_9_9", "no_pdf") in reasons
    assert ("2025_4_1_5", "era_out_of_scope") in reasons
    assert (paths.root / "errors.jsonl").read_text() == ""


def test_parse_chunk_resumes_finished_years(built, settings):
    paths, proc, _ = built
    calls = []

    def counting(row):
        calls.append(row["path"])
        return proc(row)

    (rep,) = b.parse_chunk(settings, [2025], paths=paths, processor=counting)
    assert rep.skipped and calls == []
    b.parse_chunk(settings, [2025], paths=paths, processor=counting, force=True)
    assert len(calls) == 3


def test_errors_are_logged_and_over_the_share_exit_4(tmp_path, settings, pdfs):
    paths = b.BuildPaths(tmp_path)
    manifest_rows().to_parquet(paths.manifest, index=False)

    def broken(tar, off, size):
        raise OSError("tar unreadable")

    proc = b.DocProcessor(settings, count_tokens=len, read_pdf=broken)
    with pytest.raises(SystemExit) as exc:
        b.parse_chunk(settings, [2025], paths=paths, processor=proc)
    assert exc.value.code == b.ERROR_EXIT_CODE
    errors = [json.loads(x) for x in (paths.root / "errors.jsonl").read_text().splitlines()]
    assert {e["path"] for e in errors} == {D_PATH, "2025_4_1_5"}
    assert errors[0]["error"].startswith("OSError")
    # the judgment rows are still written, without text
    j = pd.read_parquet(paths.judgments(2025))
    assert j["clean_text"].isna().all()


def test_g6_failure_exits_3(tmp_path, pdfs):
    s = load_settings(env_file=None)
    g6 = s.g6.model_copy(update={"min_paragraph_integrity": 1.01})
    s = s.model_copy(update={"g6": g6})
    paths = b.BuildPaths(tmp_path)
    manifest_rows().to_parquet(paths.manifest, index=False)
    proc = b.DocProcessor(s, count_tokens=len, read_pdf=lambda t, o, n: pdfs[t])
    with pytest.raises(SystemExit) as exc:
        b.parse_chunk(s, [2025], paths=paths, processor=proc)
    assert exc.value.code == b.G6_EXIT_CODE


def test_tokens_not_counted_fails_the_gate(tmp_path, settings, pdfs):
    paths = b.BuildPaths(tmp_path)
    manifest_rows().to_parquet(paths.manifest, index=False)
    proc = b.DocProcessor(settings, count_tokens=None, read_pdf=lambda t, o, n: pdfs[t])
    with pytest.raises(SystemExit) as exc:
        b.parse_chunk(settings, [2025], paths=paths, processor=proc)
    assert exc.value.code == b.G6_EXIT_CODE


# --- embed, then the loader reads it -----------------------------------------------------


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts, batch_size=16):
        self.calls += 1
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), DIM)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_embed_writes_vectors_the_loader_accepts(built, settings):
    paths, _, _ = built
    fake = FakeEmbedder()
    (rep,) = b.embed(settings, [2025], paths=paths, embedder=fake, count_tokens=len)
    assert rep.chunks > 0 and not rep.skipped
    ids = paths.vector_ids(2025).read_text().split()
    chunk_ids = [c.chunk_id for c in loader.read_chunks(paths.root)]
    assert ids == chunk_ids
    vecs = np.load(paths.vectors(2025))
    assert vecs.shape == (len(ids), DIM) and np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-5)

    data = loader.read_build(paths.root, DIM, years=[2025])
    assert loader.check_ids([c.chunk_id for c in data.chunks], data.vector_ids) == []
    assert loader.check_g6(paths.root / "chunk_metrics.json", settings) == []
    assert {a[2] for a in data.aliases} <= data.paths

    # finished: skipped; force: re-embedded
    calls = fake.calls
    (again,) = b.embed(settings, [2025], paths=paths, embedder=fake, count_tokens=len)
    assert again.skipped and fake.calls == calls
    b.embed(settings, [2025], paths=paths, embedder=fake, count_tokens=len, force=True)
    assert fake.calls > calls


def test_embed_redoes_a_year_whose_chunks_changed(built, settings):
    paths, _, _ = built
    fake = FakeEmbedder()
    b.embed(settings, [2025], paths=paths, embedder=fake, count_tokens=len)
    lines = paths.chunks(2025).read_text(encoding="utf-8").splitlines()
    paths.chunks(2025).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    calls = fake.calls
    (rep,) = b.embed(settings, [2025], paths=paths, embedder=fake, count_tokens=len)
    assert not rep.skipped and fake.calls > calls
    assert len(paths.vector_ids(2025).read_text().split()) == len(lines) - 1


def test_loader_reads_only_the_requested_years(built, settings):
    paths, _, _ = built
    b.embed(settings, [2025], paths=paths, embedder=FakeEmbedder(), count_tokens=len)
    with pytest.raises(ValueError, match="no judgment files"):
        loader.read_build(paths.root, DIM, years=[2019])


def test_chunk_records_round_trip(built):
    paths, _, _ = built
    for line in paths.chunks(2025).read_text(encoding="utf-8").splitlines():
        c = ChunkRecord.model_validate_json(line)
        assert c.chunk_id.startswith(D_PATH + "#")


def test_cli_parser_accepts_the_documented_commands():
    p = b.build_parser()
    assert p.parse_args(["parse-chunk", "--years", "2010-2026"]).years[-1] == 2026
    assert p.parse_args(["embed", "--years", "2025"]).years == [2025]
    assert p.parse_args(["load", "--version", "v1"]).years is None
    assert p.parse_args(["run", "--years", "2025", "--version", "v1"]).workers == 1
