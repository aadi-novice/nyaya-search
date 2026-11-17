"""The BM25 keyword leg (nyaya.search.bm25) and its use in PostgresSearch."""

import json

import numpy as np
import pytest

from nyaya.config import load_settings
from nyaya.contracts import Filters
from nyaya.search import bm25 as bm
from nyaya.search import postgres as pg
from test_postgres_sql import FakeConn

ROWS = [
    ("A#judgment#001", "Anticipatory bail under section 438 needs gravity of accusation weighed"),
    ("B#judgment#001", "The arbitral award was set aside under section 34 for patent illegality"),
    ("C#judgment#001", "A dying declaration must inspire confidence before conviction rests on it"),
    ("D#judgment#001", "Land acquisition compensation: market value from sale deeds of the court"),
    ("E#judgment#001", "The court said the court may hear the court appeal in court"),
]


@pytest.fixture(scope="module")
def index():
    return bm.Bm25Index.build(iter(ROWS), index_version="v1")


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def test_build_meta_and_search(index):
    assert index.meta == bm.Bm25Meta("v1", 5, "", 1.5, 0.75)
    assert index.search("arbitral award illegality", 3)[0] == "B#judgment#001"
    assert index.search("the of and", 5) == []  # stop words only
    assert index.search("zebra", 5) == []
    assert index.search("bail", 0) == []


def test_unstemmed_by_default(index):
    assert index.search("declarations", 1) == []
    assert index.search("dying declaration", 1) == ["C#judgment#001"]


def test_stemming_matches_word_forms():
    pytest.importorskip("Stemmer")
    idx = bm.Bm25Index.build(iter(ROWS), index_version="v1", stemmer="english")
    # "declarations" and "convicted" stem to the indexed "declar" / "convict"
    assert idx.search("dying declarations convicted", 1) == ["C#judgment#001"]


def test_idf_beats_repetition_of_common_words(index):
    # "court" is repeated in E, but the rare "acquisition" decides
    assert index.search("court land acquisition", 1) == ["D#judgment#001"]


def test_allowed_mask_filters(index):
    allowed = index.mask(["A#judgment#001", "C#judgment#001", "unknown"])
    assert allowed.sum() == 2
    assert index.search("section arbitral award", 5, allowed=allowed) == ["A#judgment#001"]
    assert index.search("arbitral", 5, allowed=allowed) == []


def test_phrase_matches_come_first(index):
    first = index.mask(["E#judgment#001"])
    out = index.search("arbitral award court", 5, first=first)
    assert out[0] == "E#judgment#001" and "B#judgment#001" in out
    # a phrase match is returned even with no scored word in common
    assert index.search("zebra", 5, first=first) == ["E#judgment#001"]


def test_ties_break_by_chunk_order():
    rows = [("b", "same words here"), ("a", "same words here"), ("c", "other text")]
    idx = bm.Bm25Index.build(iter(rows), index_version="v", stemmer="")
    assert idx.search("same words", 5) == ["b", "a"]  # index order (as built)


def test_build_rejects_empty_and_duplicates():
    with pytest.raises(ValueError):
        bm.Bm25Index.build(iter([]), index_version="v")
    with pytest.raises(ValueError):
        bm.Bm25Index.build(iter([("a", "x y"), ("a", "z w")]), index_version="v")
    with pytest.raises(ValueError):
        bm.Bm25Index.build(iter([("a", "x y")]), index_version="v", stemmer="latin")


def test_save_load_round_trip(index, tmp_path):
    d = index.save(tmp_path / "v1")
    assert {"ids.json", "meta.json"} <= {p.name for p in d.iterdir()}
    loaded = bm.Bm25Index.load(d)
    assert loaded.meta == index.meta and loaded.ids == index.ids
    for q in ("arbitral award", "dying declarations", "court land acquisition"):
        assert loaded.search(q, 5) == index.search(q, 5)
    # replacing an existing (not memory-mapped) directory works
    other = index.save(tmp_path / "v1b")
    index.save(other)
    assert bm.Bm25Index.load(other, mmap=False).ids == index.ids
    assert not (tmp_path / "v1b.tmp").exists()


def test_open_index_reuses_current_and_rebuilds_stale(index, tmp_path, monkeypatch):
    d = tmp_path / "v1"
    index.save(d)
    calls = []

    def fake_build(conn, version, *, stemmer, directory):
        calls.append((version, stemmer, directory))
        return index

    monkeypatch.setattr(bm, "build_index", fake_build)
    monkeypatch.setattr(bm, "chunk_count", lambda conn, v: 5)
    assert bm.open_index(None, "v1", stemmer="", directory=d).ids == index.ids
    assert calls == []
    bm.open_index(None, "v1", stemmer="english", directory=d)  # stemmer changed
    monkeypatch.setattr(bm, "chunk_count", lambda conn, v: 6)
    bm.open_index(None, "v1", stemmer="", directory=d)  # chunk count changed
    bm.open_index(None, "v2", stemmer="", directory=d)  # another version
    (d / "meta.json").write_text("{not json", encoding="utf-8")
    bm.open_index(None, "v1", stemmer="", directory=d)  # torn
    bm.open_index(None, "v1", stemmer="", directory=None)  # in memory only
    assert len(calls) == 5 and calls[-1][2] is None


def test_keyword_index_respects_the_engine_setting(settings, monkeypatch):
    pg_settings = settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"keyword_engine": "postgres"})}
    )
    assert bm.keyword_index(pg_settings, None, "v1") is None
    seen = {}
    monkeypatch.setattr(
        bm, "open_index", lambda conn, v, *, stemmer, directory: seen.update(d=directory)
    )
    bm.keyword_index(settings, None, "v1")
    assert seen["d"].parts[-2:] == ("bm25", "v1")
    bm.keyword_index(settings, None, "v1", save=False)
    assert seen["d"] is None


def test_index_dir():
    assert bm.index_dir("build", "p1").as_posix() == "build/bm25/p1"


# --- PostgresSearch with a BM25 index ------------------------------------------------------


def test_engine_keyword_uses_bm25_without_filters(index, settings):
    conn = FakeConn([])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn, bm25=index)
    hits = eng.keyword("arbitral award illegality", None, 2)
    assert hits[0].chunk_id == "B#judgment#001"
    assert [(h.keyword_rank, h.fused_rank) for h in hits] == [(1, 1.0), (2, 2.0)][: len(hits)]
    assert conn.executed == []  # no filters, no phrases: no SQL at all


def test_engine_keyword_bm25_applies_filters_in_sql(index, settings):
    conn = FakeConn([[("A#judgment#001",), ("C#judgment#001",)]])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn, bm25=index)
    hits = eng.keyword("section arbitral award", Filters(year_from=2020), 5)
    assert [h.chunk_id for h in hits] == ["A#judgment#001"]
    sql, params = conn.executed[0]
    assert "j.decision_year >= %(year_from)s" in sql and params["year_from"] == 2020
    assert params["index_version"] == "v1"


def test_engine_keyword_bm25_no_chunk_passes_filters(index, settings):
    conn = FakeConn([[]])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn, bm25=index)
    assert eng.keyword("arbitral", Filters(year_from=2030), 5) == []
    assert len(conn.executed) == 1


def test_engine_keyword_bm25_puts_phrase_matches_first(index, settings):
    conn = FakeConn([[("E#judgment#001",)]])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn, bm25=index)
    hits = eng.keyword('"court appeal" arbitral award', None, 5)
    assert hits[0].chunk_id == "E#judgment#001"
    sql, params = conn.executed[0]
    assert "phraseto_tsquery('english', %(phrase_0)s)" in sql
    assert params["phrase_0"] == "court appeal" and params["cap"] == pg.PHRASE_CAP


def test_engine_fts_keyword_still_available(index, settings):
    conn = FakeConn([[("a", 0.5)]])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn, bm25=index)
    assert [h.chunk_id for h in eng.fts_keyword("bail", None, 2)] == ["a"]
    assert "ts_rank(c.tsv, q.query, 1)" in conn.executed[0][0]


def test_mask_is_boolean(index):
    m = index.mask([])
    assert m.dtype == np.bool_ and m.shape == (5,) and not m.any()


def test_meta_file_is_plain_json(index, tmp_path):
    d = index.save(tmp_path / "x")
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["format"] == bm.FORMAT
