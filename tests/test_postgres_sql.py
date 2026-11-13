"""SQL builders and row mapping for PostgresSearch, without a database."""

from datetime import date

import pytest

from nyaya.config import load_settings
from nyaya.contracts import Filters, Zone
from nyaya.search import postgres as pg


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


@pytest.fixture(scope="module")
def omap(settings):
    return settings.outcome_map


# --- filters ---------------------------------------------------------------------


def test_no_filters(omap):
    w = pg.build_filter_sql(None, omap)
    assert (w.sql, w.params, w.needs_judgments) == ("", {}, False)
    w = pg.build_filter_sql(Filters(), omap)
    assert w.sql == "" and w.params == {}


def test_year_range(omap):
    w = pg.build_filter_sql(Filters(year_from=2015, year_to=2020), omap)
    assert "j.decision_year >= %(year_from)s" in w.sql
    assert "j.decision_year <= %(year_to)s" in w.sql
    assert w.params == {"year_from": 2015, "year_to": 2020}
    assert w.needs_judgments


def test_judge_is_normalised(omap):
    w = pg.build_filter_sql(Filters(judge="Abhay S. Oka, J."), omap)
    assert "j.judges_norm @> ARRAY[%(judge)s]::text[]" in w.sql
    assert w.params["judge"] == "ABHAY S OKA"


def test_act_uses_exists_on_refs_and_normaliser(omap):
    w = pg.build_filter_sql(Filters(act="  Indian  Penal Code, 1860 "), omap)
    assert "EXISTS (SELECT 1 FROM refs r" in w.sql
    assert "r.kind = 'act'" in w.sql
    assert w.params["act"] == "indian penal code, 1860"
    assert not w.needs_judgments
    w2 = pg.build_filter_sql(Filters(act="IPC"), omap, act_normalizer=lambda s: "penal code, 1860")
    assert w2.params["act"] == "penal code, 1860"


def test_blank_string_filters_are_ignored(omap):
    w = pg.build_filter_sql(Filters(judge=" ", act="", outcome="  "), omap)
    assert w.sql == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("allowed", "allowed"),
        ("Dismissed", "dismissed"),
        ("other", "other"),
        ("Appeal(s) allowed", "allowed"),
        ("leave granted & disposed off", "disposed"),
    ],
)
def test_outcome_values(omap, value, expected):
    w = pg.build_filter_sql(Filters(outcome=value), omap)
    assert "j.outcome = %(outcome)s" in w.sql
    assert w.params["outcome"] == expected


def test_unknown_outcome_rejected(omap):
    with pytest.raises(ValueError, match="unknown outcome"):
        pg.build_filter_sql(Filters(outcome="won"), omap)


def test_filter_values_are_never_spliced(omap):
    evil = "x'); DROP TABLE chunks; --"
    w = pg.build_filter_sql(Filters(judge=evil, act=evil), omap)
    assert "DROP" not in w.sql


# --- keyword ---------------------------------------------------------------------


def test_keyword_terms_words_and_phrases():
    words, phrases = pg.keyword_terms('Is "anticipatory bail" barred under Section 438?')
    assert words == ["is", "anticipatory", "bail", "barred", "under", "section", "438"]
    assert phrases == ["anticipatory bail"]


def test_keyword_terms_curly_quotes_and_punctuation():
    words, phrases = pg.keyword_terms("“right to privacy” & art. 21 | !x:*")
    assert phrases == ["right to privacy"]
    assert words == ["right", "to", "privacy", "art", "21", "x"]
    assert all(w.isalnum() for w in words)


def test_keyword_terms_dedupe():
    assert pg.keyword_terms("bail bail BAIL")[0] == ["bail"]


def test_build_tsquery():
    tsq, params = pg.build_tsquery(["bail", "438"], ["anticipatory bail"])
    assert tsq == (
        "(to_tsquery('english', %(or_words)s) || phraseto_tsquery('english', %(phrase_0)s))"
    )
    assert params == {"or_words": "bail | 438", "phrase_0": "anticipatory bail"}
    with pytest.raises(ValueError):
        pg.build_tsquery([], [])


def test_build_keyword_sql(omap):
    w = pg.build_filter_sql(Filters(year_from=2020), omap)
    sql, params = pg.build_keyword_sql("bail", w, "v1", 50)
    assert "ts_rank_cd(c.tsv, q.query)" in sql
    assert "c.tsv @@ q.query" in sql
    assert "JOIN judgments j" in sql
    assert "ORDER BY score DESC, c.chunk_id" in sql
    assert params["index_version"] == "v1" and params["k"] == 50
    assert params["year_from"] == 2020


def test_build_keyword_sql_without_judgment_filters_skips_join(omap):
    sql, _ = pg.build_keyword_sql("bail", pg.build_filter_sql(None, omap), "v1", 5)
    assert "JOIN judgments" not in sql


def test_build_keyword_sql_nothing_searchable(omap):
    assert pg.build_keyword_sql("?? !!", pg.build_filter_sql(None, omap), "v1", 5) is None


# --- vector ----------------------------------------------------------------------


def test_build_vector_sql(omap):
    w = pg.build_filter_sql(Filters(judge="Oka"), omap)
    sql, params = pg.build_vector_sql(w, "v2", 50)
    assert "v.embedding <=> %(qvec)s" in sql
    assert "AS MATERIALIZED" in sql
    assert "JOIN judgments j" in sql
    assert sql.rstrip().endswith("ORDER BY distance, chunk_id")
    assert params["index_version"] == "v2" and params["k"] == 50 and "judge" in params


def test_vector_settings_hnsw():
    sql, params = pg.vector_settings_sql(100, exact=False)
    assert "set_config('hnsw.ef_search', %(ef)s, true)" in sql
    assert "set_config('hnsw.iterative_scan', 'relaxed_order', true)" in sql
    assert params == {"ef": "100"}


def test_vector_settings_exact():
    sql, params = pg.vector_settings_sql(100, exact=True)
    assert "set_config('enable_indexscan', 'off', true)" in sql
    assert params == {}


# --- schema and rows -------------------------------------------------------------


@pytest.mark.parametrize("name", ["g4_step_1", "public", "_x"])
def test_valid_schema(name):
    assert pg.validate_schema(name) == name


@pytest.mark.parametrize("name", ["G4", "1abc", "a;b", "", "a b", "x" * 64])
def test_invalid_schema(name):
    with pytest.raises(ValueError):
        pg.validate_schema(name)


def test_row_to_chunk():
    row = (
        "p#judgment#001",
        "p",
        "judgment",
        None,
        "majority",
        ["1", "2"],
        None,
        None,
        "text",
        "Header",
        0,
        4,
        [3],
        None,
        "ct1",
        "ch1",
    )
    c = pg.row_to_chunk(row)
    assert c.zone is Zone.JUDGMENT
    assert c.para_refs == [] and c.flags == [] and c.pages == [3]


def test_row_to_judgment_fills_nulls():
    row = (
        "p",
        None,
        "A",
        "B",
        None,
        None,
        "[2025] 1 S.C.R. 230",
        None,
        "ESCR010004242025",
        None,
        "other",
        date(2025, 1, 2),
        2025,
        "d",
        True,
        9,
        "clean",
        [[0, 230], [10, 231]],
    )
    j = pg.row_to_judgment(row)
    assert j.title == "" and j.disposal == "" and j.judges == []
    assert j.position_map == [(0, 230), (10, 231)]


# --- engine with a fake connection --------------------------------------------------


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeTx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, responses):
        self.responses = list(responses)
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self.responses.pop(0) if self.responses else [])

    def transaction(self):
        return FakeTx()


def test_engine_reads_active_version(settings):
    conn = FakeConn([[("v9",)]])
    eng = pg.PostgresSearch(settings=settings, conn=conn)
    assert eng.index_version == "v9"
    assert "status = 'active'" in conn.executed[0][0]


def test_engine_without_active_version(settings):
    with pytest.raises(pg.NoActiveVersion):
        pg.PostgresSearch(settings=settings, conn=FakeConn([[]]))


def test_engine_schema_sets_search_path(settings):
    conn = FakeConn([])
    pg.PostgresSearch(index_version="g4_step_1", schema="g4_step_1", settings=settings, conn=conn)
    assert conn.executed[0][1] == ("g4_step_1, public",)


def test_engine_keyword_and_vector(settings):
    conn = FakeConn([[("a", 0.5), ("b", 0.1)], [], [("b", 0.1), ("c", 0.2)]])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn)
    kw = eng.keyword("bail", None, 2)
    assert [(h.chunk_id, h.keyword_rank, h.fused_rank) for h in kw] == [
        ("a", 1, 1.0),
        ("b", 2, 2.0),
    ]
    vec = eng.vector([0.0] * 4, None, 2)
    assert [(h.chunk_id, h.vector_rank) for h in vec] == [("b", 1), ("c", 2)]
    assert "hnsw.ef_search" in conn.executed[1][0]
    assert conn.executed[1][1] == {"ef": str(settings.retrieval.ef_search)}
    assert eng.keyword("?!", None, 2) == []


def test_engine_lookups(settings):
    conn = FakeConn(
        [
            [("2025_1_230_238",)],
            [],
            [("h", "c1"), ("h", "c2"), ("g", "c3")],
            [(0.35, 0.2)],
            [("p#editorial#001",), ("p#issue#001",)],
        ]
    )
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn)
    assert eng.resolve_alias("insc:2025:31") == "2025_1_230_238"
    assert eng.resolve_alias("insc:2025:99") is None
    assert eng.links_from(["h", "g", "x"]) == {"h": ["c1", "c2"], "g": ["c3"]}
    assert eng.thresholds() == {"qwen": 0.35, "cohere": 0.2}
    assert eng.chunk_ids_for("p") == ["p#editorial#001", "p#issue#001"]
    assert eng.links_from([]) == {} and eng.get_chunks([]) == []
    assert len(conn.executed) == 5


def test_engine_get_chunks_keeps_order(settings):
    def row(cid):
        return (cid, "p", "judgment", None, None, [], [], None, "t", "h", 0, 1, [], [], "c", "c")

    conn = FakeConn([[row("p#judgment#002"), row("p#judgment#001")]])
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=conn)
    got = eng.get_chunks(["p#judgment#001", "missing", "p#judgment#002"])
    assert [c.chunk_id for c in got] == ["p#judgment#001", "p#judgment#002"]


def test_engine_get_judgment_missing(settings):
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=FakeConn([[]]))
    assert eng.get_judgment("nope") is None


def test_engine_thresholds_missing_version(settings):
    eng = pg.PostgresSearch(index_version="v1", settings=settings, conn=FakeConn([[]]))
    with pytest.raises(LookupError):
        eng.thresholds()
