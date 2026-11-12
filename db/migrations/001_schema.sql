-- 001_schema.sql: the Nyaya Search schema (design file 12).
-- Every per-version table carries index_version and cascades from
-- index_versions, so deleting a version removes all of its rows.
-- Changing the embedding dimension needs a new migration and a full rebuild.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE index_versions (
    id                 text PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    status             text NOT NULL
                       CHECK (status IN ('building', 'ready', 'active', 'retired')),
    clean_text_version text NOT NULL,
    chunker_version    text NOT NULL,
    config_version     text NOT NULL,
    embed_model        text NOT NULL,
    embed_revision     text,
    reranker_model     text,
    -- read by the refusal gate; seeded from config at load, set by tune-threshold
    threshold_qwen     double precision,
    threshold_cohere   double precision,
    snapshot_date      date,
    notes              text
);

CREATE TABLE judgments (
    index_version    text NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    path             text NOT NULL,
    title            text,
    petitioner       text,
    respondent       text,
    judges           text[] NOT NULL DEFAULT '{}',
    judges_norm      text[] NOT NULL DEFAULT '{}',
    author_judge     text,
    scr_citation     text,
    neutral_citation text,
    cnr              text,
    disposal         text,
    outcome          text,
    decision_date    date,
    decision_year    smallint,
    -- era, clean_text and position_map are null for rows without a PDF
    era              text CHECK (era IN ('c', 'd', 'out_of_scope')),
    has_pdf          boolean NOT NULL,
    tar_file         text,
    offset_data      bigint,
    byte_size        bigint,
    pages            integer,
    clean_text       text,
    position_map     jsonb,
    PRIMARY KEY (index_version, path)
);

CREATE TABLE chunks (
    index_version      text NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    chunk_id           text NOT NULL,
    judgment_id        text NOT NULL,
    zone               text NOT NULL CHECK (zone IN ('issue', 'editorial', 'judgment')),
    opinion_author     text,
    opinion_type       text
                       CHECK (opinion_type IN ('majority', 'concurring', 'dissenting', 'unknown')),
    para_labels        text[] NOT NULL DEFAULT '{}',
    para_refs          text[] NOT NULL DEFAULT '{}',
    section_label      text,
    text               text NOT NULL,
    header             text NOT NULL,
    char_start         integer NOT NULL,
    char_end           integer NOT NULL,
    pages              integer[] NOT NULL DEFAULT '{}',
    flags              text[] NOT NULL DEFAULT '{}',
    clean_text_version text NOT NULL,
    chunker_version    text NOT NULL,
    tsv                tsvector
                       GENERATED ALWAYS AS (to_tsvector('english', header || ' ' || text)) STORED,
    PRIMARY KEY (index_version, chunk_id)
);

CREATE TABLE chunk_vectors (
    index_version text NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    chunk_id      text NOT NULL,
    embedding     vector(1024) NOT NULL,
    PRIMARY KEY (index_version, chunk_id)
);

CREATE TABLE aliases (
    index_version text NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    alias_norm    text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('scr', 'neutral', 'cnr')),
    path          text NOT NULL
);

-- Held links only: an editorial chunk to the judgment chunks it names.
CREATE TABLE links (
    index_version text NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    from_chunk_id text NOT NULL,
    to_chunk_id   text NOT NULL,
    PRIMARY KEY (index_version, from_chunk_id, to_chunk_id)
);

CREATE TABLE refs (
    index_version text NOT NULL REFERENCES index_versions (id) ON DELETE CASCADE,
    path          text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('act', 'keyword')),
    value         text NOT NULL,
    value_norm    text NOT NULL
);

CREATE TABLE traces (
    request_id             text PRIMARY KEY,
    created_at             timestamptz NOT NULL DEFAULT now(),
    question               text,
    route                  text,
    filters                jsonb,
    versions               jsonb,
    retrieved              jsonb,
    rows_returned          jsonb,
    top_score              double precision,
    threshold              double precision,
    status                 text,
    rule_breaks            jsonb,
    unverified_count       integer,
    gaps_count             integer,
    reranker_fallback_used boolean,
    llm_error              text,
    timings_ms             jsonb
);

CREATE TABLE eval_runs (
    run_id          text PRIMARY KEY,
    created_at      timestamptz NOT NULL DEFAULT now(),
    dataset_version text,
    index_version   text,
    schema_name     text,
    config          jsonb,
    metrics         jsonb,
    CHECK (index_version IS NOT NULL OR schema_name IS NOT NULL)
);

CREATE INDEX chunk_vectors_embedding_hnsw
    ON chunk_vectors USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX judgments_year ON judgments (index_version, decision_year);
CREATE INDEX judgments_scr ON judgments (index_version, scr_citation);
CREATE INDEX judgments_neutral ON judgments (index_version, neutral_citation);
CREATE INDEX judgments_judges_norm_gin ON judgments USING gin (judges_norm);
CREATE INDEX aliases_lookup ON aliases (index_version, alias_norm);
CREATE INDEX refs_lookup ON refs (index_version, kind, value_norm);
