-- WortRadar – Supabase/Postgres Schema
-- Im Supabase SQL-Editor ausfuehren (einmalig beim Setup).
-- Alle Tabellen sind idempotent (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL DEFAULT 'local',
    title      TEXT NOT NULL,
    doc_type   TEXT NOT NULL DEFAULT 'text',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);

CREATE TABLE IF NOT EXISTS lines (
    doc_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    line_no  INTEGER NOT NULL,
    text     TEXT NOT NULL,
    t_start  REAL,
    PRIMARY KEY (doc_id, line_no)
);

CREATE TABLE IF NOT EXISTS lemmas (
    id          BIGSERIAL PRIMARY KEY,
    lemma       TEXT NOT NULL UNIQUE,
    zipf        REAL NOT NULL DEFAULT 0,
    is_function INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tokens (
    doc_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    line_no  INTEGER NOT NULL,
    pos      INTEGER NOT NULL,
    surface  TEXT NOT NULL,
    c0       INTEGER NOT NULL DEFAULT 0,
    c1       INTEGER NOT NULL DEFAULT 0,
    lemma_id BIGINT REFERENCES lemmas(id),
    kind     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tokens_doc   ON tokens(doc_id, line_no, pos);
CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma_id);

CREATE TABLE IF NOT EXISTS knowledge (
    user_id    TEXT NOT NULL DEFAULT 'local',
    lemma_id   BIGINT NOT NULL REFERENCES lemmas(id),
    status     TEXT,
    fsrs       TEXT,
    due        TEXT,
    updated_at TEXT,
    PRIMARY KEY (user_id, lemma_id)
);
CREATE INDEX IF NOT EXISTS idx_know_user ON knowledge(user_id);

CREATE TABLE IF NOT EXISTS dictionary (
    lemma  TEXT PRIMARY KEY,
    de     TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manuell'
);

CREATE TABLE IF NOT EXISTS test_answers (
    id      BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'local',
    session TEXT NOT NULL,
    word    TEXT NOT NULL,
    zipf    REAL NOT NULL,
    pseudo  INTEGER NOT NULL,
    answer  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    user_id TEXT NOT NULL DEFAULT 'local',
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (user_id, key)
);
