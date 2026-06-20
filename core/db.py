"""SQLite-Schicht von WortRadar.

Eine Datei, null Server: data/wortradar.db neben dem Projekt.
Alle Funktionen oeffnen kurzlebige Verbindungen (robust unter Streamlit).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "wortradar.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    doc_type    TEXT NOT NULL DEFAULT 'text',   -- song | buch | blog | film | text
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lines (
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    line_no  INTEGER NOT NULL,
    text     TEXT NOT NULL,
    t_start  REAL,                               -- Sekunden (aus .lrc/.srt), sonst NULL
    PRIMARY KEY (doc_id, line_no)
);

CREATE TABLE IF NOT EXISTS lemmas (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma        TEXT NOT NULL UNIQUE,
    zipf         REAL NOT NULL DEFAULT 0,
    is_function  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tokens (
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    line_no  INTEGER NOT NULL,
    pos      INTEGER NOT NULL,                   -- Position innerhalb der Zeile
    surface  TEXT NOT NULL,
    c0       INTEGER NOT NULL DEFAULT 0,        -- Zeichen-Offset Beginn in der Zeile
    c1       INTEGER NOT NULL DEFAULT 0,        -- Zeichen-Offset Ende
    lemma_id INTEGER REFERENCES lemmas(id),
    kind     TEXT NOT NULL                       -- content | function | proper | other
);
CREATE INDEX IF NOT EXISTS idx_tokens_doc   ON tokens(doc_id, line_no, pos);
CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma_id);

CREATE TABLE IF NOT EXISTS knowledge (
    lemma_id   INTEGER PRIMARY KEY REFERENCES lemmas(id),
    status     TEXT,                             -- known | unknown | learning | ignored | NULL (= nur Prior)
    fsrs       TEXT,                             -- serialisierte FSRS-Karte (JSON)
    due        TEXT,                             -- ISO-Zeitpunkt der naechsten Wiederholung
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS dictionary (
    lemma  TEXT PRIMARY KEY,
    de     TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manuell'       -- ding | dictcc | llm | manuell
);

CREATE TABLE IF NOT EXISTS test_answers (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    word    TEXT NOT NULL,
    zipf    REAL NOT NULL,
    pseudo  INTEGER NOT NULL,
    answer  INTEGER NOT NULL                     -- 1 = "kenne ich", 0 = "kenne ich nicht"
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------- meta ----
def meta_get(key: str, default=None):
    with get_conn() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def meta_set(key: str, value) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )


def meta_get_json(key: str, default=None):
    raw = meta_get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def meta_set_json(key: str, obj) -> None:
    meta_set(key, json.dumps(obj, ensure_ascii=False))


# ----------------------------------------------------------- documents ----
def create_document(title: str, doc_type: str) -> int:
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO documents(title, doc_type, created_at) VALUES(?,?,?)",
            (title, doc_type, now_iso()),
        )
        return cur.lastrowid


def delete_document(doc_id: int) -> None:
    with get_conn() as c:
        c.execute("DELETE FROM tokens WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM lines WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def list_documents() -> list[sqlite3.Row]:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()


def get_document(doc_id: int):
    with get_conn() as c:
        return c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()


def insert_lines(doc_id: int, lines: list[tuple[int, str, float | None]]) -> None:
    with get_conn() as c:
        c.executemany(
            "INSERT INTO lines(doc_id, line_no, text, t_start) VALUES(?,?,?,?)",
            [(doc_id, n, t, ts) for (n, t, ts) in lines],
        )


def get_lines(doc_id: int) -> list[sqlite3.Row]:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM lines WHERE doc_id=? ORDER BY line_no", (doc_id,)
        ).fetchall()


# -------------------------------------------------------------- lemmas ----
def upsert_lemmas(entries: list[tuple[str, float, int]]) -> dict[str, int]:
    """entries: (lemma, zipf, is_function). Liefert lemma -> id."""
    if not entries:
        return {}
    with get_conn() as c:
        c.executemany(
            "INSERT INTO lemmas(lemma, zipf, is_function) VALUES(?,?,?) "
            "ON CONFLICT(lemma) DO NOTHING",
            entries,
        )
        marks = ",".join("?" * len(entries))
        rows = c.execute(
            f"SELECT id, lemma FROM lemmas WHERE lemma IN ({marks})",
            [e[0] for e in entries],
        ).fetchall()
    return {r["lemma"]: r["id"] for r in rows}


def insert_tokens(rows: list[tuple]) -> None:
    """rows: (doc_id, line_no, pos, surface, c0, c1, lemma_id, kind)"""
    with get_conn() as c:
        c.executemany(
            "INSERT INTO tokens(doc_id, line_no, pos, surface, c0, c1, lemma_id, kind) "
            "VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )


def doc_token_rows(doc_id: int) -> list[sqlite3.Row]:
    with get_conn() as c:
        return c.execute(
            """SELECT t.line_no, t.pos, t.surface, t.c0, t.c1, t.kind, t.lemma_id,
                      l.lemma, l.zipf, l.is_function,
                      k.status, k.due
               FROM tokens t
               LEFT JOIN lemmas l    ON l.id = t.lemma_id
               LEFT JOIN knowledge k ON k.lemma_id = t.lemma_id
               WHERE t.doc_id = ?
               ORDER BY t.line_no, t.pos""",
            (doc_id,),
        ).fetchall()


def doc_kind_counts(doc_id: int) -> dict[str, int]:
    """Token-Anzahl je Art (content/function/proper) - fuer schnelle Stats."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT kind, COUNT(*) AS n FROM tokens WHERE doc_id=? GROUP BY kind",
            (doc_id,),
        ).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def lemma_first_context(lemma_id: int, doc_id: int | None = None):
    """Erste Vorkommens-Zeile eines Lemmas (fuer Karten-Kontext): (text, surface)."""
    sql = """SELECT li.text, t.surface
             FROM tokens t
             JOIN lines li ON li.doc_id = t.doc_id AND li.line_no = t.line_no
             WHERE t.lemma_id = ? {flt}
             ORDER BY t.doc_id, t.line_no, t.pos LIMIT 1"""
    params: list = [lemma_id]
    flt = ""
    if doc_id is not None:
        flt = "AND t.doc_id = ?"
        params.append(doc_id)
    with get_conn() as c:
        return c.execute(sql.format(flt=flt), params).fetchone()


def doc_lemma_summary(doc_id: int) -> list[sqlite3.Row]:
    """Alle Inhaltslemmata eines Dokuments mit Haeufigkeit, Status, Uebersetzung."""
    with get_conn() as c:
        return c.execute(
            """SELECT l.id AS lemma_id, l.lemma, l.zipf,
                      COUNT(*) AS cnt,
                      MIN(t.line_no) AS first_line,
                      k.status, k.fsrs, k.due,
                      d.de AS translation
               FROM tokens t
               JOIN lemmas l        ON l.id = t.lemma_id
               LEFT JOIN knowledge k ON k.lemma_id = l.id
               LEFT JOIN dictionary d ON d.lemma = l.lemma
               WHERE t.doc_id = ? AND t.kind = 'content'
               GROUP BY l.id
               ORDER BY cnt DESC, l.zipf DESC""",
            (doc_id,),
        ).fetchall()


def doc_marked_function_lemmas(doc_id: int) -> list[sqlite3.Row]:
    """Funktions- UND Eigennamen-Woerter eines Dokuments, die der Nutzer
    EXPLIZIT markiert hat (status unknown/learning). Fuer das Deck - Coverage
    bleibt unberuehrt (Inhaltswoerter zaehlen ueber doc_lemma_summary, diese
    hier nicht). Spaltenform wie doc_lemma_summary()."""
    with get_conn() as c:
        return c.execute(
            """SELECT l.id AS lemma_id, l.lemma, l.zipf,
                      COUNT(*) AS cnt,
                      MIN(t.line_no) AS first_line,
                      k.status, k.fsrs, k.due,
                      d.de AS translation
               FROM tokens t
               JOIN lemmas l        ON l.id = t.lemma_id
               JOIN knowledge k     ON k.lemma_id = l.id
               LEFT JOIN dictionary d ON d.lemma = l.lemma
               WHERE t.doc_id = ? AND t.kind IN ('function', 'proper')
                     AND k.status IN ('unknown', 'learning')
               GROUP BY l.id
               ORDER BY cnt DESC, l.zipf DESC""",
            (doc_id,),
        ).fetchall()


def all_doc_content_counts() -> list[sqlite3.Row]:
    """Fuer Unlocks: (doc_id, lemma_id, lemma, zipf, cnt, status) ueber alle Docs."""
    with get_conn() as c:
        return c.execute(
            """SELECT t.doc_id, l.id AS lemma_id, l.lemma, l.zipf,
                      COUNT(*) AS cnt, k.status
               FROM tokens t
               JOIN lemmas l        ON l.id = t.lemma_id
               LEFT JOIN knowledge k ON k.lemma_id = l.id
               WHERE t.kind = 'content'
               GROUP BY t.doc_id, l.id""",
        ).fetchall()


# ----------------------------------------------------------- knowledge ----
def set_status(lemma_id: int, status: str | None) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO knowledge(lemma_id, status, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(lemma_id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at",
            (lemma_id, status, now_iso()),
        )


def set_status_bulk(lemma_ids: list[int], status: str) -> None:
    ts = now_iso()
    with get_conn() as c:
        c.executemany(
            "INSERT INTO knowledge(lemma_id, status, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(lemma_id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at",
            [(lid, status, ts) for lid in lemma_ids],
        )


def save_card(lemma_id: int, fsrs_json: str, due_iso: str, status: str) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO knowledge(lemma_id, status, fsrs, due, updated_at) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(lemma_id) DO UPDATE SET status=excluded.status, "
            "fsrs=excluded.fsrs, due=excluded.due, updated_at=excluded.updated_at",
            (lemma_id, status, fsrs_json, due_iso, now_iso()),
        )


def get_knowledge(lemma_id: int):
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM knowledge WHERE lemma_id=?", (lemma_id,)
        ).fetchone()


def due_cards(doc_id: int | None = None, include_new: bool = True) -> list[sqlite3.Row]:
    """Faellige Lernkarten, optional auf ein Dokument beschraenkt.

    'Neu' = status unknown ohne FSRS-Karte. 'Faellig' = learning mit due <= jetzt.
    """
    now = now_iso()
    base = """SELECT DISTINCT l.id AS lemma_id, l.lemma, l.zipf,
                     k.status, k.fsrs, k.due, d.de AS translation
              FROM lemmas l
              JOIN knowledge k       ON k.lemma_id = l.id
              LEFT JOIN dictionary d ON d.lemma = l.lemma
              {join}
              WHERE ( (k.status='learning' AND k.due IS NOT NULL AND k.due <= ?)
                      {newpart} )
              {docfilter}
              ORDER BY k.due IS NULL, k.due"""
    newpart = "OR (k.status='unknown')" if include_new else ""
    join, docfilter, params = "", "", [now]
    if doc_id is not None:
        join = "JOIN tokens t ON t.lemma_id = l.id"
        docfilter = "AND t.doc_id = ?"
        params.append(doc_id)
    sql = base.format(join=join, newpart=newpart, docfilter=docfilter)
    with get_conn() as c:
        return c.execute(sql, params).fetchall()


def due_counts_by_doc() -> dict[int, int]:
    """Anzahl faelliger/neuer Lernkarten je Dokument (fuer die Song-Auswahl
    in 'Heute lernen'). Logik wie due_cards(): learning&faellig oder unknown."""
    now = now_iso()
    with get_conn() as c:
        rows = c.execute(
            """SELECT t.doc_id AS doc_id, COUNT(DISTINCT l.id) AS n
               FROM lemmas l
               JOIN knowledge k ON k.lemma_id = l.id
               JOIN tokens t    ON t.lemma_id = l.id
               WHERE (k.status='learning' AND k.due IS NOT NULL AND k.due <= ?)
                     OR (k.status='unknown')
               GROUP BY t.doc_id""",
            (now,),
        ).fetchall()
    return {r["doc_id"]: r["n"] for r in rows}


def knowledge_stats() -> dict:
    with get_conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) AS n FROM knowledge "
            "WHERE status IS NOT NULL GROUP BY status"
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


# ----------------------------------------------------------- dictionary ----
def dict_lookup(lemmas: list[str]) -> dict[str, str]:
    if not lemmas:
        return {}
    out: dict[str, str] = {}
    with get_conn() as c:
        for i in range(0, len(lemmas), 500):
            chunk = lemmas[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = c.execute(
                f"SELECT lemma, de FROM dictionary WHERE lemma IN ({marks})", chunk
            ).fetchall()
            out.update({r["lemma"]: r["de"] for r in rows})
    return out


def dict_store(entries: dict[str, str], source: str) -> int:
    """Speichert Uebersetzungen; ueberschreibt vorhandene nicht (ausser manuell)."""
    n = 0
    with get_conn() as c:
        for lemma, de in entries.items():
            if not lemma or not de:
                continue
            if source == "manuell":
                c.execute(
                    "INSERT INTO dictionary(lemma, de, source) VALUES(?,?,?) "
                    "ON CONFLICT(lemma) DO UPDATE SET de=excluded.de, source=excluded.source",
                    (lemma, de, source),
                )
            else:
                c.execute(
                    "INSERT INTO dictionary(lemma, de, source) VALUES(?,?,?) "
                    "ON CONFLICT(lemma) DO NOTHING",
                    (lemma, de, source),
                )
            n += 1
    return n


def dict_size() -> int:
    with get_conn() as c:
        return c.execute("SELECT COUNT(*) AS n FROM dictionary").fetchone()["n"]


# --------------------------------------------------------- test answers ----
def save_test_answer(session: str, word: str, zipf: float, pseudo: bool, answer: bool):
    with get_conn() as c:
        c.execute(
            "INSERT INTO test_answers(session, word, zipf, pseudo, answer) "
            "VALUES(?,?,?,?,?)",
            (session, word, zipf, int(pseudo), int(answer)),
        )


def get_test_answers(session: str) -> list[sqlite3.Row]:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM test_answers WHERE session=?", (session,)
        ).fetchall()
