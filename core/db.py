"""SQLite/Postgres-Schicht von WortRadar.

Lokal (kein DATABASE_URL): SQLite in data/wortradar.db, user_id = 'local'.
Online  (DATABASE_URL gesetzt): Postgres via psycopg2, user_id = Supabase-UUID.

Nutzertrennung: set_current_user() direkt nach dem Login aufrufen.
Alle nutzerspezifischen Tabellen filtern automatisch nach der gesetzten user_id.
"""
from __future__ import annotations

import contextvars
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# ─── Konfiguration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "wortradar.db"

# Lazy: DATABASE_URL kann durch Streamlit-Secrets erst nach dem Import gesetzt
# werden; deshalb nie als Modul-Konstante cachen.
def _pg_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()

def _is_pg() -> bool:
    return bool(_pg_url())


# Thread-sicheres user_id-Handling (Streamlit: jede Session = eigener Thread)
_user_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_user_ctx", default="local"
)


def set_current_user(uid: str | None) -> None:
    """Nach dem Login mit der Supabase-user-id aufrufen.
    Ohne Login (lokal): uid=None -> 'local'."""
    _user_ctx.set(uid if uid else "local")


def _uid() -> str:
    return _user_ctx.get()


# ─── Verbindungsabstraktion ───────────────────────────────────────────────────
class _Conn:
    """Einheitliche Schnittstelle ueber SQLite-Connection und psycopg2-Connection."""

    def __init__(self, raw, pg: bool) -> None:
        self._r = raw
        self._pg = pg

    def _q(self, sql: str) -> str:
        """SQLite-Platzhalter ? -> Postgres %s konvertieren."""
        return sql.replace("?", "%s") if self._pg else sql

    def execute(self, sql: str, params=()) -> "_Cur":
        if self._pg:
            import psycopg2.extras  # noqa: PLC0415
            cur = self._r.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(self._q(sql), params if params else None)
            return _Cur(cur, pg=True)
        return _Cur(self._r.execute(sql, params), pg=False)

    def executemany(self, sql: str, params_list) -> "_Cur":
        if self._pg:
            import psycopg2.extras  # noqa: PLC0415
            cur = self._r.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            psycopg2.extras.execute_batch(cur, self._q(sql), params_list)
            return _Cur(cur, pg=True)
        return _Cur(self._r.executemany(sql, params_list), pg=False)

    def init_schema(self, sql: str) -> None:
        if self._pg:
            cur = self._r.cursor()
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        else:
            self._r.executescript(sql)


class _Cur:
    def __init__(self, raw, pg: bool) -> None:
        self._r = raw
        self._pg = pg

    def fetchone(self):
        row = self._r.fetchone()
        if row is None:
            return None
        return dict(row) if self._pg else row

    def fetchall(self):
        rows = self._r.fetchall()
        return [dict(r) for r in rows] if self._pg else rows

    @property
    def lastrowid(self) -> int:
        return self._r.lastrowid  # nur SQLite; Postgres nutzt RETURNING id


@contextmanager
def get_conn():
    if _is_pg():
        import psycopg2  # noqa: PLC0415
        conn = psycopg2.connect(_pg_url())
        try:
            yield _Conn(conn, pg=True)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield _Conn(conn, pg=False)
            conn.commit()
        finally:
            conn.close()


# ─── Schema ───────────────────────────────────────────────────────────────────
# user_id DEFAULT 'local' erlaubt lokalen Betrieb ohne Login-Overhead.
# In Postgres wird user_id per Supabase-UUID gesetzt.

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT 'local',
    title       TEXT NOT NULL,
    doc_type    TEXT NOT NULL DEFAULT 'text',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);

CREATE TABLE IF NOT EXISTS lines (
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    line_no  INTEGER NOT NULL,
    text     TEXT NOT NULL,
    t_start  REAL,
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
    pos      INTEGER NOT NULL,
    surface  TEXT NOT NULL,
    c0       INTEGER NOT NULL DEFAULT 0,
    c1       INTEGER NOT NULL DEFAULT 0,
    lemma_id INTEGER REFERENCES lemmas(id),
    kind     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tokens_doc   ON tokens(doc_id, line_no, pos);
CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma_id);

CREATE TABLE IF NOT EXISTS knowledge (
    user_id    TEXT NOT NULL DEFAULT 'local',
    lemma_id   INTEGER NOT NULL REFERENCES lemmas(id),
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
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
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
"""

_SCHEMA_PG = """
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
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    schema = _SCHEMA_PG if _is_pg() else _SCHEMA_SQLITE
    with get_conn() as c:
        c.init_schema(schema)


# ─── meta ─────────────────────────────────────────────────────────────────────
def meta_get(key: str, default=None):
    with get_conn() as c:
        row = c.execute(
            "SELECT value FROM meta WHERE user_id=? AND key=?", (_uid(), key)
        ).fetchone()
    return row["value"] if row else default


def meta_set(key: str, value) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO meta(user_id, key, value) VALUES(?,?,?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value",
            (_uid(), key, str(value)),
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


# ─── documents ────────────────────────────────────────────────────────────────
def create_document(title: str, doc_type: str) -> int:
    sql = "INSERT INTO documents(user_id, title, doc_type, created_at) VALUES(?,?,?,?)"
    params = (_uid(), title, doc_type, now_iso())
    with get_conn() as c:
        if _is_pg():
            cur = c.execute(sql + " RETURNING id", params)
            return cur.fetchone()["id"]
        cur = c.execute(sql, params)
        return cur.lastrowid


def delete_document(doc_id: int) -> None:
    with get_conn() as c:
        c.execute("DELETE FROM tokens WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM lines WHERE doc_id=?", (doc_id,))
        c.execute(
            "DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, _uid())
        )


def list_documents() -> list:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM documents WHERE user_id=? ORDER BY created_at DESC",
            (_uid(),),
        ).fetchall()


def get_document(doc_id: int):
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, _uid())
        ).fetchone()


# ─── lines ────────────────────────────────────────────────────────────────────
def insert_lines(doc_id: int, lines: list[tuple[int, str, float | None]]) -> None:
    with get_conn() as c:
        c.executemany(
            "INSERT INTO lines(doc_id, line_no, text, t_start) VALUES(?,?,?,?)",
            [(doc_id, n, t, ts) for (n, t, ts) in lines],
        )


def get_lines(doc_id: int) -> list:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM lines WHERE doc_id=? ORDER BY line_no", (doc_id,)
        ).fetchall()


# ─── lemmas / tokens ──────────────────────────────────────────────────────────
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


def doc_token_rows(doc_id: int) -> list:
    with get_conn() as c:
        return c.execute(
            """SELECT t.line_no, t.pos, t.surface, t.c0, t.c1, t.kind, t.lemma_id,
                      l.lemma, l.zipf, l.is_function,
                      k.status, k.due
               FROM tokens t
               LEFT JOIN lemmas l    ON l.id = t.lemma_id
               LEFT JOIN knowledge k ON k.lemma_id = t.lemma_id AND k.user_id = ?
               WHERE t.doc_id = ?
               ORDER BY t.line_no, t.pos""",
            (_uid(), doc_id),
        ).fetchall()


def doc_kind_counts(doc_id: int) -> dict[str, int]:
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


def doc_lemma_summary(doc_id: int) -> list:
    """Alle Inhaltslemmata eines Dokuments mit Haeufigkeit, Status, Uebersetzung."""
    with get_conn() as c:
        return c.execute(
            """SELECT l.id AS lemma_id, l.lemma, l.zipf,
                      COUNT(*) AS cnt,
                      MIN(t.line_no) AS first_line,
                      k.status, k.fsrs, k.due,
                      d.de AS translation
               FROM tokens t
               JOIN lemmas l         ON l.id = t.lemma_id
               LEFT JOIN knowledge k ON k.lemma_id = l.id AND k.user_id = ?
               LEFT JOIN dictionary d ON d.lemma = l.lemma
               WHERE t.doc_id = ? AND t.kind = 'content'
               GROUP BY l.id, l.lemma, l.zipf, k.status, k.fsrs, k.due, d.de
               ORDER BY cnt DESC, l.zipf DESC""",
            (_uid(), doc_id),
        ).fetchall()


def doc_marked_function_lemmas(doc_id: int) -> list:
    """Funktions- und Eigennamen-Woerter, die der Nutzer explizit markiert hat."""
    with get_conn() as c:
        return c.execute(
            """SELECT l.id AS lemma_id, l.lemma, l.zipf,
                      COUNT(*) AS cnt,
                      MIN(t.line_no) AS first_line,
                      k.status, k.fsrs, k.due,
                      d.de AS translation
               FROM tokens t
               JOIN lemmas l         ON l.id = t.lemma_id
               JOIN knowledge k      ON k.lemma_id = l.id AND k.user_id = ?
               LEFT JOIN dictionary d ON d.lemma = l.lemma
               WHERE t.doc_id = ? AND t.kind IN ('function', 'proper')
                     AND k.status IN ('unknown', 'learning')
               GROUP BY l.id, l.lemma, l.zipf, k.status, k.fsrs, k.due, d.de
               ORDER BY cnt DESC, l.zipf DESC""",
            (_uid(), doc_id),
        ).fetchall()


def all_doc_content_counts() -> list:
    """Fuer Unlocks: (doc_id, lemma_id, lemma, zipf, cnt, status) - nur eigene Docs."""
    uid = _uid()
    with get_conn() as c:
        return c.execute(
            """SELECT t.doc_id, l.id AS lemma_id, l.lemma, l.zipf,
                      COUNT(*) AS cnt, k.status
               FROM tokens t
               JOIN lemmas l         ON l.id = t.lemma_id
               JOIN documents doc    ON doc.id = t.doc_id AND doc.user_id = ?
               LEFT JOIN knowledge k ON k.lemma_id = l.id AND k.user_id = ?
               WHERE t.kind = 'content'
               GROUP BY t.doc_id, l.id, l.lemma, l.zipf, k.status""",
            (uid, uid),
        ).fetchall()


# ─── knowledge ────────────────────────────────────────────────────────────────
def set_status(lemma_id: int, status: str | None) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO knowledge(user_id, lemma_id, status, updated_at) "
            "VALUES(?,?,?,?) "
            "ON CONFLICT(user_id, lemma_id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at",
            (_uid(), lemma_id, status, now_iso()),
        )


def set_status_bulk(lemma_ids: list[int], status: str) -> None:
    ts = now_iso()
    uid = _uid()
    with get_conn() as c:
        c.executemany(
            "INSERT INTO knowledge(user_id, lemma_id, status, updated_at) "
            "VALUES(?,?,?,?) "
            "ON CONFLICT(user_id, lemma_id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at",
            [(uid, lid, status, ts) for lid in lemma_ids],
        )


def save_card(lemma_id: int, fsrs_json: str, due_iso: str, status: str) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO knowledge(user_id, lemma_id, status, fsrs, due, updated_at) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(user_id, lemma_id) DO UPDATE SET "
            "status=excluded.status, fsrs=excluded.fsrs, "
            "due=excluded.due, updated_at=excluded.updated_at",
            (_uid(), lemma_id, status, fsrs_json, due_iso, now_iso()),
        )


def get_knowledge(lemma_id: int):
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM knowledge WHERE user_id=? AND lemma_id=?",
            (_uid(), lemma_id),
        ).fetchone()


def due_cards(doc_id: int | None = None, include_new: bool = True) -> list:
    """Faellige Lernkarten, optional auf ein Dokument beschraenkt."""
    now = now_iso()
    uid = _uid()
    # (k.due IS NULL) als Sortier-Spalte: bei SELECT DISTINCT muss jede
    # ORDER-BY-Expression in der SELECT-Liste stehen (Postgres-Pflicht).
    # 0/false (faellig) vor 1/true (neu) -> in SQLite und Postgres gleich.
    base = """SELECT DISTINCT l.id AS lemma_id, l.lemma, l.zipf,
                     k.status, k.fsrs, k.due, d.de AS translation,
                     (k.due IS NULL) AS due_is_null
              FROM lemmas l
              JOIN knowledge k       ON k.lemma_id = l.id AND k.user_id = ?
              LEFT JOIN dictionary d ON d.lemma = l.lemma
              {join}
              WHERE ( (k.status='learning' AND k.due IS NOT NULL AND k.due <= ?)
                      {newpart} )
              {docfilter}
              ORDER BY due_is_null, k.due"""
    newpart = "OR (k.status='unknown')" if include_new else ""
    join, docfilter, params = "", "", [uid, now]
    if doc_id is not None:
        join = "JOIN tokens t ON t.lemma_id = l.id"
        docfilter = "AND t.doc_id = ?"
        params.append(doc_id)
    with get_conn() as c:
        return c.execute(base.format(join=join, newpart=newpart, docfilter=docfilter),
                         params).fetchall()


def due_counts_by_doc() -> dict[int, int]:
    """Anzahl faelliger/neuer Karten je Dokument (nur eigene Docs)."""
    now = now_iso()
    uid = _uid()
    with get_conn() as c:
        rows = c.execute(
            """SELECT t.doc_id AS doc_id, COUNT(DISTINCT l.id) AS n
               FROM lemmas l
               JOIN knowledge k   ON k.lemma_id = l.id AND k.user_id = ?
               JOIN tokens t      ON t.lemma_id = l.id
               JOIN documents doc ON doc.id = t.doc_id AND doc.user_id = ?
               WHERE (k.status='learning' AND k.due IS NOT NULL AND k.due <= ?)
                     OR (k.status='unknown')
               GROUP BY t.doc_id""",
            (uid, uid, now),
        ).fetchall()
    return {r["doc_id"]: r["n"] for r in rows}


def knowledge_stats() -> dict:
    with get_conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) AS n FROM knowledge "
            "WHERE user_id=? AND status IS NOT NULL GROUP BY status",
            (_uid(),),
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


# ─── dictionary ───────────────────────────────────────────────────────────────
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
                    "ON CONFLICT(lemma) DO UPDATE SET de=excluded.de, "
                    "source=excluded.source",
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


# ─── test_answers ─────────────────────────────────────────────────────────────
def save_test_answer(
    session: str, word: str, zipf: float, pseudo: bool, answer: bool
):
    with get_conn() as c:
        c.execute(
            "INSERT INTO test_answers(user_id, session, word, zipf, pseudo, answer) "
            "VALUES(?,?,?,?,?,?)",
            (_uid(), session, word, zipf, int(pseudo), int(answer)),
        )


def get_test_answers(session: str) -> list:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM test_answers WHERE user_id=? AND session=?",
            (_uid(), session),
        ).fetchall()
