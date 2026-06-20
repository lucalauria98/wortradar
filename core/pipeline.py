"""Ingest-Pipeline: Import-Zeilen -> NLP -> Datenbank."""
from __future__ import annotations

from . import db
from .importers import Line
from .nlp import analyze_lines


def ingest_document(title: str, doc_type: str, lines: list[Line]) -> int:
    """Legt ein Dokument an, analysiert alle Zeilen, speichert Tokens.

    Liefert die doc_id. Eine Vokabel, die es schon gibt (aus anderen
    Dokumenten), wird wiederverwendet - inklusive ihres Lernstatus.
    """
    lines = [(n, t, ts) for (n, t, ts) in lines if t.strip()]
    if not lines:
        raise ValueError("Der Text enthaelt keine verwertbaren Zeilen.")

    doc_id = db.create_document(title.strip() or "Ohne Titel", doc_type)
    db.insert_lines(doc_id, lines)

    analyzed = analyze_lines([t for (_, t, _) in lines])

    # 1) alle Lemmata einsammeln und anlegen - AUCH Eigennamen, damit JEDES
    #    Wort eine lemma_id hat und im Text anklick-/markierbar ist. (Eigennamen
    #    bleiben kind="proper" und zaehlen weiter NICHT gegen die Coverage,
    #    weil doc_lemma_summary nur kind='content' liest.)
    lemma_entries: dict[str, tuple[str, float, int]] = {}
    for toks in analyzed:
        for t in toks:
            is_fn = 1 if t.kind == "function" else 0
            prev = lemma_entries.get(t.lemma)
            if prev is None or (prev[2] == 1 and is_fn == 0):
                lemma_entries[t.lemma] = (t.lemma, t.zipf, is_fn)
    lemma_ids = db.upsert_lemmas(list(lemma_entries.values()))

    # 2) Tokens mit Positionen speichern
    rows = []
    for (line_no, _, _), toks in zip(lines, analyzed):
        for pos, t in enumerate(toks):
            lid = lemma_ids.get(t.lemma)
            rows.append((doc_id, line_no, pos, t.surface, t.c0, t.c1, lid, t.kind))
    db.insert_tokens(rows)
    return doc_id
