"""Spaced-Repetition-Schicht (FSRS v6) von WortRadar.

FSRS ist der moderne Nachfolger des klassischen Anki-SM-2-Algorithmus:
er modelliert Stabilitaet und Schwierigkeit jeder Karte und plant die
naechste Wiederholung kurz bevor du das Wort vergessen wuerdest.

Eine Vokabel = genau EINE Karte, egal in wie vielen Songs/Buechern sie
vorkommt. Wer "rain" im Song gelernt hat, muss es im Buch nicht nochmal
lernen - die Coverage aller Dokumente profitiert sofort mit.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fsrs import Card, Rating, Scheduler

from . import db

# Ab dieser Stabilitaet (Tage) gilt ein Wort als dauerhaft gekonnt
GRADUATE_STABILITY_DAYS = 21.0

_scheduler = Scheduler()

RATING_LABELS = [
    ("Nochmal", Rating.Again),
    ("Schwer", Rating.Hard),
    ("Gut", Rating.Good),
    ("Einfach", Rating.Easy),
]


def _load_card(fsrs_json: str | None) -> Card:
    if fsrs_json:
        try:
            return Card.from_dict(json.loads(fsrs_json))
        except Exception:
            pass
    return Card()


def review(lemma_id: int, fsrs_json: str | None, rating: Rating) -> dict:
    """Verarbeitet eine Bewertung, speichert die Karte, liefert Infos zurueck."""
    card = _load_card(fsrs_json)
    card, _log = _scheduler.review_card(card, rating, datetime.now(timezone.utc))
    graduated = bool(card.stability and card.stability >= GRADUATE_STABILITY_DAYS)
    status = "known" if graduated else "learning"
    db.save_card(lemma_id, json.dumps(card.to_dict()), card.due.isoformat(), status)
    return {
        "due": card.due,
        "stability": card.stability,
        "graduated": graduated,
    }


def start_learning(lemma_ids: list[int]) -> None:
    """Nimmt Woerter ohne Karte neu ins Lernen auf (sofort faellig)."""
    now = datetime.now(timezone.utc).isoformat()
    for lid in lemma_ids:
        row = db.get_knowledge(lid)
        if row and row["fsrs"]:
            continue  # Karte existiert schon
        card = Card()
        db.save_card(lid, json.dumps(card.to_dict()), now, "learning")
