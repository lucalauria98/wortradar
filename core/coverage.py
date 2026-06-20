"""Auswertung: Coverage, Zeilen-Ampel, Quiz-Auswahl und Unlock-Mechanik.

Kernmetrik ist die ERWARTETE Coverage: der mittlere p_known-Wert ueber
alle zaehlbaren Tokens eines Dokuments. Forschungs-Daumenregeln
(Lexical-Coverage-Literatur, u. a. Paul Nation):
  ~98 %  -> komfortables Verstehen
  ~95 %  -> verstehbar mit Anstrengung
  darunter -> frustrierend, erst Vokabeln vorbereiten
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import db
from .vocab_model import effective_p, load_model

TARGET_COMFORT = 0.98
TARGET_OK = 0.95

# Quiz: nur Woerter abfragen, bei denen das Modell unsicher ist
QUIZ_P_MIN, QUIZ_P_MAX = 0.10, 0.92


@dataclass
class LineView:
    line_no: int
    text: str
    t_start: float | None
    tokens: list[dict] = field(default_factory=list)  # surface,c0,c1,kind,p,lemma,lemma_id,status
    expected_unknown: float = 0.0

    @property
    def ampel(self) -> str:
        if self.expected_unknown < 0.25:
            return "gruen"
        if self.expected_unknown < 1.25:
            return "gelb"
        return "rot"


@dataclass
class DocStats:
    coverage: float
    n_tokens: int
    n_content_lemmas: int
    n_unknown: int          # explizit unbekannt oder Prior < 0.5
    n_learning: int
    n_known: int
    n_untested: int         # Inhaltslemmata ohne expliziten Status
    words_to_98: int
    words_to_95: int


def _lemma_p_map(doc_id: int, params) -> tuple[dict[int, float], list]:
    """p_known je Inhaltslemma des Dokuments + die Summary-Zeilen."""
    summary = db.doc_lemma_summary(doc_id)
    lemma_p = {
        r["lemma_id"]: effective_p(r["status"], r["zipf"] or 0.0, params)
        for r in summary
    }
    return lemma_p, summary


def doc_stats(doc_id: int) -> DocStats:
    """Schnelle Statistik auf Lemma-Ebene (laedt keine Einzeltokens)."""
    params = load_model()
    lemma_p, summary = _lemma_p_map(doc_id, params)
    kinds = db.doc_kind_counts(doc_id)
    n_easy = kinds.get("function", 0) + kinds.get("proper", 0)

    p_sum = float(n_easy)
    n_tok = n_easy
    n_unknown = n_learning = n_known = n_untested = 0
    deficits: list[tuple[float, int]] = []
    for row in summary:
        p = lemma_p[row["lemma_id"]]
        p_sum += p * row["cnt"]
        n_tok += row["cnt"]
        st = row["status"]
        if st == "learning":
            n_learning += 1
        elif st == "known" or (st is None and p >= 0.9):
            n_known += 1
        elif st == "unknown" or p < 0.5:
            n_unknown += 1
        if st is None:
            n_untested += 1
        gain = (1.0 - p) * row["cnt"]
        if gain > 1e-6 and st != "ignored":
            deficits.append((gain, row["lemma_id"]))

    coverage = (p_sum / n_tok) if n_tok else 1.0
    deficits.sort(reverse=True)

    def words_needed(target: float) -> int:
        if coverage >= target or n_tok == 0:
            return 0
        need = (target - coverage) * n_tok
        acc, k = 0.0, 0
        for gain, _ in deficits:
            acc += gain
            k += 1
            if acc >= need:
                return k
        return k

    return DocStats(
        coverage=coverage, n_tokens=n_tok, n_content_lemmas=len(summary),
        n_unknown=n_unknown, n_learning=n_learning, n_known=n_known,
        n_untested=n_untested,
        words_to_98=words_needed(TARGET_COMFORT),
        words_to_95=words_needed(TARGET_OK),
    )


def doc_analysis(doc_id: int) -> tuple[list[LineView], DocStats]:
    """Zeilenansicht (mit Wort-p und Ampel) + Statistik."""
    params = load_model()
    stats = doc_stats(doc_id)
    lemma_p, _ = _lemma_p_map(doc_id, params)
    lines = {r["line_no"]: LineView(r["line_no"], r["text"], r["t_start"])
             for r in db.get_lines(doc_id)}

    for r in db.doc_token_rows(doc_id):
        kind = r["kind"]
        if kind == "proper" or r["lemma_id"] is None or kind == "function" \
                or r["is_function"]:
            p = 1.0
        else:
            p = lemma_p.get(
                r["lemma_id"],
                effective_p(r["status"], r["zipf"] or 0.0, params),
            )
        lv = lines.get(r["line_no"])
        if lv is not None:
            lv.tokens.append({
                "surface": r["surface"], "c0": r["c0"], "c1": r["c1"],
                "kind": kind, "p": p, "lemma": r["lemma"],
                "lemma_id": r["lemma_id"], "status": r["status"],
            })
            if kind == "content":
                lv.expected_unknown += (1.0 - p)

    return sorted(lines.values(), key=lambda l: l.line_no), stats


def unlock_words(doc_id: int, target: float = TARGET_COMFORT) -> list[dict]:
    """Die konkreten Woerter, die dieses Dokument bis zum Ziel freischalten."""
    params = load_model()
    stats = doc_stats(doc_id)
    rows = db.doc_lemma_summary(doc_id)
    scored = []
    for r in rows:
        if r["status"] in ("known", "ignored"):
            continue
        p = effective_p(r["status"], r["zipf"] or 0.0, params)
        gain = (1.0 - p) * r["cnt"]
        if gain > 1e-6:
            scored.append({
                "lemma_id": r["lemma_id"], "lemma": r["lemma"],
                "cnt": r["cnt"], "zipf": r["zipf"], "p": p, "gain": gain,
                "status": r["status"], "translation": r["translation"],
            })
    scored.sort(key=lambda x: x["gain"], reverse=True)
    if stats.n_tokens == 0:
        return []
    need = max(0.0, (target - stats.coverage) * stats.n_tokens)
    out, acc = [], 0.0
    for s in scored:
        if acc >= need and out:
            break
        out.append(s)
        acc += s["gain"]
    return out


def quiz_candidates(doc_id: int, limit: int = 60) -> list[dict]:
    """Woerter dieses Dokuments, die das Quiz klaeren sollte: ohne expliziten
    Status und mit unsicherem Prior. Sortiert nach Wichtigkeit im Dokument."""
    params = load_model()
    rows = db.doc_lemma_summary(doc_id)
    cands = []
    for r in rows:
        if r["status"] is not None:
            continue
        p = effective_p(None, r["zipf"] or 0.0, params)
        if QUIZ_P_MIN <= p <= QUIZ_P_MAX:
            cands.append({
                "lemma_id": r["lemma_id"], "lemma": r["lemma"],
                "cnt": r["cnt"], "zipf": r["zipf"], "p": p,
                "first_line": r["first_line"],
            })
    cands.sort(key=lambda x: (-x["cnt"], -(x["zipf"] or 0)))
    return cands[:limit]


def global_roi_words(limit: int = 50) -> list[dict]:
    """Cross-Dokument-Lernliste: Woerter, die in MEHREREN Dokumenten
    vorkommen, zuerst - maximaler Coverage-Gewinn pro gelerntem Wort."""
    params = load_model()
    rows = db.all_doc_content_counts()
    agg: dict[int, dict] = {}
    for r in rows:
        if r["status"] in ("known", "ignored", "learning"):
            continue
        p = effective_p(r["status"], r["zipf"] or 0.0, params)
        if p >= 0.9:
            continue
        e = agg.setdefault(r["lemma_id"], {
            "lemma_id": r["lemma_id"], "lemma": r["lemma"],
            "zipf": r["zipf"], "docs": 0, "total_cnt": 0, "p": p,
        })
        e["docs"] += 1
        e["total_cnt"] += r["cnt"]
    out = sorted(agg.values(),
                 key=lambda x: (x["docs"], x["total_cnt"] * (1 - x["p"])),
                 reverse=True)
    return out[:limit]
