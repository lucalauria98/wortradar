"""Wortschatzmodell von WortRadar.

Idee (gestuetzt auf Vokabelforschung, vgl. IRT-basierte Tests wie LexTALE
oder den Polish Vocabulary Size Test): Die Wahrscheinlichkeit, ein Wort zu
kennen, haengt stark von seiner Korpus-Frequenz ab. Wir testen eine kleine
Stichprobe quer ueber die Frequenzbaender (Zipf-Skala 1..7) und fitten

    P(kennt Wort | zipf) = g + (1 - g) * sigmoid(a * (zipf - b))

g  = Guessing-Rate, geschaetzt aus den Ja-Antworten auf Pseudowoerter
b  = "Schwellen-Zipf": dort liegt die 50%-Grenze des Lerners
a  = Steilheit der Kurve

Damit bekommt JEDES englische Wort sofort eine Kenn-Wahrscheinlichkeit
(Prior). Explizite Antworten (Dokument-Quiz, Lernkarten) ueberschreiben
den Prior pro Wort. Je mehr du interagierst, desto genauer wird alles.
"""
from __future__ import annotations

import math
import random

import numpy as np
from scipy.optimize import minimize
from wordfreq import top_n_list, zipf_frequency

from . import db
from .nlp import FUNCTION_WORDS
from .pseudowords import PSEUDOWORDS

# Frequenzbaender (Zipf): von "kennt fast jeder Lerner" bis "sehr selten"
BANDS = [
    (6.0, 7.5), (5.4, 6.0), (4.8, 5.4), (4.2, 4.8),
    (3.6, 4.2), (3.0, 3.6), (2.4, 3.0), (1.7, 2.4),
]
WORDS_PER_BAND = 6
PSEUDO_PER_TEST = 10

# Defaults, solange kein Test gemacht wurde (mittlerer Lerner, deutlich
# als "unkalibriert" markiert in der UI)
DEFAULT_A, DEFAULT_B, DEFAULT_G = 1.8, 4.2, 0.0


# ------------------------------------------------------------ Testpool ----
def _build_pool() -> dict[str, list[list]]:
    """Kandidaten je Band aus der wordfreq-Top-Liste, nur saubere Grundformen."""
    try:
        from lemminflect import getAllLemmas
    except ImportError:  # pragma: no cover
        getAllLemmas = lambda w: {}
    pool: dict[str, list[list]] = {f"{lo}-{hi}": [] for lo, hi in BANDS}
    seen: set[str] = set()
    for w in top_n_list("en", 40000):
        if not w.isalpha() or len(w) < 3 or w in FUNCTION_WORDS or w in seen:
            continue
        z = zipf_frequency(w, "en")
        band = next((f"{lo}-{hi}" for lo, hi in BANDS if lo <= z < hi), None)
        if band is None:
            continue
        cands = getAllLemmas(w) or {}
        base_forms = {lm.lower() for tup in cands.values() for lm in tup}
        if base_forms and w not in base_forms:
            continue  # flektierte Form -> nicht als Testwort verwenden
        pool[band].append([w, round(z, 2)])
        seen.add(w)
    return pool


def get_pool() -> dict[str, list[list]]:
    pool = db.meta_get_json("test_pool")
    if not pool:
        pool = _build_pool()
        db.meta_set_json("test_pool", pool)
    return pool


def make_test_items(seed: int | None = None) -> list[dict]:
    """Zufaellige Testitems: echte Woerter ueber alle Baender + Pseudowoerter."""
    rng = random.Random(seed)
    pool = get_pool()
    items: list[dict] = []
    for band, cands in pool.items():
        take = rng.sample(cands, min(WORDS_PER_BAND, len(cands)))
        for w, z in take:
            items.append({"word": w, "zipf": z, "pseudo": False})
    for w in rng.sample(PSEUDOWORDS, PSEUDO_PER_TEST):
        items.append({"word": w, "zipf": 0.0, "pseudo": True})
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------- Fit ----
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_curve(answers: list[dict]) -> dict:
    """answers: [{zipf, pseudo, answer(bool)}] -> {a, b, g, n_real, n_pseudo}"""
    pseudo = [x for x in answers if x["pseudo"]]
    real = [x for x in answers if not x["pseudo"]]
    g = 0.0
    if pseudo:
        g = sum(1 for x in pseudo if x["answer"]) / len(pseudo)
    g = float(min(max(g, 0.0), 0.45))

    z = np.array([x["zipf"] for x in real], dtype=float)
    y = np.array([1.0 if x["answer"] else 0.0 for x in real])

    def nll(params):
        a, b = params
        p = g + (1 - g) * _sigmoid(a * (z - b))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # leichte Regularisierung haelt den Fit bei Extremantworten stabil
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) \
            + 0.05 * (params[0] - DEFAULT_A) ** 2

    best = minimize(
        nll, x0=[DEFAULT_A, DEFAULT_B], method="L-BFGS-B",
        bounds=[(0.4, 6.0), (0.8, 7.5)],
    )
    a, b = (float(v) for v in best.x)
    return {"a": a, "b": b, "g": g, "n_real": len(real), "n_pseudo": len(pseudo)}


def save_model(params: dict) -> None:
    db.meta_set_json("vocab_model", params)


def load_model() -> dict | None:
    return db.meta_get_json("vocab_model")


def is_calibrated() -> bool:
    return load_model() is not None


def p_known_from_zipf(zipf: float, params: dict | None = None) -> float:
    """Prior-Wahrscheinlichkeit, dass der Lerner ein Wort dieses Zipf kennt."""
    if params is None:
        params = load_model() or {"a": DEFAULT_A, "b": DEFAULT_B, "g": 0.0}
    if zipf <= 0:
        return 0.05  # nicht im Korpus (Slang, Tippfehler, sehr selten)
    raw = 1.0 / (1.0 + math.exp(-params["a"] * (zipf - params["b"])))
    return float(min(max(raw, 0.01), 0.995))


def effective_p(status: str | None, zipf: float, params: dict | None = None) -> float:
    """Kombiniert expliziten Status mit dem Frequenz-Prior."""
    if status == "known":
        return 1.0
    if status == "ignored":
        return 1.0  # zaehlt nicht gegen die Coverage (z. B. Eigennamen)
    if status == "unknown":
        return 0.0
    if status == "learning":
        return 0.5
    return p_known_from_zipf(zipf, params)


# ------------------------------------------------- Wortschatz-Schaetzung ----
def estimate_vocab_size(params: dict | None = None) -> int:
    """Grobe Schaetzung der gekannten Wortfamilien: Summe der Priors ueber
    die Grundformen der 40k haeufigsten Woerter."""
    if params is None:
        params = load_model()
    if params is None:
        return 0
    pool = get_pool()
    total = 0.0
    for band in pool.values():
        for _, z in band:
            total += p_known_from_zipf(z, params)
    # Funktions-/Kernwortschatz, der im Pool bewusst fehlt, pauschal dazu
    return int(total) + len(FUNCTION_WORDS)
