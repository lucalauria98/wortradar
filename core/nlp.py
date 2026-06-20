"""NLP-Schicht von WortRadar (Englisch).

Zwei Pfade, ein Interface:
  1. spaCy + en_core_web_sm  -> beste Qualitaet (POS-gestuetzte Lemmata,
     saubere Eigennamen-Erkennung). Wird genutzt, sobald das Modell
     installiert ist:  python -m spacy download en_core_web_sm
  2. Fallback ohne Modell: lemminflect (Lemma-Kandidaten) + wordfreq
     (Frequenz-Disambiguierung) + Heuristiken. Laeuft sofort ueberall.

Beide liefern pro Zeile eine Liste von Tok-Objekten.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from wordfreq import zipf_frequency

# Woerter, die fuer Coverage automatisch als bekannt gelten und nie im Deck
# landen: Artikel, Pronomen, Praepositionen, Hilfsverben, Konjunktionen,
# haeufige Kontraktionen. (Lerner ab A2 kennen diese ohnehin; klassische
# Coverage-Forschung zaehlt sie als Tokens mit, also tun wir das auch.)
FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "no",
    "every", "each", "either", "neither", "both", "all", "such", "own", "same",
    "other", "another",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "yourselves", "themselves", "who", "whom", "whose",
    "which", "what", "where", "when", "why", "how", "there", "here",
    "someone", "anyone", "everyone", "nobody", "somebody", "anybody",
    "everybody", "something", "anything", "everything", "nothing", "one",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "of", "off", "over", "under", "again", "further",
    "out", "around", "near", "until", "till", "since", "upon", "within",
    "without", "along", "across", "behind", "beyond", "among", "toward",
    "towards", "onto", "per",
    "and", "but", "or", "nor", "so", "yet", "if", "because", "as", "while",
    "although", "though", "whether", "unless", "than", "then", "once",
    "be", "am", "is", "are", "was", "were", "been", "being", "do", "does",
    "did", "doing", "have", "has", "had", "having", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "ought",
    "not", "n't", "yes", "ok", "okay", "oh", "ah", "uh", "hey", "hmm", "yeah",
    "wow", "ooh", "whoa", "huh", "uhm", "um", "la", "na", "mmm", "hm",
    "very", "too", "just", "also", "only", "even", "still", "already", "ever",
    "never", "always", "often", "sometimes", "now", "soon", "well", "really",
    "quite", "rather", "almost", "enough", "much", "many", "more", "most",
    "few", "less", "least", "lot", "lots",
    # haeufige Kontraktionen, falls der Tokenizer sie nicht aufspaltet
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've",
    "you've", "we've", "they've", "i'll", "you'll", "he'll", "she'll",
    "we'll", "they'll", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't",
    "mustn't", "haven't", "hasn't", "hadn't", "let's", "that's", "there's",
    "here's", "what's", "who's", "where's", "ain't", "gonna", "wanna",
    "gotta", "'s", "'re", "'ve", "'ll", "'d", "'m",
}

# Interjektionen/Fuellwoerter, die nie als Vokabel taugen
_NOISE = {"ooh", "oohh", "lala", "nanana", "dadada", "shh", "psst"}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z''\-]*[A-Za-z]|[A-Za-z]")

_SPACY_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
_SPACY_FUNCTION_POS = {"DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "INTJ", "NUM", "SYM", "PUNCT", "SPACE", "X"}


@dataclass
class Tok:
    surface: str
    c0: int
    c1: int
    lemma: str          # normalisiertes Lemma (lowercase)
    zipf: float
    kind: str           # content | function | proper


def _is_contraction_suffix(surface: str) -> bool:
    """spaCy zerlegt 'don't' -> 'do'+'n't', 'it's' -> 'it'+''s'. Solche
    Suffixe wollen wir wieder ans vorige Wort haengen (eine Box statt zwei)."""
    s = surface.lower()
    return s in ("n't", "n’t") or (len(s) >= 2 and s[0] in "'’")


class _SpacyEngine:
    name = "spaCy (en_core_web_sm)"

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def analyze_lines(self, lines: list[str]) -> list[list[Tok]]:
        out: list[list[Tok]] = []
        for doc in self.nlp.pipe(lines, batch_size=256):
            toks: list[Tok] = []
            for t in doc:
                # Kontraktions-Suffix ('don't'->'do'+'n't', 'it's'->'it'+''s')
                # ans direkt davor stehende Wort haengen -> eine Box statt zwei.
                if toks and _is_contraction_suffix(t.text) and t.idx == toks[-1].c1:
                    prev = toks[-1]
                    prev.surface += t.text
                    prev.c1 = t.idx + len(t.text)
                    prev.kind = "function"   # Kontraktionen gelten als bekannt
                    continue
                if not WORD_RE.fullmatch(t.text):
                    continue
                low = t.text.lower().strip("'")
                lemma = (t.lemma_ or low).lower().strip("'")
                if t.pos_ == "PROPN":
                    kind = "proper"
                elif low in FUNCTION_WORDS or lemma in FUNCTION_WORDS \
                        or t.pos_ in _SPACY_FUNCTION_POS or low in _NOISE:
                    kind = "function"
                elif t.pos_ in _SPACY_CONTENT_POS:
                    kind = "content"
                else:
                    kind = "function"
                z = zipf_frequency(lemma, "en")
                toks.append(Tok(t.text, t.idx, t.idx + len(t.text), lemma, z, kind))
            out.append(toks)
        return out


class _FallbackEngine:
    name = "Fallback (lemminflect)"

    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

    def __init__(self):
        from lemminflect import getAllLemmas
        self._all_lemmas = getAllLemmas

    def _lemma_of(self, low: str) -> tuple[str, set]:
        cands = self._all_lemmas(low) or {}
        pos_set = set(cands.keys())
        options = {lm.lower() for tup in cands.values() for lm in tup}
        if not options:
            return low, pos_set
        # Frequenz-Disambiguierung: plausibelstes (haeufigstes) Lemma waehlen
        best = max(options, key=lambda w: zipf_frequency(w, "en"))
        return best, pos_set

    def analyze_lines(self, lines: list[str]) -> list[list[Tok]]:
        out: list[list[Tok]] = []
        for line in lines:
            toks: list[Tok] = []
            for m in WORD_RE.finditer(line):
                surface = m.group(0)
                low = surface.lower().replace("'", "'").strip("'")
                lemma, pos_set = self._lemma_of(low)
                z_low = zipf_frequency(low, "en")
                if low in FUNCTION_WORDS or lemma in FUNCTION_WORDS or low in _NOISE:
                    kind = "function"
                elif surface[:1].isupper() and z_low < 1.5 and not pos_set:
                    # Grossgeschrieben, im Korpus quasi unbekannt, kein
                    # englisches Flexionsmuster -> sehr wahrscheinlich Eigenname
                    kind = "proper"
                else:
                    kind = "content"
                z = zipf_frequency(lemma, "en")
                if z == 0:
                    z = z_low
                toks.append(Tok(surface, m.start(), m.end(), lemma, z, kind))
            out.append(toks)
        return out


_engine = None


def get_engine():
    """Liefert die beste verfuegbare Engine (cached)."""
    global _engine
    if _engine is not None:
        return _engine
    try:
        _engine = _SpacyEngine()
    except Exception:
        _engine = _FallbackEngine()
    return _engine


def engine_name() -> str:
    return get_engine().name


def analyze_lines(lines: list[str]) -> list[list[Tok]]:
    return get_engine().analyze_lines(lines)
