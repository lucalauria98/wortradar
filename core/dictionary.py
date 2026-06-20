"""Uebersetzungen Englisch -> Deutsch.

Drei Quellen, alle optional und kombinierbar:

1. Offline-Woerterbuch (empfohlen, einmalig, gratis):
   - TU-Chemnitz-"Ding"-Liste (GPL): https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en/de-en.txt.gz
     (die .gz direkt in den Einstellungen hochladen, kein Entpacken noetig)
   - oder ein dict.cc-Export (fuer den persoenlichen Gebrauch nach
     Registrierung herunterladbar; TSV-Format)
   Die Datei bleibt lokal; nichts wird hochgeladen.

2. Claude-API (optional): uebersetzt fehlende Woerter im Batch MIT
   Kontextzeile aus deinem Dokument - dadurch trifft es auch Slang und
   die im Kontext richtige Bedeutung. Braucht einen eigenen API-Key.

3. Manuell: jede Uebersetzung ist im Deck direkt editierbar.
"""
from __future__ import annotations

import json
import re

import requests
from wordfreq import zipf_frequency

from . import db

_ANNOT = re.compile(r"\{[^}]*\}|\[[^\]]*\]|<[^>]*>")
_PAREN = re.compile(r"\([^)]*\)")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5"

# Google Gemini - grosszuegiger kostenloser Tarif, gut fuer Online-Betrieb
# (ein Server-Key fuer alle Besucher, keine Kreditkarte noetig).
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

# Groq - beste kostenlose Limits (schnell, ~30 Anfragen/Min, keine Kreditkarte),
# OpenAI-kompatible API. Empfohlen fuers Gratis-Modell.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Freie Ding-Liste der TU Chemnitz (GPL) - direkt per Ein-Klick ladbar.
DING_URL = "https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en/de-en.txt.gz"


def download_and_import_ding() -> int:
    """Laedt die freie Ding-Liste (GPL) direkt herunter und importiert sie.
    Kein Account, kein manueller Download noetig. Liefert die Anzahl Eintraege."""
    import gzip
    r = requests.get(DING_URL, timeout=120)
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    return import_dictionary_file(text)


def _clean_en_key(s: str) -> str:
    s = _ANNOT.sub("", s)
    s = _PAREN.sub("", s)
    # Ding-Objektplatzhalter entfernen: "to forget sth." -> "forget"
    s = re.sub(r"\b(?:sth|sb|so|sm|sn)\.", "", s, flags=re.I)
    s = re.sub(r"\b(?:something|somebody|someone|oneself|one's)\b", "", s, flags=re.I)
    s = re.sub(r"^to\s+", "", s.strip())  # "to depart" -> "depart"
    return re.sub(r"\s+", " ", s).strip().strip(";,").lower()


def _clean_de(s: str) -> str:
    s = _ANNOT.sub("", s)               # {f}, [comp.], <html> entfernen
    # Deutsche Objektplatzhalter weg ("etw. vergessen" -> "vergessen");
    # "sich" bleibt erhalten (gehoert zum Reflexivverb).
    s = re.sub(r"\b(?:etw|jdn|jdm|jds|jd)\.", "", s)
    s = re.sub(r"\s+", " ", s).strip().strip(";,")
    return s


_DE_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_EN_KEY_OK = re.compile(r"[a-z][a-z'\-]*")


def _de_score(de_val: str) -> tuple:
    """Kleiner = bessere Lernkarten-Uebersetzung. Reihenfolge der Kriterien:
      1. moeglichst EIN Wort (Grundbedeutung statt Umschreibung)
      2. keine Klammer-Erklaerung, kein 'etw./jdn.'-Ballast
      3. unter Einzelwoertern das HAEUFIGSTE (wordfreq) - 'schnell' statt 'flugs'
      4. am Ende kuerzer
    Frequenz steht bewusst NACH der Wortzahl, sonst gewinnen Phrasen mit
    haeufigen Fuellwoertern ('ueber', 'nicht')."""
    words = _DE_WORD.findall(de_val)
    n_words = len(words) or 99
    bad = 1 if ("(" in de_val or "…" in de_val or "..." in de_val
                or re.search(r"\b(etw|jdn|jdm|jds)\b", de_val)) else 0
    freq = max((zipf_frequency(w, "de") for w in words), default=0.0)
    return (n_words, bad, -round(freq, 1), len(de_val))


def _better(old: str | None, new: str) -> str:
    if not old:
        return new
    return new if _de_score(new) < _de_score(old) else old


def import_dictionary_file(text: str) -> int:
    """Erkennt Ding- oder dict.cc-Format automatisch und importiert.

    Pro englischem Wort werden ALLE Kandidaten gesammelt und die fuer
    Lernkarten beste gewaehlt (kurz, klar, gelaeufig) - statt einfach die
    erste Zeile zu nehmen.
    """
    entries: dict[str, str] = {}

    def consider(en_raw: str, de_raw: str) -> None:
        en_key = _clean_en_key(en_raw)
        if not en_key or not _EN_KEY_OK.fullmatch(en_key):
            return
        de_raw = _ANNOT.sub("", de_raw)               # {f}, {forgot; ...} etc. weg
        de_first = de_raw.split(";")[0]               # erste Synonym-Variante
        de_clean = _clean_de(_PAREN.sub("", de_first)) or _clean_de(de_first)
        if de_clean:
            entries[en_key] = _better(entries.get(en_key), de_clean)

    sample = next((l for l in text.splitlines() if l.strip() and not l.startswith("#")), "")
    is_ding = " :: " in sample

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if is_ding:
            if " :: " not in line:
                continue
            de_side, en_side = line.split(" :: ", 1)
            for de_p, en_p in zip(de_side.split("|"), en_side.split("|")):
                en_p = _ANNOT.sub("", en_p)           # Annotationen VOR dem ;-Split
                for en_syn in en_p.split(";"):        # jede engl. Synonym-Variante
                    consider(en_syn, de_p)
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                consider(parts[0], parts[1])
    return db.dict_store(entries, "ding" if is_ding else "dictcc")


# --------------------------------------------------------- KI-Uebersetzung ----
def _translate_prompt(words_with_context: list[tuple[str, str]]) -> str:
    listing = "\n".join(
        f'- "{w}" (Kontext: {ctx[:160]})' if ctx else f'- "{w}"'
        for w, ctx in words_with_context
    )
    return (
        "Du bist ein Woerterbuch Englisch->Deutsch fuer einen deutschen "
        "Englischlerner. Uebersetze jedes Wort knapp (1-4 Woerter), passend "
        "zum jeweiligen Kontext (z. B. 'country lane' -> 'Landstrasse', nicht "
        "'Gasse'). Bei Slang gib die Bedeutung plus Hinweis in Klammern, "
        "z. B. 'abhaengen (Slang)'.\n\n"
        f"{listing}\n\n"
        "Antworte AUSSCHLIESSLICH mit einem JSON-Objekt "
        '{"wort": "uebersetzung", ...} ohne Markdown und ohne Erklaerung.'
    )


def _parse_json_object(text: str) -> dict[str, str]:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return {str(k).lower(): str(v) for k, v in parsed.items() if v}


class AITranslationError(Exception):
    """Verständliche Fehlermeldung für die KI-Übersetzung (statt Traceback)."""


def _http_error_message(resp) -> str:
    detail = ""
    try:
        detail = ((resp.json().get("error") or {}).get("message") or "")[:200]
    except Exception:  # noqa: BLE001
        detail = (getattr(resp, "text", "") or "")[:200]
    code = resp.status_code
    if code == 429:
        return ("Ratenlimit des kostenlosen Tarifs erreicht (nur wenige "
                "Anfragen pro Minute). Kurz warten und erneut versuchen.")
    if code in (401, 403):
        return f"API-Key ungültig oder ohne Berechtigung. {detail}".strip()
    if code == 404:
        return ("Modell nicht gefunden. In den Einstellungen ein anderes "
                "probieren, z. B. „gemini-1.5-flash“ oder „gemini-2.0-flash-lite“.")
    return f"Fehler {code}: {detail}".strip()


def _ai_post(url: str, headers: dict, payload: dict):
    """POST mit einem Retry bei 429; wirft AITranslationError mit klarer Meldung."""
    import time
    resp = None
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            raise AITranslationError(f"Netzwerkfehler: {e}")
        if resp.status_code == 429 and attempt == 0:
            time.sleep(3)            # kurz warten, dann ein zweiter Versuch
            continue
        break
    if resp.status_code >= 400:
        raise AITranslationError(_http_error_message(resp))
    return resp


def gemini_translate(words_with_context: list[tuple[str, str]],
                     api_key: str, model: str = GEMINI_DEFAULT_MODEL) -> dict[str, str]:
    """Wie llm_translate, aber ueber Google Gemini (kostenloser Tarif)."""
    if not words_with_context:
        return {}
    resp = _ai_post(
        GEMINI_URL.format(model=model),
        {"x-goog-api-key": api_key, "content-type": "application/json"},
        {
            "contents": [{"parts": [{"text": _translate_prompt(words_with_context)}]}],
            "generationConfig": {"temperature": 0.2,
                                 "responseMimeType": "application/json"},
        },
    )
    data = resp.json()
    text = "".join(
        p.get("text", "")
        for cand in data.get("candidates", [])
        for p in cand.get("content", {}).get("parts", [])
    )
    out = _parse_json_object(text)
    if out:
        db.dict_store(out, "llm")
    return out


def groq_translate(words_with_context: list[tuple[str, str]],
                   api_key: str, model: str = GROQ_DEFAULT_MODEL) -> dict[str, str]:
    """Wie llm_translate, aber ueber Groq (OpenAI-kompatibel, beste Gratis-Limits)."""
    if not words_with_context:
        return {}
    resp = _ai_post(
        GROQ_URL,
        {"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        {
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user",
                          "content": _translate_prompt(words_with_context)}],
        },
    )
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    out = _parse_json_object(text)
    if out:
        db.dict_store(out, "llm")
    return out


def llm_translate(words_with_context: list[tuple[str, str]],
                  api_key: str, model: str = DEFAULT_MODEL) -> dict[str, str]:
    """Uebersetzt [(wort, kontextzeile)] -> {wort: deutsch} ueber Claude.

    Kontextzeile stammt aus dem Dokument des Nutzers (eine Zeile) und sorgt
    dafuer, dass Slang und Mehrdeutigkeiten richtig getroffen werden.
    """
    if not words_with_context:
        return {}
    resp = _ai_post(
        ANTHROPIC_URL,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        {
            "model": model,
            "max_tokens": 2000,
            "messages": [{"role": "user",
                          "content": _translate_prompt(words_with_context)}],
        },
    )
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    out = _parse_json_object(text)
    if out:
        db.dict_store(out, "llm")
    return out


# Provider-Konfiguration: liest Key/Modell aus meta-Tabelle ODER (fuer das
# Online-Deployment) aus Umgebungsvariablen GROQ_API_KEY / GEMINI_API_KEY /
# ANTHROPIC_API_KEY. So braucht im Online-Betrieb kein Besucher einen Account.
# Reihenfolge = Vorrang: Groq (beste Gratis-Limits) > Gemini > Claude.
def get_ai_config() -> dict:
    import os
    groq_key = (db.meta_get("groq_key") or os.environ.get("GROQ_API_KEY") or "").strip()
    gem_key = (db.meta_get("gemini_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
    cla_key = (db.meta_get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if groq_key:
        return {"provider": "groq", "key": groq_key,
                "model": db.meta_get("groq_model") or GROQ_DEFAULT_MODEL}
    if gem_key:
        return {"provider": "gemini", "key": gem_key,
                "model": db.meta_get("gemini_model") or GEMINI_DEFAULT_MODEL}
    if cla_key:
        return {"provider": "claude", "key": cla_key,
                "model": db.meta_get("api_model") or DEFAULT_MODEL}
    return {"provider": None, "key": "", "model": ""}


def ai_translate(words_with_context: list[tuple[str, str]],
                 cfg: dict | None = None) -> dict[str, str]:
    """Dispatcher: nutzt den konfigurierten KI-Anbieter (Groq bevorzugt)."""
    cfg = cfg or get_ai_config()
    if cfg["provider"] == "groq":
        return groq_translate(words_with_context, cfg["key"], cfg["model"])
    if cfg["provider"] == "gemini":
        return gemini_translate(words_with_context, cfg["key"], cfg["model"])
    if cfg["provider"] == "claude":
        return llm_translate(words_with_context, cfg["key"], cfg["model"])
    return {}


def context_translate(word: str, context: str) -> str | None:
    """Einzelnes Wort im Satzkontext uebersetzen (fuer den Karten-Button).
    Liefert die deutsche Uebersetzung oder None, wenn keine KI konfiguriert ist."""
    cfg = get_ai_config()
    if not cfg["provider"]:
        return None
    got = ai_translate([(word, context)], cfg)
    return got.get(word.lower())


def ai_retranslate(words_with_context: list[tuple[str, str]]) -> tuple[int, str | None]:
    """Uebersetzt ALLE uebergebenen Woerter im Kontext per KI neu und speichert
    sie als 'manuell' - ueberschreibt also auch falsche Offline-Treffer
    (z. B. 'lane' -> 'Gasse' wird zu 'Fahrbahn'). Liefert (anzahl, fehler).
    Eine Anfrage pro 40 Woerter, bleibt damit im kostenlosen Ratenlimit."""
    cfg = get_ai_config()
    if not cfg["provider"]:
        return 0, ("Keine KI konfiguriert - kostenlosen Gemini-Key in den "
                   "Einstellungen hinterlegen.")
    n = 0
    try:
        for i in range(0, len(words_with_context), 40):
            got = ai_translate(words_with_context[i:i + 40], cfg)
            if got:
                db.dict_store(got, "manuell")   # ueberschreibt vorhandene
                n += len(got)
    except AITranslationError as e:
        return n, str(e)
    return n, None


_MM_GARBAGE = re.compile(
    r"MYMEMORY WARNING|YOU USED ALL|QUOTA|INVALID|PLEASE", re.I)


def mymemory_translate(words: list[str]) -> tuple[dict[str, str], bool]:
    """Uebersetzt Woerter EN->DE via MyMemory (kostenlos, kein Account noetig).

    Liefert (uebersetzungen, limit_erreicht). Bei erreichtem Tageslimit gibt
    MyMemory keinen 429, sondern Warntext im Feld translatedText zurueck - das
    wird hier erkannt, damit kein Muell als Uebersetzung gespeichert wird.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stop = threading.Event()

    def _fetch(word: str) -> tuple[str, str] | None:
        if stop.is_set():
            return None
        try:
            r = requests.get(
                "https://api.mymemory.translated.net/get",
                params={"q": word, "langpair": "en|de"},
                timeout=10,
            )
            if r.status_code == 429:
                stop.set()
                return None
            data = r.json()
            try:
                status = int(str(data.get("responseStatus", 0)).split()[0])
            except (ValueError, IndexError):
                status = 0
            txt = (data.get("responseData") or {}).get("translatedText", "") or ""
            if _MM_GARBAGE.search(txt) or status == 429:
                stop.set()              # Tageslimit erreicht -> Rest abbrechen
                return None
            txt = txt.strip()
            # Einzelwoerter haben kurze Uebersetzungen. MyMemory liefert bei
            # Einzelwoertern oft Phrasen-Muell ("mountain -> Berg hinunter und")
            # aus seinem Uebersetzungsgedaechtnis -> solche Treffer verwerfen.
            if " " not in word and len(txt.split()) > 2:
                return None
            if status == 200 and txt and txt.lower() != word.lower() \
                    and len(txt) <= 60:
                return (word.lower(), txt)
        except Exception:
            pass
        return None

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed(ex.submit(_fetch, w) for w in words):
            result = fut.result()
            if result:
                out[result[0]] = result[1]
    if out:
        db.dict_store(out, "mymemory")
    return out, stop.is_set()


def translate_missing(rows: list[dict], context_of: dict[str, str]) -> dict:
    """Fuellt fehlende Uebersetzungen der Reihe nach auf:
      1. Offline-Woerterbuch (gratis, sofort, deckt das meiste)
      2. KI (Gemini/Claude) - KONTEXTBEWUSST, loest 'country lane'->'Landstrasse'
      3. MyMemory nur als Fallback, falls keine KI konfiguriert ist

    rows: [{lemma, translation}]; liefert
      {offline, ai, mymemory, missing, total_new, error}
    """
    res = {"offline": 0, "ai": 0, "mymemory": 0, "missing": 0,
           "total_new": 0, "error": None}
    missing = [r["lemma"] for r in rows if not r.get("translation")]
    if not missing:
        return res

    found = db.dict_lookup(missing)
    res["offline"] = len(found)
    still = [w for w in missing if w not in found]

    cfg = get_ai_config()
    if still and cfg["provider"]:
        try:
            for i in range(0, len(still), 40):
                chunk = [(w, context_of.get(w, "")) for w in still[i:i + 40]]
                got = ai_translate(chunk, cfg)
                res["ai"] += len(got)
                still = [w for w in still if w not in got]
        except AITranslationError as e:
            res["error"] = str(e)        # bereits verstaendlich formuliert
        except Exception as e:  # noqa: BLE001
            res["error"] = f"KI-Uebersetzung fehlgeschlagen: {e}"
    elif still:
        # Keine KI konfiguriert -> MyMemory als kostenloser Fallback (ohne
        # Kontext, daher bei Mehrdeutigkeit weniger treffsicher).
        try:
            mm, quota_hit = mymemory_translate(still)
            res["mymemory"] = len(mm)
            still = [w for w in still if w not in mm]
            if still and quota_hit:
                res["error"] = ("MyMemory-Tageslimit erreicht. Fuer bessere, "
                                "kontextgenaue Uebersetzungen einen kostenlosen "
                                "Gemini-Key in den Einstellungen hinterlegen.")
        except Exception:  # noqa: BLE001
            pass

    res["missing"] = len(still)
    res["total_new"] = res["offline"] + res["ai"] + res["mymemory"]
    return res
