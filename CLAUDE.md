# CLAUDE.md — Projektkontext WortRadar

> Diese Datei wird von Claude Code beim Start automatisch gelesen. Sie fasst
> alles zusammen, was in der vorherigen Chat-Sitzung (claude.ai) erarbeitet
> wurde, damit hier in VS Code nahtlos weitergearbeitet werden kann.
> Sprache mit dem Nutzer: **Deutsch**. Nutzer: Luca ("Maestro"), arbeitet auf
> Windows mit VS Code, Python; bevorzugt CMD über PowerShell.

---

## Was ist WortRadar?

Eine lokale Sprachlern-App (Streamlit) für **Englisch → Deutsch**. Kernidee:
*Bevor* der Nutzer ein Lied / einen Blogpost / ein Buch konsumiert, analysiert
die App den Text, vergleicht ihn mit dem geschätzten Wortschatz des Nutzers und
prognostiziert das Verständnis ("Coverage") — pro Zeile und pro Wort. Aus jedem
Dokument entsteht ein **Vokabel-Deck** (als editierbare Liste UND als
Flip-Übung mit Spaced Repetition). Eine **Unlock-Mechanik** zeigt, mit wie
wenigen neuen Vokabeln ein Dokument die 98%-Verständnisschwelle erreicht.

Vorbild war **jpdb.io** (gibt es nur für Japanisch) — WortRadar überträgt das
Konzept auf Englisch und auf Musik/eigene Medien.

## Eiserne Designentscheidungen (vom Nutzer bestätigt, NICHT ändern ohne Rückfrage)

1. **Kein automatischer Lyrics-/Inhalts-Download aus dem Netz.** Urheberrecht.
   Der Nutzer bringt eigene Dateien mit oder fügt Text per Copy-Paste ein.
   Alles bleibt lokal in `data/wortradar.db`. Netz-Ausnahmen sind nur:
   (a) die optionale KI-Übersetzung (Gemini/Claude) schickt einzelne Wörter +
   eine Kontextzeile an den Anbieter; (b) MyMemory-Fallback; (c) der
   freiwillige Ein-Klick-Download des GPL-Wörterbuchs. Kein Liedtext/Buchinhalt
   verlässt je den Rechner.
2. **Nur Englisch → Deutsch.** Andere Sprachen bewusst ausgeklammert (wären
   eine `lang`-Spalte + weitere spaCy-Modelle — für später vorgemerkt).
3. **Pro Dokument ein Deck**, verfügbar als Liste und als Flip-Übung.
4. **Unlock-Mechanik** ("Lerne N Wörter → Dokument erreicht 98%") ist ein
   Kernfeature, kein Beiwerk.

## Wissenschaftliche Grundlage (knapp)

- **Lexical Coverage** (Paul Nation): ~98% bekannte Wörter = komfortables
  Verstehen, ~95% = machbar, darunter frustrierend. Die Schwellen TARGET_COMFORT
  und TARGET_OK in `core/coverage.py` bilden das ab.
- **Wortschatzschätzung** per Yes/No-Vocabulary-Test (Meara/LexTALE-Ansatz):
  logistische Kurve `P(bekannt | Zipf-Frequenz) = g + (1-g)·sigmoid(a·(z-b))`.
  Pseudowörter messen die Rate-Quote `g` und korrigieren sie heraus.
- **Spaced Repetition**: FSRS v6 (moderner Anki-Nachfolger).

---

## Architektur / Modulkarte

```
wortradar/
├── app.py                 Streamlit-UI (Deutsch). Sidebar-Navigation +
│                          Doc-Ansicht mit Tabs (Text/Quiz/Deck-Liste/Flip).
├── clickable_text/        HTML/JS-Streamlit-Komponente: rendert die Text-
│   └── index.html         Ansicht, jedes Wort als klickbare Box (siehe unten).
├── core/
│   ├── db.py              SQLite-Schema + alle Zugriffsfunktionen.
│   ├── nlp.py             Tokenisierung + Lemmatisierung. ZWEI Engines mit
│   │                      identischem Interface (siehe Stolperfalle unten);
│   │                      engine_name() liefert den aktiven Pfad.
│   ├── importers.py       parse_pasted / parse_upload für
│   │                      TXT,MD,LRC,SRT,VTT,PDF,EPUB. Liefert
│   │                      list[(line_no, text, t_start|None)].
│   ├── pipeline.py        ingest_document(): Zeilen → NLP → DB.
│   ├── vocab_model.py     Wortschatztest, fit_curve(), p_known_from_zipf(),
│   │                      effective_p(), estimate_vocab_size().
│   ├── coverage.py        doc_stats()/doc_analysis(), Zeilen-Ampel,
│   │                      unlock_words(), quiz_candidates() (QUIZ_P_MIN=0.10,
│   │                      QUIZ_P_MAX=0.92), global_roi_words().
│   ├── srs.py             FSRS-Wrapper. review(), start_learning().
│   ├── auth.py            Supabase-Login (Schritt 1 Infrastruktur). Opt-in:
│   │                      nur aktiv, wenn SUPABASE_URL+SUPABASE_ANON_KEY gesetzt
│   │                      (sonst Dev-Modus ohne Login). login_gate() in app.py
│   │                      sperrt die App; user liegt in st.session_state["user"].
│   │                      NOCH OFFEN: Daten pro user_id (Postgres-Migration),
│   │                      Nutzungslimit + Bezahlung (Modell: 1 Server-KI-Key).
│   ├── dictionary.py      Übersetzungskette: Offline-Ding/dict.cc → KI
│   │                      (kontextbewusst, ai_translate()) → MyMemory-Fallback.
│   │                      Ding-Parser wählt pro Wort die beste Übersetzung
│   │                      (Einzelwort, ohne Klammer-/etw.-Ballast, häufigstes
│   │                      per wordfreq). download_and_import_ding() = Ein-Klick-
│   │                      Download des freien Ding-Wörterbuchs.
│   │                      KI-Anbieter: get_ai_config() liest Key/Modell aus der
│   │                      meta-Tabelle ODER aus env (GROQ_API_KEY/GEMINI_API_KEY/
│   │                      ANTHROPIC_API_KEY, fürs Online-Deployment; app.py
│   │                      spiegelt st.secrets nach os.environ). Vorrang:
│   │                      Groq > Gemini > Claude. groq_translate() (empfohlen,
│   │                      beste Gratis-Limits, GROQ_DEFAULT_MODEL=
│   │                      "llama-3.3-70b-versatile"), gemini_translate()
│   │                      ("gemini-2.0-flash"), llm_translate() (Claude,
│   │                      "claude-haiku-4-5"). context_translate(wort, zeile)
│   │                      löst Kontext-Mehrdeutigkeit (country lane→Landstraße).
│   │                      MyMemory nur Fallback ohne KI (bei Einzelwörtern
│   │                      unzuverlässig → Treffer > 2 Wörter werden verworfen).
│   │                      ai_retranslate() = ganzes Deck im Kontext neu (1 Req/
│   │                      40 W., überschreibt falsche Offline-Treffer). KI-Fehler
│   │                      werfen AITranslationError (klare Meldung, Retry bei
│   │                      429) statt Traceback — Aufrufer fangen das ab.
│   └── pseudowords.py     24 verifizierte Pseudowörter.
├── examples/              Freie Beispieldateien (selbst gedichteter Song .lrc,
│                          Blogpost .txt) — KEINE echten urheberrechtl. Lyrics.
├── .streamlit/config.toml Dunkles Theme.
├── data/wortradar.db      Entsteht zur Laufzeit (nicht eingecheckt).
└── requirements.txt
```

### Datenmodell (SQLite, in db.py)

- `documents(id, title, doc_type, created_at)`
- `lines(doc_id, line_no, text, t_start)` — t_start = Sekunden aus .lrc/.srt
- `lemmas(id, lemma, zipf, is_function)` — eine Grundform, global geteilt
- `tokens(doc_id, line_no, pos, surface, c0, c1, lemma_id, kind)` — c0/c1 =
  Zeichen-Offsets in der Zeile (für farbiges Rendering), kind ∈
  content|function|proper|other
- `knowledge(lemma_id, status, fsrs, due, updated_at)` — status ∈
  known|unknown|learning|ignored|NULL. fsrs = serialisierte FSRS-Karte (JSON).
- `dictionary(lemma, de, source)` — source ∈ ding|dictcc|llm|manuell
- `test_answers(...)`, `meta(key, value)` — Modellparameter & Testpool gecacht.

### Kernlogik der Coverage

- Funktionswörter (the, of, …) und Eigennamen zählen als bekannt (p=1).
- Pro Inhaltslemma: `effective_p(status, zipf)` — expliziter Status schlägt den
  Frequenz-Prior (known/ignored→1, unknown→0, learning→0.5, NULL→Kurvenwert).
- Coverage = Σp / N über alle Tokens.
- Zeilen-Ampel über erwartete unbekannte Wörter pro Zeile:
  grün <0.25, gelb <1.25, sonst rot.
- Eine Vokabel = EINE FSRS-Karte über alle Dokumente. Ab Stabilität ≥ 21 Tage
  graduiert sie automatisch zu status "known" und hebt sofort die Coverage
  aller Texte.
- **Wortfarben im Text-Tab sind BINÄR** (`word_known()` in app.py): grün
  („wr-known") = kenne ich, rot („wr-unknown") = kenne ich nicht. Gefüllte
  Boxen, weißer Text. Regel: status known/ignored→grün, unknown/learning→rot,
  sonst Prior p ≥ 0.5 → grün, sonst rot. (Bewusst KEIN gelb/blau mehr — der
  Nutzer wollte „entweder ich kenne ein Wort oder nicht".) Die Deck-Schwelle
  in `deck_rows()` nutzt dieselbe 0.5-Grenze, damit Text und Deck konsistent
  sind.
- **Wörter anklicken** (Text-Tab, `clickable_text`-Komponente): EIN Klick
  dreht die Farbe um (grün→rot setzt status „unknown" = landet im Deck;
  rot→grün setzt „known"). Die Komponente meldet die aktuell sichtbare Farbe
  als `was_known` mit; Python setzt das Gegenteil (kein Doppelklick-Timing,
  das war mit Rerun unzuverlässig). JEDES Wort hat eine `lemma_id` und ist
  klickbar — Inhalts-, Funktions- UND Eigennamen (pipeline vergibt jetzt auch
  für `kind="proper"` eine lemma_id). Vom Nutzer markierte Funktions-/Eigennamen
  erscheinen über `db.doc_marked_function_lemmas()` (kind IN function,proper)
  additiv im Deck, ohne die Coverage zu verändern (Coverage liest nur
  doc_lemma_summary = kind='content').
- **Markieren von Passagen** (`clickable_text`): Wort/Passage mit der Maus
  markieren → schwebender „❓ kenne ich nicht"-Knopf (`#floatmark`) erscheint
  daneben. Klick sendet `{action:"mark", lemma_ids:[…]}`; Python setzt alle per
  `set_status_bulk(..., "unknown")`. Die Komponente sammelt die Spans per
  `getSelection().containsNode()`. (Gilt für ALT importierte Docs nur nach
  Re-Import — dort hatten Eigennamen noch keine lemma_id.)
- **Decks/Lernen pro Song**: Deck-Liste und Flip-Übung sind je `doc_id`
  gefiltert. „Heute lernen" (`page_learn`) bietet ein Dropdown pro Song
  (`db.due_counts_by_doc()`) plus optional „alle gemischt". Die FSRS-KARTE
  bleibt aber global (eine Vokabel = eine Karte über alle Songs).

---

## STOLPERFALLEN (wichtig, schon einmal aufgetreten)

1. **spaCy-Modell `en_core_web_sm`.** Es gibt ZWEI NLP-Engines in `nlp.py`:
   `_SpacyEngine` (Primärpfad, braucht `python -m spacy download en_core_web_sm`)
   und `_FallbackEngine` (lemminflect + wordfreq, läuft ohne Modell). Beide
   müssen dasselbe Interface `analyze_lines(lines)->list[list[Tok]]` behalten.
   Auf dem Rechner des Nutzers sollte spaCy laufen; Fallback nur als Sicherung.
2. **Streamlit `session_state` + Widget-Keys.** Man darf `st.session_state["nav"]`
   NICHT überschreiben, nachdem das Radio-Widget mit `key="nav"` gebaut wurde.
   Deshalb läuft die Navigation über einen Zwischen-Key `_pending_nav`, der ganz
   oben im Sidebar-Block — VOR dem Radio — angewendet wird. Siehe `goto()` und
   den Sidebar-Block am Ende von app.py. Beim Ändern der Navigation dieses Muster
   beibehalten.
   **Nebeneffekt `_last_nav`:** Ein direkter Sidebar-Klick (kein `goto()`-Aufruf)
   setzt `doc_id` auf `None`, damit die Bibliotheksliste statt der zuletzt
   geöffneten Doc-Ansicht erscheint (`if pending is None and nav != _last_nav`
   im Sidebar-Block). Wer die Sidebar-Logik anfasst, dieses Verhalten
   beibehalten.
3. **Mehrere Python-Installationen (Windows).** Der Nutzer hatte Python 3.10
   (kaputt) neben 3.13. Immer `python -m pip` / `python -m streamlit` empfehlen,
   nie das blanke `pip`/`streamlit`.
   **Neue Streamlit-Komponenten** (`clickable_text/`) werden nur beim
   App-START registriert — nach Änderungen an der Komponente die App komplett
   neu starten, ein Browser-Reload reicht NICHT. Das `Streamlit`-Objekt im
   iframe muss als `postMessage`-Shim selbst definiert sein (sonst Endlos-
   Spinner, weil `componentReady` nie gesendet wird). Die iframe-Höhe per
   `setFrameHeight` mehrfach setzen (sofort + rAF + Fallback + ResizeObserver),
   sonst bleibt der Text unsichtbar.
4. **Deutsche Anführungszeichen in f-Strings.** In `st.toast(f"…")` o.ä. KEINE
   geraden `"` innerhalb des f-Strings verwenden — typografische „…“ nehmen,
   sonst SyntaxError.
5. **PDF/EPUB-Import**: PyMuPDF (`import fitz`) für PDF; EPUB über stdlib
   `zipfile` + HTMLParser. Silbentrennung am Zeilenende wird zusammengefügt.

## Setup-Kommandos (Windows CMD)

```bat
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m streamlit run app.py
```

## Wörterbuch (vom Nutzer selbst zu laden)

Freie Ding-Liste der TU Chemnitz (GPL):
`https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en/de-en.txt.gz`
→ in der App unter Einstellungen hochladen (die .gz geht direkt). Alternativ
persönlicher dict.cc-Export (TSV). Auto-Format-Erkennung in dictionary.py.

---

## Offene Punkte / Nächste Schritte (Stand letzte Sitzung)

- **Wortschatztest „Genau-Modus" (DISKUTIERT, NOCH NICHT GEBAUT):** Der Nutzer
  fragte, ob der Test statt nur „Kenne ich / Kenne ich nicht" auch direkt nach
  der **Bedeutung** fragen sollte. Konsens: Das schnelle Yes/No-Format bleibt
  als Kalibrierung (breit, schnell, durch Pseudowörter ratekorrigiert) — die
  echte Bedeutungs-Abfrage passiert ohnehin im Flip-Deck. Mögliche Erweiterung:
  ein optionaler Umschalter „Schnell / Genau" am Teststart, der im Genau-Modus
  Multiple-Choice mit der deutschen Bedeutung anbietet. **Wenn der Nutzer das
  will, hier umsetzen.** Quiz-Kandidaten kämen aus `quiz_candidates()`, die
  deutschen Distraktoren aus dem Wörterbuch (zufällige andere Lemmata ähnlicher
  Zipf-Frequenz).
- Mehrwort-Ausdrücke (give up, look forward to) werden noch als Einzelwörter
  behandelt — bewusst, evtl. später.
- Performance bei sehr großen Büchern (>100k Wörter) noch nicht optimiert.

## Verhaltenshinweise für die Arbeit an diesem Projekt

- Auf Deutsch antworten.
- Vor Änderungen an Coverage-Mathematik oder Datenmodell kurz Rücksprache —
  daran hängt viel.
- Kleine Tests headless laufen lassen (es gibt keinen UI-Testrunner): Module
  lassen sich direkt importieren und mit einer Test-DB prüfen
  (`db.DB_PATH = db.DATA_DIR / "test.db"`).
- Die vier eisernen Designentscheidungen oben respektieren.
