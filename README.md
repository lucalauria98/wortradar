# 📡 WortRadar

**Verstehen, bevor du anfängst.** WortRadar analysiert englische Songtexte,
Blogposts, Untertitel und Bücher, vergleicht sie mit *deinem* Wortschatz und
sagt dir vorher: *„Diese Zeile wirst du verstehen, diese nicht — und das sind
die 12 Vokabeln, mit denen du den Song freischaltest."*

Pro Dokument entsteht automatisch ein **Vokabel-Deck** (als Liste und als
Flip-Übung mit Spaced Repetition). Nur **Englisch → Deutsch**.

---

## Schnellstart (Windows, Command Prompt)

```bat
cd wortradar
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m streamlit run app.py
```

Der Browser öffnet sich automatisch (sonst: http://localhost:8501).

## Online stellen (Streamlit Community Cloud)

1. Repo zu GitHub pushen, auf [share.streamlit.io](https://share.streamlit.io)
   ein neues App-Deployment auf `app.py` anlegen.
2. Unter **App → Settings → Secrets** hinterlegen, z. B.:
   ```toml
   GROQ_API_KEY = "gsk_…"          # ein Server-Key für alle Besucher
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_ANON_KEY = "eyJ…"      # aktiviert den Login
   ```
   `app.py` spiegelt Secrets automatisch in Umgebungsvariablen.
   - **KI:** ein Server-Key nutzt alle Besucher (Vorrang Groq > Gemini > Claude;
     auch `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` möglich).
   - **Login:** Sobald `SUPABASE_URL` + `SUPABASE_ANON_KEY` gesetzt sind, ist der
     Login aktiv. Ohne sie läuft die App im Einzelnutzer-Modus (gut für lokal).
3. Kostenlosen Groq-Key: https://console.groq.com/keys ·
   Supabase-Projekt + Keys: https://supabase.com (Project Settings → API).
   Tipp: In Supabase unter *Authentication → Providers → Email* die
   E-Mail-Bestätigung für den Start optional deaktivieren (sofortiger Login).

> **Datenhaltung:** `data/wortradar.db` liegt lokal und ist auf der kostenlosen
> Streamlit-Cloud **flüchtig** (wird bei Neustart/Redeploy zurückgesetzt). Für
> dauerhaften Nutzerdaten-Speicher später eine gehostete DB (z. B. Postgres/
> Supabase) anbinden — fürs erste Ausprobieren reicht SQLite.

> **Hinweis:** Ohne das spaCy-Modell läuft die App trotzdem — dann mit einem
> einfacheren Fallback-Lemmatisierer (lemminflect). Für die beste Qualität
> das Modell installieren und die App neu starten.

## Die ersten 10 Minuten

1. **🧪 Wortschatztest** machen (~3 Minuten, Ja/Nein-Klicks). Daraus wird
   eine persönliche Kenn-Wahrscheinlichkeit für *jedes* englische Wort
   geschätzt. Achtung: erfundene Wörter sind dabei — ehrlich bleiben!
2. **📖 Wörterbuch laden** (einmalig, optional aber empfohlen):
   `https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en/de-en.txt.gz`
   herunterladen und unter *Einstellungen* hochladen (die `.gz` direkt,
   kein Entpacken nötig). ~Hunderttausende freie Einträge (GPL).
3. **📚 Importieren:** Text einfügen oder Datei hochladen — zum Ausprobieren
   liegen in `examples/` ein selbstgeschriebener Demo-Song (`.lrc` mit
   Zeitstempeln) und ein Blogpost.
4. Im Dokument: **Text** (eingefärbte Prognose), **Schnell-Quiz** (macht die
   Prognose scharf), **Deck-Liste**, **Deck-Lernen** (Flip-Karten).
5. **🔓 Unlocks** zeigt, welches Dokument du mit den wenigsten neuen
   Vokabeln auf 98 % Verständnis bringst.

## Was bedeuten die Farben?

| Anzeige | Bedeutung |
|---|---|
| 🟢 Zeilenrand | Zeile verstehst du sehr wahrscheinlich komplett |
| 🟡 Zeilenrand | vermutlich ~1 unbekanntes Wort |
| 🔴 Zeilenrand | mehrere unbekannte Wörter |
| <span style="color:#ff5d5d">rotes Wort</span> | kennst du sehr wahrscheinlich nicht |
| <span style="color:#ffb454">gelbes Wort</span> | unsicher — das Schnell-Quiz klärt es |
| <span style="color:#4cc2ff">blaues Wort</span> | gerade im Lernen |

**Coverage-Schwellen** (aus der Lexical-Coverage-Forschung, u. a. Paul
Nation): ab **98 %** liest/hört sich ein Text komfortabel, ab **95 %**
machbar, darunter frustrierend. Genau dahin optimiert die Unlock-Mechanik.

## Übersetzungen — der Reihe nach

1. **Offline-Wörterbuch** (freie Ding-Liste, **Ein-Klick-Download** unter
   *Einstellungen*) — gratis, lokal, deckt das meiste ab. Sobald geladen,
   erscheinen Übersetzungen im Deck **automatisch**. *Empfohlen, Hauptquelle.*
2. **KI im Kontext** (optional): übersetzt mehrdeutige Wörter **mit der
   Songzeile als Kontext** — löst z. B. *„country lane → Landstraße"* statt
   „Gasse". Empfohlen: **Google Gemini Flash** (kostenloser Tarif, keine
   Kreditkarte) — ideal fürs Online-Stellen, da ein Server-Key für alle
   Besucher reicht. Alternativ Claude. Key unter *Einstellungen* oder per
   Umgebungsvariable `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`.
3. **MyMemory** (gratis, kein Account): Fallback, wenn keine KI hinterlegt ist.
   Ohne Kontext, bei Einzelwörtern weniger treffsicher.
4. **Manuell:** jede Übersetzung ist im Deck/auf der Lernkarte direkt editierbar.

Im Deck arbeitet **„📖 Übersetzungen ergänzen"** diese Quellen ab; auf jeder
Lernkarte korrigiert **„🔄 Im Kontext übersetzen"** eine unpassende Bedeutung.

## Design-Entscheidung: kein automatischer Lyrics-Download

WortRadar lädt **bewusst keine** Songtexte oder Buchinhalte aus dem
Internet. Songtexte sind urheberrechtlich geschützt; automatisches Abrufen
und Speichern wäre rechtlich heikel. Stattdessen gilt das „LingQ-Prinzip":
**du bringst Inhalte mit, die du besitzt** (gekaufte Songs mit `.lrc`-Datei,
eigene E-Books/PDFs, kopierte Artikel) — per Copy-Paste oder Datei-Upload.
Alles bleibt lokal in `data/wortradar.db`. Nichts wird hochgeladen
(einzige Ausnahme: die optionale Claude-Übersetzung sendet einzelne Wörter
plus eine Kontextzeile an die Anthropic-API).

## Wie funktioniert die Prognose? (Kurzfassung)

- **Lemmatisierung:** jedes Wort wird auf seine Grundform reduziert
  (*singing → sing*), via spaCy (oder lemminflect-Fallback).
- **Frequenz-Prior:** aus dem Wortschatztest wird eine logistische Kurve
  `P(bekannt | Zipf-Häufigkeit)` gefittet — inklusive Rate-Korrektur über
  Pseudowörter (Ansatz wie bei LexTALE).
- **Explizites Wissen schlägt Prior:** jede Quiz-Antwort und jede Lernkarte
  überschreibt die Schätzung für genau dieses Wort.
- **Coverage** = erwarteter Anteil verstandener Wörter; Funktionswörter
  (the, of, …) und Eigennamen zählen als bekannt.
- **Spaced Repetition:** FSRS v6 (moderner Anki-Nachfolger). Eine Vokabel =
  eine Karte über alle Dokumente; ab 21 Tagen Stabilität gilt sie als
  dauerhaft *bekannt* und verbessert sofort die Coverage aller Texte.

## Projektstruktur

```
wortradar/
├── app.py                 Streamlit-Oberfläche (Deutsch)
├── core/
│   ├── db.py              SQLite-Schema & Zugriff
│   ├── nlp.py             spaCy-/Fallback-Engine, Funktionswörter
│   ├── importers.py       TXT/MD/LRC/SRT/VTT/PDF/EPUB + Copy-Paste
│   ├── pipeline.py        Import → NLP → Datenbank
│   ├── vocab_model.py     Wortschatztest, logistischer Fit, p_known
│   ├── coverage.py        Coverage, Zeilen-Ampel, Unlocks, ROI
│   ├── srs.py             FSRS-Wrapper (Lernkarten)
│   ├── dictionary.py      Ding-/dict.cc-Import, Claude-Übersetzung
│   └── pseudowords.py     verifizierte Pseudowörter für den Test
├── examples/              freie Beispieldateien (Song .lrc, Blogpost)
├── data/                  entsteht zur Laufzeit (wortradar.db)
└── requirements.txt
```

## Bekannte Grenzen des Prototyps

- Nur Englisch→Deutsch (bewusst — Mehrsprachigkeit wäre eine `lang`-Spalte
  plus weitere spaCy-Modelle, ist aber für später vorgesehen).
- Mehrwort-Ausdrücke (*give up*, *look forward to*) werden als Einzelwörter
  behandelt.
- Die Wortschatz-Schätzung ist eine grobe, ehrliche Näherung — sie wird mit
  jedem Quiz und jeder Lernkarte präziser.
- Sehr große Bücher (>100k Wörter) brauchen beim Import etwas Geduld.

Viel Spaß beim Freischalten! 🔓
