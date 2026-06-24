"""WortRadar - Verstehen, bevor du anfaengst.

Streamlit-App: Songtexte, Blogposts, Buecher importieren (eigene Dateien
oder Copy-Paste), Verstaendnis-Prognose pro Zeile sehen, unbekannte
Vokabeln VOR dem Hoeren/Lesen lernen - mit Spaced Repetition (FSRS).

Start:  streamlit run app.py
"""
from __future__ import annotations

import gzip
import html
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc

st.set_page_config(page_title="WortRadar", page_icon="📡", layout="wide")

from fsrs import Rating  # noqa: E402

from core import auth, db, dictionary, srs  # noqa: E402
from core import vocab_model as vm  # noqa: E402
from core.coverage import (  # noqa: E402
    doc_analysis, doc_stats, global_roi_words, quiz_candidates, unlock_words,
)
from core.importers import parse_pasted, parse_upload  # noqa: E402
from core.nlp import engine_name  # noqa: E402
from core.pipeline import ingest_document  # noqa: E402

@st.cache_resource
def _ensure_db():
    """Schema nur EINMAL pro Prozess anlegen - nicht bei jedem Rerun.
    Spart bei Postgres pro Klick mehrere CREATE-TABLE-Roundtrips."""
    db.init_db()
    return True


_ensure_db()

# Online-Deployment: in Streamlit Secrets hinterlegte Keys auch als
# Umgebungsvariablen bereitstellen (dictionary.get_ai_config liest os.environ).
# So reicht EIN Server-Key fuer alle Besucher - niemand braucht einen Account.
import os  # noqa: E402
try:
    for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
               "SUPABASE_URL", "SUPABASE_ANON_KEY", "DATABASE_URL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:  # noqa: BLE001 - keine secrets.toml lokal: einfach ignorieren
    pass

_clickable_text = stc.declare_component(
    "clickable_text",
    path=str(Path(__file__).parent / "clickable_text"),
)

# ------------------------------------------------------------------ CSS ----
AMPEL = {"gruen": "#3ddc84", "gelb": "#ffb454", "rot": "#ff5d5d"}
st.markdown("""
<style>
.wr-line {
    border-left: 4px solid #444;
    padding: 2px 10px;
    margin: 2px 0;
    line-height: 1.7;
    font-size: 1.02rem;
}
.wr-ts { color: #777; font-family: monospace; font-size: 0.8rem; margin-right: 8px; }
.wr-unk   { color: #ff5d5d; font-weight: 600; }
.wr-half  { color: #ffb454; border-bottom: 1px dotted #ffb454; }
.wr-learn { color: #4cc2ff; font-weight: 600; }
.wr-card {
    text-align: center; padding: 1.2rem 0 0.4rem 0;
}
.wr-card h1 { font-size: 2.6rem; margin-bottom: 0.4rem; }
.wr-ctx { color: #999; font-style: italic; }
.wr-de { color: #3ddc84; font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

DOC_TYPES = {"song": "🎵 Song", "blog": "📰 Blog/Artikel", "buch": "📖 Buch",
             "film": "🎬 Film/Untertitel", "text": "📄 Sonstiges"}


# ------------------------------------------------------------- Helfer ----
def fmt_ts(t: float | None) -> str:
    if t is None:
        return ""
    m, s = divmod(int(t), 60)
    return f'<span class="wr-ts">[{m:02d}:{s:02d}]</span>'


def word_known(tok: dict) -> bool:
    """Binaer: kenne ich das Wort (gruen) oder nicht (rot)?
    Expliziter Status schlaegt die Frequenz-Prognose."""
    st = tok["status"]
    if st in ("known", "ignored"):
        return True
    if st in ("unknown", "learning"):
        return False
    return tok["p"] >= 0.5


def render_line(lv, translations: dict[str, str]) -> str:
    text = lv.text
    parts, cur = [], 0
    for t in sorted(lv.tokens, key=lambda x: x["c0"]):
        parts.append(html.escape(text[cur:t["c0"]]))
        surf = html.escape(t["surface"])
        lid  = t.get("lemma_id")
        cls  = "wr-known" if word_known(t) else "wr-unknown"  # gruen oder rot

        if lid:   # Inhalts- UND Funktionswoerter haben eine lemma_id -> klickbar
            tip = t["lemma"] or ""
            de  = translations.get(t["lemma"] or "")
            if de:
                tip += f" → {de}"
            parts.append(
                f'<span class="wr-word {cls}" title="{html.escape(tip)}" '
                f'data-lid="{lid}">{surf}</span>')
        else:     # Eigennamen: keine lemma_id -> immer gruen, nicht klickbar
            parts.append(f'<span class="wr-word wr-known">{surf}</span>')
        cur = t["c1"]
    parts.append(html.escape(text[cur:]))
    color = AMPEL[lv.ampel]
    return (f'<div class="wr-line" style="border-left-color:{color}">'
            f'{fmt_ts(lv.t_start)}{"".join(parts)}</div>')


def goto(page: str, doc_id: int | None = None):
    # Nicht direkt den Widget-Key "nav" setzen (das verbietet Streamlit, sobald
    # das Radio gebaut ist). Stattdessen einen Wunsch hinterlegen, der ganz oben
    # im naechsten Lauf - vor dem Radio - angewendet wird.
    st.session_state["_pending_nav"] = page
    st.session_state["doc_id"] = doc_id


def coverage_label(c: float) -> str:
    if c >= 0.98:
        return "🟢 komfortabel"
    if c >= 0.95:
        return "🟡 machbar"
    return "🔴 anstrengend"


def calib_warning():
    if not vm.is_calibrated():
        st.warning(
            "**Noch unkalibriert.** Alle Prognosen nutzen einen Durchschnitts-"
            "Lerner. Mach den ~3-Minuten-Wortschatztest, damit WortRadar "
            "DEINEN Wortschatz kennt.", icon="🧪",
        )
        st.button("🧪 Jetzt Wortschatztest machen",
                  on_click=goto, args=("🧪 Wortschatztest",))


# ------------------------------------------------------------ Import ----
def import_panel():
    with st.expander("➕ Neues Dokument importieren",
                     expanded=not db.list_documents()):
        st.caption(
            "Nur **eigene** Texte und Dateien verwenden (gekaufte Songs mit "
            ".lrc-Datei, eigene E-Books, kopierte Artikel …). WortRadar laedt "
            "nichts aus dem Netz - alles bleibt lokal auf deinem Rechner.")
        tab_paste, tab_file = st.tabs(["📋 Text einfuegen", "📁 Datei hochladen"])

        with tab_paste:
            c1, c2 = st.columns([3, 1])
            title = c1.text_input("Titel", key="imp_title",
                                  placeholder="z. B. Midnight Rain")
            dtype = c2.selectbox("Typ", list(DOC_TYPES), key="imp_type",
                                 format_func=lambda k: DOC_TYPES[k])
            text = st.text_area(
                "Englischer Text (LRC-Zeitstempel werden automatisch erkannt)",
                height=180, key="imp_text")
            if st.button("Analysieren & speichern", type="primary",
                         disabled=not text.strip()):
                with st.spinner("Analysiere Woerter …"):
                    lines = parse_pasted(text)
                    doc_id = ingest_document(title or "Ohne Titel", dtype, lines)
                goto("📚 Bibliothek", doc_id)
                st.rerun()

        with tab_file:
            up = st.file_uploader(
                "Datei (.txt, .md, .lrc, .srt, .vtt, .pdf, .epub)",
                type=["txt", "md", "lrc", "srt", "vtt", "pdf", "epub"])
            c1, c2 = st.columns([3, 1])
            ftitle = c1.text_input(
                "Titel", key="imp_ftitle",
                value=(up.name.rsplit(".", 1)[0] if up else ""))
            fdtype = c2.selectbox("Typ ", list(DOC_TYPES), key="imp_ftype",
                                  format_func=lambda k: DOC_TYPES[k])
            if up and st.button("Datei analysieren & speichern", type="primary"):
                with st.spinner("Lese Datei und analysiere Woerter …"):
                    lines = parse_upload(up.name, up.getvalue())
                    doc_id = ingest_document(ftitle or up.name, fdtype, lines)
                goto("📚 Bibliothek", doc_id)
                st.rerun()


# --------------------------------------------------------- Bibliothek ----
def page_library():
    doc_id = st.session_state.get("doc_id")
    if doc_id:
        doc_view(doc_id)
        return

    st.title("📡 WortRadar")
    st.caption("Verstehen, bevor du anfaengst: Coverage-Prognose und "
               "Vokabel-Decks fuer deine Songs, Artikel und Buecher.")
    calib_warning()
    import_panel()

    docs = db.list_documents()
    if not docs:
        st.info("Noch keine Dokumente. Importiere oben deinen ersten Text - "
                "im Ordner `examples/` liegen zwei Beispieldateien.")
        return

    st.subheader("Deine Bibliothek")
    for d in docs:
        s = doc_stats(d["id"])
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([4, 2, 2, 1.2])
            c1.markdown(f"**{DOC_TYPES.get(d['doc_type'], '📄')}  {d['title']}**")
            c1.progress(min(s.coverage, 1.0),
                        text=f"Verstaendnis: {s.coverage:.1%}  ·  "
                             f"{coverage_label(s.coverage)}")
            c2.metric("Bis 98 %", f"{s.words_to_98} Woerter")
            c3.metric("Im Deck", s.n_unknown + s.n_learning)
            c4.button("Oeffnen", key=f"open{d['id']}", type="primary",
                      on_click=goto, args=("📚 Bibliothek", d["id"]))


# --------------------------------------------------------- Doc-Ansicht ----
def doc_view(doc_id: int):
    doc = db.get_document(doc_id)
    if not doc:
        st.session_state["doc_id"] = None
        st.rerun()
        return

    top = st.columns([1, 6, 1.4])
    top[0].button("← Zurueck", on_click=goto, args=("📚 Bibliothek", None))
    top[1].markdown(f"## {DOC_TYPES.get(doc['doc_type'], '📄')} {doc['title']}")
    if top[2].button("🗑️ Loeschen", key="del_doc"):
        st.session_state["confirm_del"] = True
    if st.session_state.get("confirm_del"):
        st.error("Dokument wirklich loeschen? (Lernfortschritt der Woerter bleibt erhalten.)")
        cc = st.columns([1, 1, 6])
        if cc[0].button("Ja, loeschen", type="primary"):
            db.delete_document(doc_id)
            st.session_state["confirm_del"] = False
            goto("📚 Bibliothek", None)
            st.rerun()
        if cc[1].button("Abbrechen"):
            st.session_state["confirm_del"] = False
            st.rerun()

    calib_warning()
    s = doc_stats(doc_id)
    m = st.columns(4)
    m[0].metric("Verstaendnis", f"{s.coverage:.1%}",
                help="Erwarteter Anteil der Woerter, die du kennst "
                     "(98 % = komfortabel, 95 % = machbar).")
    m[1].metric("Bis 98 %", f"{s.words_to_98} Woerter",
                help="So viele Vokabeln musst du noch lernen, bis dieses "
                     "Dokument komfortabel verstehbar ist.")
    m[2].metric("Bis 95 %", f"{s.words_to_95} Woerter")
    m[3].metric("Vokabeln im Deck", s.n_unknown + s.n_learning)
    st.caption(f"{s.n_content_lemmas} Inhaltswoerter gesamt · "
               f"{s.n_known} bekannt · {s.n_learning} im Lernen · "
               f"{s.n_unknown} unbekannt · {s.n_untested} ungetestet")

    tabs = st.tabs(["📖 Text", "❓ Schnell-Quiz", "📋 Deck - Liste",
                    "🃏 Deck - Lernen"])
    with tabs[0]:
        text_tab(doc_id)
    with tabs[1]:
        quiz_tab(doc_id)
    with tabs[2]:
        deck_list_tab(doc_id)
    with tabs[3]:
        flashcards(doc_id)


def text_tab(doc_id: int):
    st.caption(
        "🟢 kennst du · 🔴 kennst du nicht (landet im Lern-Deck).  "
        "**Wort anklicken** dreht die Farbe um.  **Wort oder ganze Passage "
        "markieren** → „❓ kenne ich nicht“-Knopf erscheint daneben.")
    with st.spinner("Berechne Prognose …"):
        lines, _ = doc_analysis(doc_id)
        translations = {r["lemma"]: r["translation"]
                        for r in db.doc_lemma_summary(doc_id) if r["translation"]}
    PAGE = 200
    pg = 0
    if len(lines) > PAGE:
        n_pages = (len(lines) - 1) // PAGE + 1
        pg = st.number_input(f"Seite (1-{n_pages})", 1, n_pages, 1) - 1
        lines = lines[pg * PAGE:(pg + 1) * PAGE]

    all_html = "".join(render_line(lv, translations) for lv in lines)
    result = _clickable_text(html=all_html, key=f"ct_{doc_id}_{pg}")

    # Jede Aktion nur einmal verarbeiten (click_id verhindert Doppel-Ausfuehrung
    # nach dem st.rerun(), da der Komponentenwert bis zur naechsten Aktion bleibt).
    ck = f"_ct_id_{doc_id}_{pg}"
    if result and result.get("click_id") != st.session_state.get(ck):
        st.session_state[ck] = result["click_id"]
        if result.get("action") == "mark":
            # Markierte Auswahl (Wort/Passage) komplett als unbekannt setzen
            ids = [int(i) for i in result.get("lemma_ids", [])]
            if ids:
                db.set_status_bulk(ids, "unknown")
                st.toast(f"{len(ids)} Wörter als „kenne ich nicht“ markiert.")
        else:
            # Einzelklick dreht die Farbe um (Komponente meldet sichtbare Farbe).
            db.set_status(result["lemma_id"],
                          "unknown" if result.get("was_known") else "known")
        st.rerun()


# ------------------------------------------------------------- Quiz ----
def quiz_tab(doc_id: int):
    key = f"quiz_{doc_id}"
    qs = st.session_state.get(key)
    if qs is None:
        cands = quiz_candidates(doc_id)
        if not cands:
            st.success("Nichts zu klaeren - bei allen Woertern ist sich das "
                       "Modell sicher (oder du hast sie schon beantwortet).")
            return
        st.markdown(
            f"Das Modell ist sich bei **{len(cands)} Woertern** unsicher, ob "
            "du sie kennst. Ein schnelles Ja/Nein macht Prognose und Deck "
            "praeziser - ohne Schummeln, es geht nur um dich. 😉")
        if st.button("Quiz starten", type="primary"):
            st.session_state[key] = {"items": cands, "i": 0, "yes": 0}
            st.rerun()
        return

    items, i = qs["items"], qs["i"]
    if i >= len(items):
        st.success(f"Fertig! {qs['yes']} von {len(items)} kanntest du schon. "
                   "Prognose und Deck sind aktualisiert.")
        if st.button("Quiz schliessen"):
            del st.session_state[key]
            st.rerun()
        return

    it = items[i]
    st.progress((i) / len(items), text=f"Wort {i + 1} von {len(items)}")
    ctx = db.lemma_first_context(it["lemma_id"], doc_id)
    st.markdown(f'<div class="wr-card"><h1>{html.escape(it["lemma"])}</h1>'
                f'<p class="wr-ctx">{html.escape(ctx["text"]) if ctx else ""}</p></div>',
                unsafe_allow_html=True)

    def answer(known: bool):
        db.set_status(it["lemma_id"], "known" if known else "unknown")
        qs["i"] += 1
        qs["yes"] += int(known)

    c = st.columns([1, 1, 1])
    c[0].button("✅ Kenne ich", key=f"y{i}", on_click=answer, args=(True,),
                width="stretch")
    c[1].button("❌ Kenne ich nicht", key=f"n{i}", on_click=answer, args=(False,),
                width="stretch")
    c[2].button("Ueberspringen", key=f"s{i}",
                on_click=lambda: qs.update(i=qs["i"] + 1),
                width="stretch")


# -------------------------------------------------------- Deck: Liste ----
STATUS_DE = {"unknown": "unbekannt", "learning": "im Lernen",
             "known": "bekannt", "ignored": "ignorieren", None: "ungetestet"}
STATUS_FROM_DE = {v: k for k, v in STATUS_DE.items()}


def deck_rows(doc_id: int, include_unsure: bool) -> list[dict]:
    params = vm.load_model()
    rows = []
    # Ein Lemma kann sowohl als Inhaltswort als auch (vom Nutzer markiert) als
    # Funktions-/Eigenname im selben Doc auftauchen -> selbe globale lemma_id.
    # Im Deck (= eine Vokabel = eine Karte) darf es nur EINMAL stehen, sonst
    # bekommt der DataFrame-Index Duplikate und data_editor wirft beim
    # Rueckschreiben "truth value of a Series is ambiguous".
    seen: set[int] = set()
    for r in db.doc_lemma_summary(doc_id):
        p = vm.effective_p(r["status"], r["zipf"] or 0.0, params)
        # Gleiche Schwelle wie die rote Wortfarbe im Text (p < 0.5 = "kenne
        # ich nicht"), damit Text und Deck konsistent sind.
        in_deck = r["status"] in ("unknown", "learning") or (
            include_unsure and r["status"] is None and p < 0.5)
        if in_deck:
            seen.add(r["lemma_id"])
            rows.append({
                "lemma_id": r["lemma_id"], "Wort": r["lemma"],
                "Uebersetzung": r["translation"] or "",
                "Anzahl": r["cnt"], "Zipf": round(r["zipf"] or 0, 1),
                "Status": STATUS_DE[r["status"]],
            })
    # Vom Nutzer explizit angeklickte Funktionswoerter zusaetzlich aufnehmen
    # (Coverage zaehlt sie weiter als bekannt - hier nur fuers Lernen).
    for r in db.doc_marked_function_lemmas(doc_id):
        if r["lemma_id"] in seen:
            continue
        seen.add(r["lemma_id"])
        rows.append({
            "lemma_id": r["lemma_id"], "Wort": r["lemma"],
            "Uebersetzung": r["translation"] or "",
            "Anzahl": r["cnt"], "Zipf": round(r["zipf"] or 0, 1),
            "Status": STATUS_DE[r["status"]],
        })
    return rows


def deck_list_tab(doc_id: int):
    # Persistente Ergebnismeldung der letzten Uebersetzung (ueberlebt st.rerun)
    msg = st.session_state.pop(f"tr_msg_{doc_id}", None)
    if msg:
        parts = []
        if msg["offline"]:
            parts.append(f"{msg['offline']}× Offline-Woerterbuch")
        if msg.get("ai"):
            parts.append(f"{msg['ai']}× KI (im Kontext)")
        if msg["mymemory"]:
            parts.append(f"{msg['mymemory']}× MyMemory")
        if msg["total_new"]:
            st.success(f"✅ {msg['total_new']} Uebersetzungen ergaenzt "
                       f"({', '.join(parts)}).")
        if msg["error"]:
            st.warning(msg["error"])
        elif msg["missing"]:
            st.info(f"ℹ️ {msg['missing']} Woerter ohne Treffer "
                    "(meist Eigennamen). Mit kostenlosem Gemini-Key in den "
                    "Einstellungen werden auch diese im Kontext uebersetzt.")
        elif not msg["total_new"]:
            st.info("Alle angezeigten Woerter hatten bereits eine Uebersetzung.")

    include_unsure = st.toggle(
        "Auch ungetestete, vermutlich unbekannte Woerter zeigen", value=True,
        key=f"unsure_{doc_id}")
    rows = deck_rows(doc_id, include_unsure)
    if not rows:
        st.success("Leeres Deck - du kennst hier (vermutlich) alles. 🎉")
        return

    st.caption(f"**{len(rows)} Vokabeln.** Uebersetzung und Status sind direkt "
               "editierbar (Status *ignorieren* z. B. fuer Namen).")

    # Ohne konfigurierte KI gibt es keine (guten) Uebersetzungen -> kurz
    # hinweisen, wie man sie aktiviert (kostenloser Groq-Key).
    if not dictionary.get_ai_config()["provider"] and db.dict_size() == 0:
        st.info("💡 Für Übersetzungen einmalig einen **kostenlosen Groq-Key** "
                "unter ⚙️ Einstellungen hinterlegen – dann übersetzt die KI "
                "deine Vokabeln im Kontext.")

    def _bump_deck():
        # data_editor nach DB-Update neu aufbauen, sonst zeigt Streamlit den
        # alten Widget-Zustand -> Versionszaehler im Key erhoehen.
        st.session_state[f"deck_ver_{doc_id}"] = \
            st.session_state.get(f"deck_ver_{doc_id}", 0) + 1

    ai_on = dictionary.get_ai_config()["provider"] is not None
    c1, c2, c3 = st.columns(3)
    if c1.button("📖 Fehlende ergaenzen", key=f"tr_{doc_id}",
                 help="Fuellt nur LEERE Felder: Offline-Woerterbuch, dann "
                      "(falls konfiguriert) KI im Kontext, sonst MyMemory."):
        need = [{"lemma": r["Wort"], "translation": r["Uebersetzung"]}
                for r in rows]
        ctx = {}
        for r in rows:
            if not r["Uebersetzung"]:
                c = db.lemma_first_context(r["lemma_id"], doc_id)
                ctx[r["Wort"]] = c["text"] if c else ""
        with st.spinner("Uebersetze …"):
            res = dictionary.translate_missing(need, ctx)
        st.session_state[f"tr_msg_{doc_id}"] = res
        _bump_deck()
        st.rerun()

    # KI-Komplettkorrektur: uebersetzt ALLE Deckwoerter im Kontext neu und
    # ueberschreibt auch falsche Offline-Treffer ("lane" -> "Gasse" -> "Fahrbahn").
    if ai_on and c2.button("🔄 Alle im Kontext (KI)", key=f"reai_{doc_id}",
                           help="Uebersetzt das ganze Deck per KI mit der "
                                "jeweiligen Songzeile neu - korrigiert auch "
                                "unpassende Woerterbuch-Treffer. Eine Anfrage "
                                "pro 40 Woerter."):
        wc = []
        for r in rows:
            c = db.lemma_first_context(r["lemma_id"], doc_id)
            wc.append((r["Wort"], c["text"] if c else ""))
        with st.spinner("KI uebersetzt das Deck im Kontext …"):
            n, err = dictionary.ai_retranslate(wc)
        st.session_state[f"tr_msg_{doc_id}"] = {
            "offline": 0, "ai": n, "mymemory": 0, "missing": 0,
            "total_new": n, "error": err}
        _bump_deck()
        st.rerun()

    new_ids = [r["lemma_id"] for r in rows if r["Status"] == "ungetestet"]
    if new_ids and c3.button(f"Alle {len(new_ids)} ins Deck",
                             key=f"all_{doc_id}",
                             help="Alle ungetesteten Woerter als „unbekannt“ "
                                  "markieren (ins Lern-Deck)."):
        db.set_status_bulk(new_ids, "unknown")
        st.rerun()

    ver = st.session_state.get(f"deck_ver_{doc_id}", 0)
    df = pd.DataFrame(rows).set_index("lemma_id")
    edited = st.data_editor(
        df, hide_index=True, width="stretch", key=f"de_{doc_id}_{ver}",
        column_config={
            "Wort": st.column_config.TextColumn(disabled=True),
            "Anzahl": st.column_config.NumberColumn(
                disabled=True, help="Vorkommen in diesem Dokument"),
            "Zipf": st.column_config.NumberColumn(
                disabled=True, help="Haeufigkeit im Englischen (7 = sehr haeufig)"),
            "Status": st.column_config.SelectboxColumn(
                options=list(STATUS_FROM_DE), required=True),
        })

    # Aenderungen zuruecksschreiben
    changed_tr, changed_st = {}, {}
    for lid in df.index:
        if edited.loc[lid, "Uebersetzung"] != df.loc[lid, "Uebersetzung"]:
            changed_tr[df.loc[lid, "Wort"]] = str(edited.loc[lid, "Uebersetzung"])
        if edited.loc[lid, "Status"] != df.loc[lid, "Status"]:
            changed_st[int(lid)] = STATUS_FROM_DE[edited.loc[lid, "Status"]]
    if changed_tr:
        db.dict_store(changed_tr, "manuell")
    for lid, stat in changed_st.items():
        db.set_status(lid, stat)
    if changed_tr or changed_st:
        st.rerun()


# --------------------------------------------------------- Flashcards ----
def flashcards(doc_id: int | None):
    cards = db.due_cards(doc_id)
    if not cards:
        st.success("Keine faelligen Karten. 🎉 Komm spaeter wieder - FSRS "
                   "meldet sich, kurz bevor du ein Wort vergessen wuerdest.")
        return

    card = cards[0]
    skey = f"flash_show_{doc_id}"   # Aufdeck-Zustand pro Song getrennt
    st.caption(f"**{len(cards)} Karten** faellig" +
               ("" if doc_id else " (alle Songs gemischt)"))

    ctx = db.lemma_first_context(card["lemma_id"], doc_id)
    ctx_text = ctx["text"] if ctx else ""
    cloze = re.sub(re.escape(ctx["surface"]), "_____", ctx_text,
                   flags=re.IGNORECASE) if ctx else ""

    with st.container(border=True):
        if not st.session_state.get(skey):
            st.markdown(
                f'<div class="wr-card"><h1>{html.escape(card["lemma"])}</h1>'
                f'<p class="wr-ctx">{html.escape(cloze)}</p></div>',
                unsafe_allow_html=True)
            if st.button("Aufdecken", type="primary", width="stretch"):
                st.session_state[skey] = True
                st.rerun()
        else:
            de = card["translation"]
            st.markdown(
                f'<div class="wr-card"><h1>{html.escape(card["lemma"])}</h1>'
                f'<p class="wr-de">{html.escape(de) if de else "(noch keine Uebersetzung)"}</p>'
                f'<p class="wr-ctx">{html.escape(ctx_text)}</p></div>',
                unsafe_allow_html=True)

            # Passt die Uebersetzung nicht zum Kontext (z. B. „lane“ → „Gasse“,
            # gemeint ist „Landstraße“)? KI im Kontext fragen.
            lid = card["lemma_id"]
            if st.button("🔄 Im Kontext übersetzen", key=f"ctx_{lid}",
                         help="Fragt die KI nach der Bedeutung genau in dieser "
                              "Zeile (braucht einen Gemini-/Claude-Key)."):
                if not dictionary.get_ai_config()["provider"]:
                    st.warning("Dafür zuerst einen kostenlosen Gemini-Key unter "
                               "⚙️ Einstellungen hinterlegen.")
                else:
                    try:
                        with st.spinner("Übersetze im Kontext …"):
                            new = dictionary.context_translate(card["lemma"], ctx_text)
                        if new:
                            db.dict_store({card["lemma"]: new}, "manuell")
                            st.toast(f"„{card['lemma']}“ → {new}")
                            st.rerun()
                        else:
                            st.warning("Keine Übersetzung erhalten – später nochmal.")
                    except dictionary.AITranslationError as e:
                        st.warning(str(e))

            c1, c2 = st.columns([3, 1])
            new_de = c1.text_input("Uebersetzung korrigieren",
                                   key=f"newde_{lid}",
                                   label_visibility="collapsed",
                                   placeholder="… oder selbst eintippen / korrigieren")
            if c2.button("Speichern", key=f"savde_{lid}") and new_de.strip():
                db.dict_store({card["lemma"]: new_de.strip()}, "manuell")
                st.rerun()

            def rate(rating):
                res = srs.review(card["lemma_id"], card["fsrs"], rating)
                st.session_state[skey] = False
                if res["graduated"]:
                    st.toast(f"„{card['lemma']}“ sitzt – als bekannt markiert! 🏆")

            cols = st.columns(4)
            for col, (label, rating) in zip(cols, srs.RATING_LABELS):
                col.button(label, key=f"r_{rating}", on_click=rate,
                           args=(rating,), width="stretch")


# ---------------------------------------------------- Wortschatztest ----
def page_test():
    st.title("🧪 Wortschatztest")
    ts = st.session_state.get("test")

    if ts is None:
        st.markdown(
            "In **~3 Minuten** schaetzt WortRadar deinen englischen "
            "Wortschatz: Du siehst nacheinander Woerter quer durch alle "
            "Haeufigkeitsstufen und klickst nur **Kenne ich / Kenne ich "
            "nicht**.\n\n"
            "⚠️ **Ehrlichkeit zahlt sich aus:** Es sind auch *erfundene* "
            "Woerter dabei. Wer raet, verschlechtert nur seine eigene "
            "Prognose.")
        if vm.is_calibrated():
            p = vm.load_model()
            st.info(f"Bereits kalibriert (geschaetzter Wortschatz: "
                    f"~{vm.estimate_vocab_size():,} Wortfamilien). Du kannst "
                    f"den Test jederzeit wiederholen.".replace(",", "."))
        if st.button("Test starten", type="primary"):
            with st.spinner("Stelle Testwoerter zusammen (beim ersten Mal "
                            "dauert das einen Moment) …"):
                items = vm.make_test_items()
            st.session_state["test"] = {
                "items": items, "i": 0, "session": str(uuid.uuid4()),
                "answers": [], "result": None,
            }
            st.rerun()
        return

    items, i = ts["items"], ts["i"]
    if i < len(items):
        it = items[i]
        st.progress(i / len(items), text=f"Wort {i + 1} von {len(items)}")
        st.markdown(f'<div class="wr-card"><h1>{html.escape(it["word"])}</h1></div>',
                    unsafe_allow_html=True)

        def answer(known: bool):
            db.save_test_answer(ts["session"], it["word"], it["zipf"],
                                it["pseudo"], known)
            ts["answers"].append({"zipf": it["zipf"], "pseudo": it["pseudo"],
                                  "answer": known})
            ts["i"] += 1

        c = st.columns(2)
        c[0].button("✅ Kenne ich", key=f"ty{i}", on_click=answer, args=(True,),
                    width="stretch")
        c[1].button("❌ Kenne ich nicht", key=f"tn{i}", on_click=answer,
                    args=(False,), width="stretch")
        return

    if ts["result"] is None:
        params = vm.fit_curve(ts["answers"])
        vm.save_model(params)
        ts["result"] = params
    p = ts["result"]
    st.success("Kalibrierung abgeschlossen! Alle Coverage-Prognosen nutzen "
               "ab jetzt dein persoenliches Profil.")
    c = st.columns(3)
    c[0].metric("Geschaetzter Wortschatz",
                f"~{vm.estimate_vocab_size(p):,}".replace(",", ".") + " Familien")
    c[1].metric("50-%-Schwelle (Zipf)", f"{p['b']:.2f}",
                help="Bei dieser Haeufigkeitsstufe kennst du etwa jedes "
                     "zweite Wort. Kleiner = groesserer Wortschatz.")
    c[2].metric("Rate-Quote", f"{p['g']:.0%}",
                help="Anteil erfundener Woerter, die du als bekannt markiert hast.")
    if p["g"] > 0.2:
        st.warning("Du hast bei vielen erfundenen Woertern „Kenne ich“ "
                   "geklickt – die Schaetzung ist entsprechend unsicherer.")

    zs = [z / 10 for z in range(15, 76, 2)]
    st.line_chart(pd.DataFrame(
        {"P(Wort bekannt)": [vm.p_known_from_zipf(z, p) for z in zs]},
        index=pd.Index(zs, name="Zipf-Haeufigkeit")))
    if st.button("Test wiederholen"):
        st.session_state["test"] = None
        st.rerun()


# ----------------------------------------------------------- Unlocks ----
def page_unlocks():
    st.title("🔓 Unlocks")
    st.caption("Welches Dokument schaltest du mit den wenigsten neuen "
               "Vokabeln frei? Lerne gezielt - der Rest ergibt sich.")
    calib_warning()
    docs = db.list_documents()
    if not docs:
        st.info("Importiere zuerst Dokumente in der Bibliothek.")
        return

    ranked = sorted(((d, doc_stats(d["id"])) for d in docs),
                    key=lambda x: (x[1].words_to_98, -x[1].coverage))
    for d, s in ranked:
        with st.container(border=True):
            c1, c2 = st.columns([4, 2])
            c1.markdown(f"**{DOC_TYPES.get(d['doc_type'], '📄')} {d['title']}**")
            c1.progress(min(s.coverage, 1.0), text=f"{s.coverage:.1%}")
            if s.words_to_98 == 0:
                c2.markdown("### ✅ freigeschaltet")
            else:
                c2.metric("Bis 98 %", f"{s.words_to_98} Woerter")
                with st.expander(f"Diese {s.words_to_98} Woerter freischalten"):
                    uw = unlock_words(d["id"])
                    st.dataframe(pd.DataFrame([{
                        "Wort": w["lemma"],
                        "Uebersetzung": w["translation"] or "",
                        "Anzahl": w["cnt"],
                    } for w in uw]), hide_index=True, width="stretch")
                    if st.button("→ Alle ins Lernen aufnehmen",
                                 key=f"ul{d['id']}", type="primary"):
                        srs.start_learning([w["lemma_id"] for w in uw])
                        st.toast(f"{len(uw)} Woerter aufgenommen – Tab "
                                 "„Heute lernen“.")
                        st.rerun()

    st.divider()
    st.subheader("🌍 Beste Investition ueber alle Dokumente")
    roi = global_roi_words(20)
    if roi:
        st.caption("Woerter, die in mehreren Dokumenten vorkommen, zuerst - "
                   "maximaler Verstaendnis-Gewinn pro gelerntem Wort.")
        st.dataframe(pd.DataFrame([{
            "Wort": r["lemma"], "Dokumente": r["docs"],
            "Vorkommen gesamt": r["total_cnt"],
        } for r in roi]), hide_index=True, width="stretch")
        if st.button("Top 10 ins Lernen aufnehmen", type="primary"):
            srs.start_learning([r["lemma_id"] for r in roi[:10]])
            st.toast("Aufgenommen – Tab „Heute lernen“.")
            st.rerun()
    else:
        st.info("Gerade keine Kandidaten - alles bekannt oder schon im Lernen.")


# ------------------------------------------------------ Heute lernen ----
def page_learn():
    st.title("🃏 Heute lernen")
    counts = db.due_counts_by_doc()
    docs = [d for d in db.list_documents() if counts.get(d["id"], 0) > 0]
    if not docs:
        st.success("Keine faelligen Karten. 🎉 Sobald du Vokabeln ins Lernen "
                   "aufnimmst, tauchen sie hier auf - pro Song getrennt.")
        return

    # So einfach wie moeglich: EIN Dropdown (Song waehlen), darunter die Karte.
    options = {f"{DOC_TYPES.get(d['doc_type'], '📄')} {d['title']}  "
               f"({counts[d['id']]})": d["id"] for d in docs}
    if len(docs) > 1:
        total = sum(counts[d["id"]] for d in docs)
        options[f"🔀 Alle Songs gemischt  ({total})"] = None

    label = st.selectbox("Welchen Song moechtest du lernen?", list(options))
    st.divider()
    flashcards(options[label])


# ------------------------------------------------------ Einstellungen ----
def page_settings():
    st.title("⚙️ Einstellungen")

    st.subheader("🤖 Übersetzung (KI)")
    st.caption("Übersetzt deine Vokabeln **im Kontext** der jeweiligen Zeile "
               "(z. B. „country lane“ → „Landstraße“). Standardmäßig nutzt du "
               "den **geteilten Schlüssel der App** – du musst nichts tun. "
               "Optional kannst du unten deinen **eigenen** Key eintragen "
               "(z. B. wenn der geteilte gerade am Limit ist).")
    cfg = dictionary.get_ai_config()
    own_key = bool(db.meta_get("groq_key") or db.meta_get("gemini_key")
                   or db.meta_get("api_key"))
    if cfg["provider"]:
        quelle = "dein eigener Schlüssel" if own_key else "geteilter App-Schlüssel"
        st.success(f"✅ Übersetzung aktiv: **{cfg['provider'].title()}** "
                   f"({cfg['model']}) – {quelle}. Jede Übersetzung wird gecacht "
                   "und steht danach allen Nutzern zur Verfügung.")
        if st.button("🧪 Übersetzung testen"):
            try:
                with st.spinner("Teste …"):
                    got = dictionary.ai_translate(
                        [("lane", "Driving at 90 down those country lanes")])
                st.success(f"Funktioniert! „lane“ → **{got.get('lane', '(leer)')}**")
            except dictionary.AITranslationError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Fehler: {e}")
    else:
        st.warning("Noch keine KI konfiguriert. Trag unten einen **kostenlosen "
                   "Groq-Key** ein – dann funktionieren Übersetzungen sofort.")

    tab_groq, tab_gem, tab_claude = st.tabs(
        ["⭐ Groq (gratis, empfohlen)", "Gemini (gratis)", "Claude / OpenAI"])
    with tab_groq:
        st.markdown(
            "**Optional – nur falls du deinen eigenen Schlüssel nutzen willst.** "
            "Ohne Eintrag läuft alles über den geteilten App-Schlüssel.\n\n"
            "**Groq** hat die besten Gratis-Limits (schnell, ~30 Anfragen/Min, "
            "**keine Kreditkarte**) und läuft Llama 3.3 70B.\n\n"
            "1. Kostenlosen Key holen: `https://console.groq.com/keys`\n"
            "2. Hier einfügen, speichern, oben testen.")
        qkey = st.text_input("Dein Groq API-Key (optional)", type="password",
                             value=db.meta_get("groq_key") or "")
        qmodel = st.text_input("Groq-Modell", value=db.meta_get("groq_model")
                               or dictionary.GROQ_DEFAULT_MODEL)
        cols = st.columns(2)
        if cols[0].button("Eigenen Key speichern", type="primary"):
            db.meta_set("groq_key", qkey.strip())
            db.meta_set("groq_model", qmodel.strip() or dictionary.GROQ_DEFAULT_MODEL)
            st.success("Gespeichert – ab jetzt nutzt du deinen eigenen Schlüssel.")
            st.rerun()
        if cols[1].button("Eigenen Key entfernen"):
            db.meta_set("groq_key", "")
            st.success("Entfernt – du nutzt wieder den geteilten App-Schlüssel.")
            st.rerun()
    with tab_gem:
        st.markdown("Google **Gemini Flash**, ebenfalls gratis (Google-Konto "
                    "nötig). Key: `https://aistudio.google.com/app/apikey`")
        gkey = st.text_input("Gemini API-Key", type="password",
                             value=db.meta_get("gemini_key") or "")
        gmodel = st.text_input("Gemini-Modell", value=db.meta_get("gemini_model")
                               or dictionary.GEMINI_DEFAULT_MODEL)
        if st.button("Gemini speichern"):
            db.meta_set("gemini_key", gkey.strip())
            db.meta_set("gemini_model", gmodel.strip()
                        or dictionary.GEMINI_DEFAULT_MODEL)
            st.success("Gespeichert.")
            st.rerun()
        st.caption("Deployment: `GEMINI_API_KEY`. Bei Ratenlimit Modell "
                   "`gemini-2.0-flash-lite` probieren.")
    with tab_claude:
        st.caption("Kostenpflichtig, sehr zuverlässig – gut für den Bezahl-Tier. "
                   "Key: console.anthropic.com")
        key = st.text_input("Claude API-Key", type="password",
                            value=db.meta_get("api_key") or "")
        model = st.text_input("Claude-Modell", value=db.meta_get("api_model")
                              or dictionary.DEFAULT_MODEL)
        if st.button("Claude speichern"):
            db.meta_set("api_key", key.strip())
            db.meta_set("api_model", model.strip() or dictionary.DEFAULT_MODEL)
            st.success("Gespeichert.")
            st.rerun()
        st.caption("Vorrang: Groq > Gemini > Claude (je nachdem, was gesetzt ist).")

    with st.expander("📖 Optional: Offline-Wörterbuch (ohne Internet/KI)"):
        size = db.dict_size()
        st.caption(f"Aktuell {size:,}".replace(",", ".") + " Einträge. "
                   "Deckt häufige Wörter ohne API-Aufruf ab (gut zum Kosten "
                   "sparen). Die freie Ding-Liste (GPL):")
        if st.button("📥 Wörterbuch herunterladen & importieren"):
            try:
                with st.spinner("Lade & importiere (~10 s) …"):
                    n = dictionary.download_and_import_ding()
                st.success(f"{n:,}".replace(",", ".") + " Einträge importiert!")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Download fehlgeschlagen: {e}")
        dic = st.file_uploader("Oder Datei (Ding-.gz / dict.cc-TSV)",
                               type=["txt", "gz", "tsv"])
        if dic and st.button("Datei importieren"):
            raw = dic.getvalue()
            if dic.name.endswith(".gz") or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            with st.spinner("Importiere …"):
                n = dictionary.import_dictionary_file(
                    raw.decode("utf-8", errors="replace"))
            st.success(f"{n:,}".replace(",", ".") + " Einträge importiert.")
            st.rerun()

    st.divider()
    st.subheader("🧠 Sprachanalyse")
    st.markdown(f"Aktive Engine: **{engine_name()}**")
    if "Fallback" in engine_name():
        st.info("Fuer beste Lemma-Qualitaet einmalig ausfuehren und die App "
                "neu starten:\n```\npython -m spacy download en_core_web_sm\n```")

    st.divider()
    st.subheader("🗑️ Zuruecksetzen")
    c1, c2 = st.columns(2)
    if c1.button("Wortschatz-Kalibrierung loeschen"):
        db.meta_set_json("vocab_model", None)
        st.success("Kalibrierung entfernt - Test einfach neu machen.")
    if c2.button("⚠️ ALLE Daten loeschen"):
        st.session_state["nuke"] = True
    if st.session_state.get("nuke"):
        st.error("Wirklich alles loeschen (Dokumente, Lernstand, Woerterbuch)?")
        if st.button("Ja, unwiderruflich loeschen", type="primary"):
            with db.get_conn() as conn:
                for t in ("tokens", "lines", "documents", "knowledge",
                          "dictionary", "test_answers", "meta", "lemmas"):
                    conn.execute(f"DELETE FROM {t}")
            st.session_state.clear()
            st.rerun()


# ---------------------------------------------------------------- Login ----
def login_gate():
    """Sichert die App ab, sobald Supabase konfiguriert ist. Ohne Config:
    Entwicklungsmodus ohne Login (App laeuft direkt).
    Setzt user_id im DB-Kontext - bei jedem Render, damit der ContextVar
    im Streamlit-Thread aktuell bleibt."""
    user = st.session_state.get("user")
    db.set_current_user(user["id"] if user else None)
    if not auth.is_configured() or user:
        return
    st.markdown("<div style='max-width:420px;margin:6vh auto'>",
                unsafe_allow_html=True)
    st.title("📡 WortRadar")
    st.caption("Verstehen, bevor du anfängst. Melde dich an, um deine Songs, "
               "Decks und deinen Lernfortschritt zu speichern.")
    tab_in, tab_up = st.tabs(["Anmelden", "Konto erstellen"])
    with tab_in:
        e = st.text_input("E-Mail", key="li_e")
        p = st.text_input("Passwort", type="password", key="li_p")
        if st.button("Anmelden", type="primary", width="stretch"):
            try:
                st.session_state["user"] = auth.sign_in(e, p)
                st.rerun()
            except Exception as ex:  # noqa: BLE001
                st.error(auth.friendly_error(ex))
    with tab_up:
        e2 = st.text_input("E-Mail", key="su_e")
        p2 = st.text_input("Passwort (min. 6 Zeichen)", type="password", key="su_p")
        if st.button("Konto erstellen", width="stretch"):
            try:
                res = auth.sign_up(e2, p2)
                if res.get("needs_confirm"):
                    st.success("Fast fertig! Bestätige deine E-Mail, dann "
                               "anmelden.")
                elif res.get("id"):
                    st.session_state["user"] = res
                    st.rerun()
            except Exception as ex:  # noqa: BLE001
                st.error(auth.friendly_error(ex))
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


login_gate()


# -------------------------------------------------------------- Main ----
PAGES = {
    "📚 Bibliothek": page_library,
    "🧪 Wortschatztest": page_test,
    "🔓 Unlocks": page_unlocks,
    "🃏 Heute lernen": page_learn,
    "⚙️ Einstellungen": page_settings,
}

with st.sidebar:
    st.markdown("# 📡 WortRadar")
    # Ausstehenden Seitenwechsel (aus goto()) anwenden, BEVOR das Radio entsteht.
    pending = st.session_state.pop("_pending_nav", None)
    if pending in PAGES:
        st.session_state["nav"] = pending
    if "nav" not in st.session_state:
        st.session_state["nav"] = list(PAGES)[0]
    nav = st.radio("Navigation", list(PAGES), key="nav",
                   label_visibility="collapsed")
    # Manueller Klick in der Sidebar (kein goto): offenes Dokument schliessen,
    # damit die Bibliotheksliste erscheint statt der zuletzt offenen Doc-Ansicht.
    if pending is None and nav != st.session_state.get("_last_nav"):
        st.session_state["doc_id"] = None
    st.session_state["_last_nav"] = nav
    st.divider()
    if vm.is_calibrated():
        st.markdown(f"**Wortschatz:** ~{vm.estimate_vocab_size():,}"
                    .replace(",", ".") + " Familien")
    else:
        st.markdown("⚠️ **Unkalibriert** - Wortschatztest machen!")
    n_due = len(db.due_cards())
    if n_due:
        st.markdown(f"🃏 **{n_due} Karten faellig**")
    st.caption(f"Engine: {engine_name()}")
    user = st.session_state.get("user")
    if user:
        st.divider()
        st.caption(f"👤 {user.get('email', '')}")
        if st.button("Abmelden", width="stretch"):
            st.session_state.pop("user", None)
            st.rerun()

PAGES[nav]()
