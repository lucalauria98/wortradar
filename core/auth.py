"""Supabase-Authentifizierung (Login) fuer WortRadar.

Opt-in per Konfiguration: Login ist nur aktiv, wenn SUPABASE_URL und
SUPABASE_ANON_KEY gesetzt sind (Streamlit-Secrets ODER Umgebungsvariablen).
Ohne Config laeuft die App im Einzelnutzer-/Entwicklungsmodus OHNE Login -
so bleibt lokales Testen einfach, im Online-Betrieb wird Login erzwungen.

Schritt 1 der Infrastruktur (Modell: ein Server-KI-Key + spaetere Nutzungs-
limits). Die Nutzerdaten liegen aktuell noch lokal; die Migration auf
Supabase-Postgres (pro Nutzer eigene Daten) ist der naechste Schritt.
"""
from __future__ import annotations

import os


def _cfg(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def is_configured() -> bool:
    """True, sobald Supabase-Zugangsdaten hinterlegt sind -> Login wird aktiv."""
    return bool(_cfg("SUPABASE_URL") and _cfg("SUPABASE_ANON_KEY"))


def _client():
    """Frischer Supabase-Client pro Aufruf (kein geteilter Sitzungszustand
    zwischen verschiedenen Besuchern - wichtig fuer Multi-User)."""
    try:
        from supabase import create_client
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Paket 'supabase' fehlt. Installiere es mit "
            "`python -m pip install supabase`."
        ) from e
    return create_client(_cfg("SUPABASE_URL"), _cfg("SUPABASE_ANON_KEY"))


def sign_in(email: str, password: str) -> dict:
    """Meldet an. Liefert {id, email, token} oder wirft mit klarer Meldung."""
    res = _client().auth.sign_in_with_password(
        {"email": email.strip(), "password": password})
    if not res or not getattr(res, "user", None):
        raise RuntimeError("Anmeldung fehlgeschlagen.")
    token = getattr(getattr(res, "session", None), "access_token", None)
    return {"id": res.user.id, "email": res.user.email, "token": token}


def sign_up(email: str, password: str) -> dict:
    """Registriert ein neues Konto. Je nach Supabase-Einstellung ist danach
    eine E-Mail-Bestaetigung noetig (dann gibt es noch keine Session)."""
    res = _client().auth.sign_up(
        {"email": email.strip(), "password": password})
    user = getattr(res, "user", None)
    session = getattr(res, "session", None)
    return {
        "id": getattr(user, "id", None),
        "email": getattr(user, "email", None),
        "token": getattr(session, "access_token", None),
        "needs_confirm": user is not None and session is None,
    }


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "invalid login" in low or "invalid" in low and "credential" in low:
        return "E-Mail oder Passwort falsch."
    if "already registered" in low or "already exists" in low:
        return "Diese E-Mail ist bereits registriert – einfach anmelden."
    if "password" in low and ("length" in low or "least" in low or "weak" in low):
        return "Passwort zu kurz (mindestens 6 Zeichen)."
    if "email" in low and "confirm" in low:
        return "Bitte zuerst die Bestätigungs-E-Mail bestätigen."
    return f"Fehler: {msg}"
