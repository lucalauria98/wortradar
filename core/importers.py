"""Import-Schicht von WortRadar.

Bewusste Design-Entscheidung (Urheberrecht): Es gibt KEINE automatische
Beschaffung von Songtexten oder Buchinhalten aus dem Netz. Der Nutzer
bringt eigene Dateien mit oder fuegt Text per Copy-Paste ein. Alles
bleibt lokal in der eigenen SQLite-Datei.

Unterstuetzte Formate:
  - Copy-Paste / .txt / .md   -> zeilenweise
  - .lrc                      -> synchronisierte Lyrics inkl. Zeitstempel
  - .srt / .vtt               -> Untertitel inkl. Zeitstempel
  - .pdf                      -> Textextraktion via PyMuPDF
  - .epub                     -> Kapiteltext via Standardbibliothek (zipfile)

Liefert immer: list[(line_no, text, t_start_seconds_or_None)]
"""
from __future__ import annotations

import io
import re
import zipfile
from html.parser import HTMLParser

Line = tuple[int, str, float | None]

_LRC_TS = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
_LRC_META = re.compile(r"^\[(ar|ti|al|by|offset|length|re|ve|id):", re.I)
_SRT_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->"
)
_TAG = re.compile(r"<[^>]+>")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z])")

MAX_LINE_LEN = 220  # laengere Absaetze werden fuer die Zeilen-Ampel in Saetze gesplittet


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _split_long(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_LINE_LEN:
        return [text] if text else []
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 <= MAX_LINE_LEN:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _finalize(raw_lines: list[tuple[str, float | None]]) -> list[Line]:
    out: list[Line] = []
    n = 0
    for text, ts in raw_lines:
        for piece in _split_long(text):
            out.append((n, piece, ts))
            ts = None  # Zeitstempel nur auf dem ersten Teilstueck
            n += 1
    return out


# ------------------------------------------------------------ Formate ----
def parse_plain(text: str) -> list[Line]:
    return _finalize([(ln, None) for ln in text.splitlines()])


def parse_lrc(text: str) -> list[Line]:
    raw: list[tuple[str, float | None]] = []
    for line in text.splitlines():
        if _LRC_META.match(line.strip()):
            continue
        stamps = _LRC_TS.findall(line)
        content = _LRC_TS.sub("", line).strip()
        if not content:
            continue
        if stamps:
            mm, ss, frac = stamps[0]
            frac = (frac or "0").ljust(3, "0")[:3]
            t = int(mm) * 60 + int(ss) + int(frac) / 1000.0
            raw.append((content, t))
        else:
            raw.append((content, None))
    return _finalize(raw)


def parse_srt(text: str) -> list[Line]:
    raw: list[tuple[str, float | None]] = []
    t_current: float | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            t_current = None
            continue
        m = _SRT_TS.search(s)
        if m:
            h, mi, se, ms = (int(x) for x in m.groups())
            t_current = h * 3600 + mi * 60 + se + ms / 1000.0
            continue
        if s.isdigit():
            continue
        if s.upper() == "WEBVTT" or s.startswith(("NOTE", "STYLE", "REGION")):
            continue
        content = _TAG.sub("", s).strip()
        if content:
            raw.append((content, t_current))
            t_current = None
    return _finalize(raw)


def parse_pdf(data: bytes) -> list[Line]:
    import fitz  # PyMuPDF

    raw: list[tuple[str, float | None]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            # PDF bricht Zeilen hart um -> Absaetze an Leerzeilen rekonstruieren
            for para in re.split(r"\n\s*\n", text):
                joined = " ".join(p.strip() for p in para.splitlines() if p.strip())
                joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", joined)  # Silbentrennung
                if joined:
                    raw.append((joined, None))
    return _finalize(raw)


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "head", "title"}
    _BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def parse_epub(data: bytes) -> list[Line]:
    raw: list[tuple[str, float | None]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith((".xhtml", ".html", ".htm"))]
        names.sort()
        for name in names:
            parser = _HTMLText()
            try:
                parser.feed(_decode(z.read(name)))
            except Exception:
                continue
            for para in parser.text().splitlines():
                para = re.sub(r"\s+", " ", para).strip()
                if para:
                    raw.append((para, None))
    return _finalize(raw)


# ------------------------------------------------------------ Routing ----
def parse_upload(filename: str, data: bytes) -> list[Line]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(data)
    if name.endswith(".epub"):
        return parse_epub(data)
    text = _decode(data)
    if name.endswith(".lrc"):
        return parse_lrc(text)
    if name.endswith((".srt", ".vtt")):
        return parse_srt(text)
    return parse_plain(text)


def parse_pasted(text: str) -> list[Line]:
    """Copy-Paste: erkennt LRC-Zeitstempel automatisch."""
    if _LRC_TS.search(text):
        return parse_lrc(text)
    if _SRT_TS.search(text):
        return parse_srt(text)
    return parse_plain(text)
