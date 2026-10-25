"""PDF -> clean paragraphs -> passages for indexing.

Paragraphs keep their page number; headings (larger or bold short lines) start
a new section. Lines repeated across many pages (running headers, footers,
page numbers) are dropped, as is everything under a references heading.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

TARGET_CHARS = 1000   # ~250-300 tokens
MAX_CHARS = 1500      # hard cut for text without sentence breaks
MIN_CHARS = 200       # a shorter tail is merged into the previous passage

_FLAGS = fitz.TEXT_DEHYPHENATE | (fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)
_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_SENTENCE = re.compile(rf"(?<=[.!?…])\s+(?=[{_UPPER}0-9„“\"(\[])")
_REFERENCES = re.compile(
    r"^\W*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited|bibliografia"
    r"|literatura(?: przedmiotu| cytowana)?|źródła|spis literatury|wykaz literatury)\W*$", re.I)
_BOILERPLATE = re.compile(r"Downloaded from|This content downloaded|All rights reserved|©|Copyright \d", re.I)


@dataclass
class Paragraph:
    page: int
    text: str
    heading: bool


@dataclass
class Passage:
    page: int
    section: str
    text: str


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("­", "").strip()


def _join_spans(spans) -> str:
    """Some PDFs set every word as its own span with no space characters:
    put a space wherever there is a visible gap between spans."""
    out = ""
    prev = None
    for s in spans:
        if prev is not None and not out.endswith(" ") and not s["text"].startswith(" "):
            if s["bbox"][0] - prev["bbox"][2] > 0.15 * min(s["size"], prev["size"]):
                out += " "
        out += s["text"]
        prev = s
    return out


def extract_paragraphs(path: Path) -> list[Paragraph]:
    raw = []  # (page, text, size, bold, y0, y1, page_height)
    sizes: Counter = Counter()
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            h = page.rect.height
            for block in page.get_text("dict", flags=_FLAGS)["blocks"]:
                lines, max_size, bold = [], 0.0, True
                for line in block.get("lines", []):
                    if abs(line["dir"][0]) < 0.9:  # rotated text, e.g. arXiv stamps
                        continue
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    text = _norm(_join_spans(spans))
                    if not text or _BOILERPLATE.search(text):
                        continue
                    lines.append(text)
                    for s in spans:
                        sizes[round(s["size"], 1)] += len(s["text"])
                        max_size = max(max_size, s["size"])
                        bold &= bool(s["flags"] & 16) or "bold" in s["font"].lower()
                if lines:
                    x0, y0, x1, y1 = block["bbox"]
                    raw.append((pno, " ".join(lines), max_size, bold, y0, y1, h))
    if not raw:
        return []

    body = sizes.most_common(1)[0][0]
    n_pages = raw[-1][0]

    # Running headers/footers: the same text (digits ignored) in the page margins on many pages
    def mkey(t):
        return re.sub(r"\d+", "#", t.lower())

    def in_margin(y0, y1, h):
        return y1 < 0.1 * h or y0 > 0.9 * h

    per_page: dict[int, set] = {}
    for p, t, _, _, y0, y1, h in raw:
        if in_margin(y0, y1, h):
            per_page.setdefault(p, set()).add(mkey(t))
    pages_with = Counter(k for keys in per_page.values() for k in keys)
    repeated = {k for k, c in pages_with.items() if c >= max(3, 0.3 * n_pages)}

    out = []
    for p, text, size, bold, y0, y1, h in raw:
        if in_margin(y0, y1, h) and (mkey(text) in repeated or text.isdigit()):
            continue
        short = len(text) <= 120 and not text.endswith((".", ",", ";"))
        wordy = (re.search(rf"[{_UPPER}a-ząćęłńóśźż]{{3}}", text) is not None
                 and not re.search(r"[-]", text)  # symbol-font glyphs: formulas
                 and sum(c.isalpha() or c.isspace() for c in text) >= 0.75 * len(text))
        heading = short and wordy and (
            size >= body + 1.0 or (bold and size >= body - 0.5 and len(text) <= 80))
        out.append(Paragraph(p, text, bool(heading)))
    return out


def _split_long(sentence: str) -> list[str]:
    parts = []
    while len(sentence) > MAX_CHARS:
        cut = sentence.rfind(" ", 0, MAX_CHARS)
        cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
        parts.append(sentence[:cut])
        sentence = sentence[cut:].lstrip()
    parts.append(sentence)
    return parts


def _mostly_text(s: str) -> bool:
    return sum(c.isalpha() for c in s) >= 0.5 * len(s)


def _squash(s: str) -> str:
    return re.sub(r"\W", "", s.casefold())


def make_passages(paragraphs: list[Paragraph], title: str = "") -> list[Passage]:
    passages: list[Passage] = []
    t = _squash(title)
    section, buf, buf_len, buf_page = "", [], 0, 0
    in_refs = False

    def flush():
        nonlocal buf, buf_len
        text = " ".join(buf).strip()
        buf, buf_len = [], 0
        if not text or not _mostly_text(text):
            return
        if len(text) < MIN_CHARS and passages and passages[-1].section == section:
            passages[-1].text += " " + text
        else:
            passages.append(Passage(buf_page, section, text))

    for para in paragraphs:
        if para.heading:
            flush()
            in_refs = bool(_REFERENCES.match(para.text))
            h = _squash(para.text)
            # the work's own title or journal banner is not a section
            if len(h) >= 10 and t and (h in t or t in h):
                continue
            section = para.text[:100]
            continue
        if in_refs:
            continue
        for sentence in _SENTENCE.split(para.text):
            for piece in _split_long(sentence.strip()):
                if not piece:
                    continue
                if buf and buf_len + len(piece) + 1 > TARGET_CHARS:
                    flush()
                if not buf:
                    buf_page = para.page
                buf.append(piece)
                buf_len += len(piece) + 1
    flush()
    return passages


def pdf_passages(path: Path, title: str = "") -> list[Passage]:
    return make_passages(extract_paragraphs(path), title)
