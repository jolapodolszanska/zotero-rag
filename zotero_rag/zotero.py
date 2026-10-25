"""Read items with PDF attachments from zotero.sqlite.

Zotero locks its database while running, so we read a temporary copy.
Nothing is ever written to Zotero.
"""

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Item:
    key: str              # parent item key
    attachment_key: str   # PDF attachment key (for zotero://open-pdf links)
    title: str
    authors: list[str]
    year: int | None
    item_type: str
    publication: str = ""
    doi: str = ""
    abstract: str = ""
    citation_key: str = ""
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    pdf_path: Path | None = None


def _connect_copy(zotero_dir: Path) -> tuple[sqlite3.Connection, Path]:
    src = zotero_dir / "zotero.sqlite"
    if not src.exists():
        raise FileNotFoundError(f"Brak bazy Zotero: {src} (ustaw ZOTERO_DIR)")
    tmp = Path(tempfile.mkdtemp(prefix="zotero_rag_")) / "zotero.sqlite"
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(tmp)
    conn.row_factory = sqlite3.Row
    return conn, tmp


def _resolve_path(raw: str, attachment_key: str, zotero_dir: Path) -> Path | None:
    if not raw:
        return None
    if raw.startswith("storage:"):
        return zotero_dir / "storage" / attachment_key / raw[len("storage:"):]
    if raw.startswith("attachments:"):
        if not config.ZOTERO_BASE_DIR:
            return None
        return Path(config.ZOTERO_BASE_DIR) / raw[len("attachments:"):]
    return Path(raw)


def _year(date: str | None) -> int | None:
    # Zotero stores dates as "YYYY-MM-DD original text"
    if date and date[:4].isdigit() and date[:4] != "0000":
        return int(date[:4])
    return None


def load_items(zotero_dir: Path | None = None) -> list[Item]:
    """Every non-deleted item that has a PDF attachment (the first PDF per item)."""
    zotero_dir = zotero_dir or config.ZOTERO_DIR
    conn, tmp = _connect_copy(zotero_dir)
    try:
        return _load(conn, zotero_dir)
    finally:
        conn.close()
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _load(conn: sqlite3.Connection, zotero_dir: Path) -> list[Item]:
    deleted = {r[0] for r in conn.execute("SELECT itemID FROM deletedItems")}

    pdfs = conn.execute("""
        SELECT ia.parentItemID AS parent, att.key AS att_key, ia.path AS path
        FROM itemAttachments ia JOIN items att ON att.itemID = ia.itemID
        WHERE ia.parentItemID IS NOT NULL
          AND (ia.contentType = 'application/pdf' OR lower(ia.path) LIKE '%.pdf')
        ORDER BY ia.parentItemID, att.dateAdded
    """).fetchall()
    first_pdf: dict[int, sqlite3.Row] = {}
    for r in pdfs:
        if r["parent"] not in deleted:
            first_pdf.setdefault(r["parent"], r)
    if not first_pdf:
        return []

    ids = list(first_pdf)
    marks = ",".join("?" * len(ids))

    fields: dict[int, dict[str, str]] = {i: {} for i in ids}
    for r in conn.execute(f"""
        SELECT d.itemID, f.fieldName, v.value
        FROM itemData d JOIN fields f ON f.fieldID = d.fieldID
        JOIN itemDataValues v ON v.valueID = d.valueID
        WHERE d.itemID IN ({marks})
    """, ids):
        fields[r[0]][r[1]] = r[2]

    authors: dict[int, list[str]] = {i: [] for i in ids}
    for r in conn.execute(f"""
        SELECT ic.itemID, c.firstName, c.lastName
        FROM itemCreators ic JOIN creators c ON c.creatorID = ic.creatorID
        WHERE ic.itemID IN ({marks}) ORDER BY ic.itemID, ic.orderIndex
    """, ids):
        authors[r[0]].append(", ".join(p for p in (r[2], r[1]) if p))

    tags: dict[int, list[str]] = {i: [] for i in ids}
    for r in conn.execute(f"""
        SELECT it.itemID, t.name FROM itemTags it JOIN tags t ON t.tagID = it.tagID
        WHERE it.itemID IN ({marks})
    """, ids):
        tags[r[0]].append(r[1])

    collections: dict[int, list[str]] = {i: [] for i in ids}
    for r in conn.execute(f"""
        SELECT ci.itemID, c.collectionName FROM collectionItems ci
        JOIN collections c ON c.collectionID = ci.collectionID
        WHERE ci.itemID IN ({marks})
    """, ids):
        collections[r[0]].append(r[1])

    meta = {r[0]: (r[1], r[2]) for r in conn.execute(f"""
        SELECT i.itemID, i.key, t.typeName FROM items i
        JOIN itemTypes t ON t.itemTypeID = i.itemTypeID WHERE i.itemID IN ({marks})
    """, ids)}

    items = []
    for item_id, pdf in first_pdf.items():
        key, item_type = meta[item_id]
        f = fields[item_id]
        publication = (f.get("publicationTitle") or f.get("bookTitle")
                       or f.get("proceedingsTitle") or f.get("publisher") or "")
        items.append(Item(
            key=key,
            attachment_key=pdf["att_key"],
            title=f.get("title", "(bez tytułu)"),
            authors=authors[item_id],
            year=_year(f.get("date")),
            item_type=item_type,
            publication=publication,
            doi=f.get("DOI", ""),
            abstract=f.get("abstractNote", ""),
            citation_key=f.get("citationKey", ""),
            tags=sorted(tags[item_id]),
            collections=sorted(collections[item_id]),
            pdf_path=_resolve_path(pdf["path"] or "", pdf["att_key"], zotero_dir),
        ))
    return items
