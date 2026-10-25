"""Incremental indexing: new or changed PDFs are (re)indexed, removed items dropped."""

import logging
import time

from . import config, models
from .pdftext import Passage, pdf_passages
from .store import Store
from .zotero import Item, load_items

log = logging.getLogger(__name__)


def context_header(item: Item, section: str) -> str:
    """Prepended to each passage for embedding only, so the vector knows its source."""
    who = item.authors[0].split(",")[0] if item.authors else ""
    if len(item.authors) > 1:
        who += " et al."
    parts = [item.title, " ".join(p for p in (who, str(item.year or "")) if p), section]
    return " — ".join(p for p in parts if p) + "\n"


def run(store: Store, full: bool = False, progress=print) -> dict:
    model_in_index = store.get_meta("embed_model")
    if model_in_index and model_in_index != config.EMBED_MODEL:
        progress(f"Zmieniono model ({model_in_index} -> {config.EMBED_MODEL}): pełna reindeksacja.")
        full = True

    items = load_items()
    known = {} if full else store.indexed_mtimes()
    current = {i.key for i in items}

    removed = [k for k in store.indexed_mtimes() if k not in current or full]
    for k in removed:
        store.delete_item(k)
    store.commit()

    done = skipped = failed = 0
    for n, item in enumerate(items, 1):
        path = item.pdf_path
        if not path or not path.exists():
            log.warning("Brak pliku PDF: %s (%s)", item.title, path)
            failed += 1
            continue
        mtime = path.stat().st_mtime
        if known.get(item.key) == mtime:
            skipped += 1
            continue
        try:
            passages = pdf_passages(path, item.title)
        except Exception as e:  # damaged or encrypted PDFs
            log.warning("Nie udało się odczytać %s: %s", path, e)
            failed += 1
            continue
        if item.abstract:
            passages.insert(0, Passage(0, "Abstract", item.abstract))
        if not passages:
            log.warning("Brak tekstu (skan bez OCR?): %s", item.title)
            failed += 1
            continue
        t0 = time.time()
        vecs = models.embed_documents([context_header(item, p.section) + p.text for p in passages])
        n_pages = max(p.page for p in passages)
        store.put_item(item, mtime, n_pages, passages, vecs)
        done += 1
        progress(f"[{n}/{len(items)}] {item.title[:70]} — {len(passages)} fragmentów, {time.time() - t0:.1f}s")

    store.set_meta("embed_model", config.EMBED_MODEL)
    store.set_meta("last_indexed", time.strftime("%Y-%m-%d %H:%M"))
    return {"indexed": done, "unchanged": skipped, "failed": failed,
            "removed": len([k for k in removed if k not in current])}
