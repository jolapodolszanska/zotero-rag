"""MCP server: exposes the Zotero index to Claude as tools.

Claude does the parts of a RAG pipeline that need a language model (reading
filters out of the question, rewriting follow-ups, writing the answer with
citations); this server only retrieves.
"""

from collections import Counter

from mcp.server.mcpserver import MCPServer

from . import config, indexer
from .search import Filters, search
from .store import Store

INSTRUCTIONS = """\
Narzędzia do przeszukiwania prywatnej biblioteki Zotero użytkownika (pełne teksty PDF, PL i EN).
- Zanim odpowiesz na pytanie merytoryczne o literaturę, użyj search_library. Zadaj kilka zapytań,
  jeśli temat ma różne sformułowania lub dotyczy prac w obu językach (zapytaj po polsku i po angielsku).
- Ograniczenia z pytania (lata, autor, tag, kolekcja) przekazuj jako parametry filtrów.
- Opieraj odpowiedź wyłącznie na zwróconych fragmentach i cytuj je jako (Nazwisko, rok, s. X).
  Jeśli fragmenty nie wystarczają, powiedz to wprost zamiast uzupełniać z własnej wiedzy.
- get_context zwraca sąsiednie fragmenty, gdy cytat jest urwany lub potrzebny jest szerszy kontekst.
"""

mcp = MCPServer("zotero-rag", instructions=INSTRUCTIONS)
_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store(config.INDEX_PATH)
    return _store


def _cite(item: dict) -> str:
    a = item["authors"]
    who = a[0].split(",")[0] if a else "b.a."
    if len(a) == 2:
        who += " i " + a[1].split(",")[0]
    elif len(a) > 2:
        who += " i in."
    return f"{who} ({item['year'] or 'b.d.'})"


def _pdf_link(item: dict, page: int) -> str:
    link = f"zotero://open-pdf/library/items/{item['attachment_key']}"
    return link + (f"?page={page}" if page else "")


def _format_passage(n: int, p: dict, item: dict) -> str:
    where = f"s. {p['page']}" if p["page"] else "abstrakt"
    section = f" | sekcja: {p['section']}" if p["section"] and p["section"] != "Abstract" else ""
    return (f"[{n}] {_cite(item)}. {item['title']}\n"
            f"    {where}{section} | item_key={item['key']} seq={p['seq']} | {_pdf_link(item, p['page'])}\n"
            f"    \"{p['text']}\"")


@mcp.tool()
def search_library(query: str, limit: int = 8, year_from: int | None = None, year_to: int | None = None,
                   author: str | None = None, tag: str | None = None, collection: str | None = None,
                   item_type: str | None = None, per_item: int = 3) -> str:
    """Hybrid semantic + keyword search over the full text of PDFs in the Zotero library.

    Returns the best matching passages with author, year, title, page and a zotero:// link.
    query: what to look for, in Polish or English (searching both languages often helps).
    author/tag/collection: case-insensitive substring match. item_type: e.g. journalArticle, book, thesis.
    per_item: max passages per work (raise it when studying a single source).
    """
    f = Filters(year_from, year_to, author, tag, collection, item_type)
    hits = search(store(), query, f, limit=min(limit, 30), per_item=per_item)
    if not hits and f.active():
        return "Brak wyników z tymi filtrami. Spróbuj bez filtrów lub z innymi wartościami."
    if not hits:
        return "Brak wyników (czy biblioteka jest zaindeksowana? zob. index_status)."
    items = {}
    lines = []
    for n, p in enumerate(hits, 1):
        item = items.setdefault(p["item_key"], store().item(p["item_key"]))
        lines.append(_format_passage(n, p, item))
    return "\n\n".join(lines)


@mcp.tool()
def search_within_item(item_key: str, query: str, limit: int = 6) -> str:
    """Search inside one work only (item_key from search results or list_items)."""
    item = store().item(item_key)
    if not item:
        return f"Nie ma pozycji {item_key} w indeksie."
    hits = search(store(), query, Filters(item_keys=[item_key]), limit=limit, per_item=limit)
    return "\n\n".join(_format_passage(n, p, item) for n, p in enumerate(hits, 1)) or "Brak wyników."


@mcp.tool()
def get_context(item_key: str, seq: int, before: int = 1, after: int = 1) -> str:
    """Passages around passage `seq` of a work, in reading order (seq comes from search results)."""
    item = store().item(item_key)
    if not item:
        return f"Nie ma pozycji {item_key} w indeksie."
    ps = store().item_passages(item_key, max(0, seq - before), seq + after)
    return f"{_cite(item)}. {item['title']}\n\n" + "\n\n".join(
        f"[seq={p['seq']}, s. {p['page']}{', ' + p['section'] if p['section'] else ''}]\n{p['text']}" for p in ps)


@mcp.tool()
def get_item(item_key: str) -> str:
    """Full bibliographic record of one work: authors, year, venue, DOI, tags, collections, abstract."""
    it = store().item(item_key)
    if not it:
        return f"Nie ma pozycji {item_key} w indeksie."
    rows = [
        ("Tytuł", it["title"]), ("Autorzy", "; ".join(it["authors"])), ("Rok", it["year"]),
        ("Typ", it["item_type"]), ("Źródło", it["publication"]), ("DOI", it["doi"]),
        ("Klucz cytowania", it["citation_key"]), ("Tagi", ", ".join(it["tags"])),
        ("Kolekcje", ", ".join(it["collections"])), ("Stron", it["n_pages"]),
        ("PDF", _pdf_link(it, 0)), ("Abstrakt", it["abstract"]),
    ]
    return "\n".join(f"{k}: {v}" for k, v in rows if v)


@mcp.tool()
def list_items(collection: str | None = None, tag: str | None = None, author: str | None = None,
               year_from: int | None = None, year_to: int | None = None) -> str:
    """List indexed works (optionally filtered), one per line: item_key, citation, title."""
    from .search import allowed_items
    f = Filters(year_from, year_to, author, tag, collection)
    keep = allowed_items(store(), f)
    rows = [it for it in store().items() if keep is None or it["key"] in keep]
    return "\n".join(f"{it['key']} | {_cite(it)} | {it['title']}" for it in rows) or "Brak pozycji."


@mcp.tool()
def index_status() -> str:
    """How many works and passages are indexed, with which model, plus collections and top tags."""
    s = store().stats()
    items = store().items()
    colls = Counter(c for it in items for c in it["collections"])
    tags = Counter(t for it in items for t in it["tags"])
    return (f"Pozycji: {s['items']}, fragmentów: {s['passages']}\n"
            f"Model: {s['embed_model']}, ostatnia indeksacja: {s['last_indexed']}\n"
            f"Kolekcje: {', '.join(f'{c} ({n})' for c, n in colls.most_common()) or '-'}\n"
            f"Tagi (top 30): {', '.join(f'{t} ({n})' for t, n in tags.most_common(30)) or '-'}")


@mcp.tool()
def update_index() -> str:
    """Index PDFs added or changed in Zotero since the last run (can take a few minutes for many new files)."""
    r = indexer.run(store(), progress=lambda *_: None)
    return (f"Zaindeksowano: {r['indexed']}, bez zmian: {r['unchanged']}, "
            f"usunięto: {r['removed']}, błędy: {r['failed']}")


def serve():
    mcp.run()
