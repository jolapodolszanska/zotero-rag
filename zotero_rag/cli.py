"""zotero-rag index | search | serve"""

import argparse
import sys

from . import config, models


def main():
    ap = argparse.ArgumentParser(prog="zotero-rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="zaindeksuj nowe/zmienione PDF-y z Zotero")
    p.add_argument("--full", action="store_true", help="przebuduj cały indeks")
    p = sub.add_parser("search", help="wyszukaj z terminala (test)")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=5)
    sub.add_parser("serve", help="uruchom serwer MCP (stdio)")
    args = ap.parse_args()

    models.setup_logging()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if args.cmd == "serve":
        from .server import serve
        serve()
        return

    from .store import Store
    store = Store(config.INDEX_PATH)
    if args.cmd == "index":
        from . import indexer
        print(f"Zotero: {config.ZOTERO_DIR}\nIndeks: {config.INDEX_PATH}\nModel: {config.EMBED_MODEL}")
        r = indexer.run(store, full=args.full)
        print(r)
    elif args.cmd == "search":
        from .server import search_library
        print(search_library(args.query, limit=args.n))
