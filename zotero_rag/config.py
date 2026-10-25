"""Settings, read from environment variables (all optional)."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Zotero data folder: the one holding zotero.sqlite and storage/
ZOTERO_DIR = Path(os.environ.get("ZOTERO_DIR", Path.home() / "Zotero"))
# Base folder for linked files ("attachments:" paths), if you use one
ZOTERO_BASE_DIR = os.environ.get("ZOTERO_BASE_DIR", "")

DATA_DIR = Path(os.environ.get("ZOTERO_RAG_DATA", PROJECT_DIR / "data"))
INDEX_PATH = DATA_DIR / "index.sqlite"

# Multilingual models, good for Polish and English. Set RERANK_MODEL="" to skip reranking.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# Some models (e5, mmlw) expect prefixes such as "query: " / "passage: "; bge-m3 does not.
QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")
DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "")
MAX_SEQ_LENGTH = int(os.environ.get("EMBED_MAX_SEQ", "512"))
