"""The index: one SQLite file with items, passages, their embeddings and an
FTS5 table for keyword (BM25) search. Embeddings are loaded into memory as
one normalised float32 matrix for brute-force cosine search, which is fast
enough for personal libraries (100k passages ~ 400 MB with bge-m3).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS items (
    key TEXT PRIMARY KEY, attachment_key TEXT, title TEXT, authors TEXT, year INTEGER,
    item_type TEXT, publication TEXT, doi TEXT, abstract TEXT, citation_key TEXT,
    tags TEXT, collections TEXT, pdf_path TEXT, pdf_mtime REAL, n_pages INTEGER
);
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY, item_key TEXT, seq INTEGER, page INTEGER, section TEXT,
    text TEXT, emb BLOB
);
CREATE INDEX IF NOT EXISTS passages_item ON passages(item_key, seq);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text, content='passages', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
    INSERT INTO passages_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
    INSERT INTO passages_fts(passages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._matrix: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    # --- meta ---
    def get_meta(self, k: str) -> str | None:
        row = self.conn.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        return row[0] if row else None

    def set_meta(self, k: str, v: str):
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, v))
        self.conn.commit()

    # --- writes ---
    def indexed_mtimes(self) -> dict[str, float]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT key, pdf_mtime FROM items")}

    def delete_item(self, key: str):
        self.conn.execute("DELETE FROM passages WHERE item_key = ?", (key,))
        self.conn.execute("DELETE FROM items WHERE key = ?", (key,))
        self._matrix = None

    def put_item(self, item, mtime: float, n_pages: int, passages, embeddings: np.ndarray):
        self.delete_item(item.key)
        self.conn.execute(
            "INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item.key, item.attachment_key, item.title, json.dumps(item.authors, ensure_ascii=False),
             item.year, item.item_type, item.publication, item.doi, item.abstract, item.citation_key,
             json.dumps(item.tags, ensure_ascii=False), json.dumps(item.collections, ensure_ascii=False),
             str(item.pdf_path), mtime, n_pages))
        self.conn.executemany(
            "INSERT INTO passages (item_key, seq, page, section, text, emb) VALUES (?,?,?,?,?,?)",
            [(item.key, i, p.page, p.section, p.text, e.astype(np.float32).tobytes())
             for i, (p, e) in enumerate(zip(passages, embeddings))])
        self.conn.commit()
        self._matrix = None

    def commit(self):
        self.conn.commit()

    # --- reads ---
    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(passage ids, their item keys, normalised embedding matrix), cached until the next write."""
        if self._matrix is None:
            rows = self.conn.execute("SELECT id, item_key, emb FROM passages ORDER BY id").fetchall()
            self._ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._keys = np.array([r[1] for r in rows], dtype=object)
            if rows:
                m = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            else:
                m = np.zeros((0, 1), dtype=np.float32)
            self._matrix = m
        return self._ids, self._keys, self._matrix

    def item(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM items WHERE key = ?", (key,)).fetchone()
        return _item_dict(row) if row else None

    def items(self) -> list[dict]:
        return [_item_dict(r) for r in self.conn.execute("SELECT * FROM items ORDER BY year, title")]

    def passages(self, ids) -> dict[int, dict]:
        ids = list(ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, item_key, seq, page, section, text FROM passages WHERE id IN ({marks})", ids)
        return {r["id"]: dict(r) for r in rows}

    def item_passages(self, key: str, seq_from: int = 0, seq_to: int = 10**9) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, item_key, seq, page, section, text FROM passages "
            "WHERE item_key = ? AND seq BETWEEN ? AND ? ORDER BY seq", (key, seq_from, seq_to))
        return [dict(r) for r in rows]

    def keyword_search(self, fts_query: str, allowed_items: set[str] | None, limit: int) -> list[int]:
        sql = ("SELECT p.id, p.item_key FROM passages_fts f JOIN passages p ON p.id = f.rowid "
               "WHERE passages_fts MATCH ? ORDER BY bm25(passages_fts) LIMIT ?")
        try:
            rows = self.conn.execute(sql, (fts_query, limit * 5 if allowed_items is not None else limit))
        except sqlite3.OperationalError:
            return []
        out = [r[0] for r in rows if allowed_items is None or r[1] in allowed_items]
        return out[:limit]

    def stats(self) -> dict:
        n_items = self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        n_pass = self.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        return {"items": n_items, "passages": n_pass,
                "embed_model": self.get_meta("embed_model"), "last_indexed": self.get_meta("last_indexed")}


def _item_dict(row) -> dict:
    d = dict(row)
    for k in ("authors", "tags", "collections"):
        d[k] = json.loads(d[k] or "[]")
    return d
