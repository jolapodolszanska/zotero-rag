"""Hybrid retrieval: dense (cosine) + keyword (BM25), fused with Reciprocal
Rank Fusion, then reranked by a cross-encoder, with a per-item cap so one
paper does not crowd out the rest."""

import re
from dataclasses import dataclass

import numpy as np

from . import models
from .store import Store

CANDIDATES = 50   # from each retriever
RERANK_TOP = 30   # fused candidates passed to the cross-encoder
RRF_K = 60


@dataclass
class Filters:
    year_from: int | None = None
    year_to: int | None = None
    author: str | None = None
    tag: str | None = None
    collection: str | None = None
    item_type: str | None = None
    item_keys: list[str] | None = None

    def active(self) -> bool:
        return any(v not in (None, "", []) for v in vars(self).values())


def _contains(values: list[str], needle: str) -> bool:
    needle = needle.casefold()
    return any(needle in v.casefold() for v in values)


def allowed_items(store: Store, f: Filters) -> set[str] | None:
    if not f.active():
        return None
    keep = set()
    for it in store.items():
        if f.item_keys and it["key"] not in f.item_keys:
            continue
        y = it["year"]
        if f.year_from and (y is None or y < f.year_from):
            continue
        if f.year_to and (y is None or y > f.year_to):
            continue
        if f.author and not _contains(it["authors"], f.author):
            continue
        if f.tag and not _contains(it["tags"], f.tag):
            continue
        if f.collection and not _contains(it["collections"], f.collection):
            continue
        if f.item_type and f.item_type.casefold() not in it["item_type"].casefold():
            continue
        keep.add(it["key"])
    return keep


def _fts_query(query: str) -> str:
    # Each word as a quoted term (prefix match for longer words, which helps
    # with Polish inflection: "pracownik" matches "pracownikami"), OR-ed together.
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    terms = [f'"{w[:max(5, len(w) - 2)]}"*' if len(w) > 6 else f'"{w}"' for w in words]
    return " OR ".join(terms)


def search(store: Store, query: str, filters: Filters | None = None,
           limit: int = 8, per_item: int = 3) -> list[dict]:
    filters = filters or Filters()
    allowed = allowed_items(store, filters)
    if allowed is not None and not allowed:
        return []

    ids, keys, matrix = store.matrix()
    if len(ids) == 0:
        return []

    # Dense
    scores = matrix @ models.embed_query(query)
    if allowed is not None:
        mask = np.array([k in allowed for k in keys])
        scores = np.where(mask, scores, -np.inf)
    top = np.argsort(-scores)[:CANDIDATES]
    dense = [int(ids[i]) for i in top if np.isfinite(scores[i])]

    # Keyword
    fts = _fts_query(query)
    keyword = store.keyword_search(fts, allowed, CANDIDATES) if fts else []

    # Reciprocal Rank Fusion
    fused: dict[int, float] = {}
    for ranking in (dense, keyword):
        for rank, pid in enumerate(ranking):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
    cand = sorted(fused, key=fused.get, reverse=True)[:RERANK_TOP]
    passages = store.passages(cand)

    rr = models.rerank(query, [passages[p]["text"] for p in cand])
    if rr is not None:
        order = sorted(zip(cand, rr), key=lambda x: x[1], reverse=True)
    else:
        order = [(p, fused[p]) for p in cand]

    results, per = [], {}
    for pid, score in order:
        p = passages[pid]
        if per.get(p["item_key"], 0) >= per_item:
            continue
        per[p["item_key"]] = per.get(p["item_key"], 0) + 1
        p["score"] = score
        results.append(p)
        if len(results) >= limit:
            break
    return results
