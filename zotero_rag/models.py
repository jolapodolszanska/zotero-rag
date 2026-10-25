"""Embedding and reranking models, loaded lazily on first use."""

import logging
import sys
from functools import lru_cache

import numpy as np

from . import config

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    log.info("Ładuję model embeddingów %s", config.EMBED_MODEL)
    model = SentenceTransformer(config.EMBED_MODEL, device=_device())
    model.max_seq_length = config.MAX_SEQ_LENGTH
    return model


@lru_cache(maxsize=1)
def _reranker():
    if not config.RERANK_MODEL:
        return None
    from sentence_transformers import CrossEncoder
    log.info("Ładuję reranker %s", config.RERANK_MODEL)
    return CrossEncoder(config.RERANK_MODEL, max_length=config.MAX_SEQ_LENGTH, device=_device())


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_documents(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return _embedder().encode([config.DOC_PREFIX + t for t in texts], batch_size=16,
                              normalize_embeddings=True, show_progress_bar=show_progress,
                              convert_to_numpy=True)


def embed_query(text: str) -> np.ndarray:
    return _embedder().encode([config.QUERY_PREFIX + text], normalize_embeddings=True,
                              convert_to_numpy=True)[0]


def rerank(query: str, texts: list[str]) -> list[float] | None:
    model = _reranker()
    if model is None or not texts:
        return None
    return [float(s) for s in model.predict([(query, t) for t in texts], batch_size=16)]


def setup_logging():
    # stdout is reserved for the MCP protocol: log to stderr only
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
