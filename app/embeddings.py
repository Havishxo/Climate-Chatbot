"""
embeddings.py — Embedding model abstraction.

Priority:
  1. IBM Granite Embedding via Watsonx.ai
  2. sentence-transformers (local, no API key needed)
  3. Simple TF-IDF fallback (offline / zero-dependency)
"""

import logging
import numpy as np
from typing import List
from app.config import (
    USE_IBM, IBM_API_KEY, IBM_PROJECT_ID, IBM_URL,
    GRANITE_EMBEDDING_MODEL, EMBEDDING_MODEL_LOCAL,
)

logger = logging.getLogger(__name__)


# ── IBM Granite Embeddings ─────────────────────────────────────────────────────
def _get_ibm_embeddings():
    try:
        from langchain_ibm import WatsonxEmbeddings
        emb = WatsonxEmbeddings(
            model_id=GRANITE_EMBEDDING_MODEL,
            url=IBM_URL,
            apikey=IBM_API_KEY,
            project_id=IBM_PROJECT_ID,
        )
        logger.info("✅ IBM Granite Embeddings loaded: %s", GRANITE_EMBEDDING_MODEL)
        return emb
    except Exception as exc:
        logger.warning("IBM Embeddings init failed: %s", exc)
        return None


# ── Sentence-Transformers Embeddings (local fallback) ─────────────────────────
def _get_local_embeddings():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_LOCAL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✅ Local HuggingFace Embeddings loaded: %s", EMBEDDING_MODEL_LOCAL)
        return emb
    except Exception as exc:
        logger.warning("HuggingFace Embeddings init failed: %s", exc)
        return None


# ── TF-IDF fallback (zero external dependency) ────────────────────────────────
class TFIDFEmbeddings:
    """
    Very simple TF-IDF vectoriser used when no embedding model is available.
    Sufficient for a small knowledge base demo; replace with real embeddings in production.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._fitted = False
        self._vocab: dict = {}

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"[a-z]+", text.lower())

    def _build_vocab(self, texts: List[str]):
        tokens = set()
        for t in texts:
            tokens.update(self._tokenize(t))
        self._vocab = {tok: i % self.dim for i, tok in enumerate(sorted(tokens))}
        self._fitted = True

    def _vectorize(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokenize(text):
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1.0
        norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self._build_vocab(texts)
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)


_embeddings_instance = None


def get_embeddings():
    """Return the best available embeddings (cached singleton)."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    if USE_IBM:
        _embeddings_instance = _get_ibm_embeddings()
    if _embeddings_instance is None:
        _embeddings_instance = _get_local_embeddings()
    if _embeddings_instance is None:
        logger.warning("⚠️  Using TF-IDF embeddings (demo mode — no real semantic search).")
        _embeddings_instance = TFIDFEmbeddings(dim=256)

    return _embeddings_instance
