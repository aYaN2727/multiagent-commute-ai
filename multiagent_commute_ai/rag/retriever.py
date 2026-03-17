"""
rag/retriever.py
Query-time FAISS retrieval with the same embedding model used at index time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from rag.ingestion import load_index
from utils.logger import get_logger

logger = get_logger("rag.retriever")
settings = get_settings()


class PolicyRetriever:
    """
    Loads the FAISS index + chunk metadata once and handles all retrieve() calls.
    Use get_retriever() to obtain the singleton instance.
    """

    def __init__(self) -> None:
        logger.info("Loading FAISS index and embedding model…", extra={"agent_name": "retriever"})
        self._index: faiss.IndexFlatIP
        self._chunks: List[Dict[str, Any]]
        self._index, self._chunks = load_index()
        self._model: SentenceTransformer = SentenceTransformer(settings.EMBED_MODEL)
        logger.info("PolicyRetriever ready.", extra={"agent_name": "retriever"})

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed `query`, search FAISS for top_k nearest chunks.

        Returns a list of dicts:
            {"text": str, "score": float, "source": str, "chunk_index": int}
        """
        query_vec: np.ndarray = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append({
                "text": chunk["text"],
                "score": float(score),
                "source": chunk.get("source_file", "unknown"),
                "chunk_index": chunk.get("chunk_index", idx),
                "section": chunk.get("estimated_section", ""),
            })

        logger.debug(
            f"Retrieved {len(results)} chunks for query '{query[:60]}…'",
            extra={"agent_name": "retriever", "top_score": results[0]["score"] if results else 0.0},
        )
        return results

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a numbered context string for LLM prompts.
        """
        parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            section = chunk.get("section", "")
            header = f"[Excerpt {i}] Source: {source}"
            if section:
                header += f" | Section: {section}"
            parts.append(f"{header}\n{chunk['text'].strip()}")
        return "\n\n---\n\n".join(parts)


# ── Module-level singleton ────────────────────────────────────────────────────
_retriever_instance: Optional[PolicyRetriever] = None


def get_retriever() -> PolicyRetriever:
    """Lazy-initialised singleton PolicyRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PolicyRetriever()
    return _retriever_instance
