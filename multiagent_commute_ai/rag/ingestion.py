"""
rag/ingestion.py
Document loading -> chunking -> embedding -> FAISS index build pipeline.

Usage:
    python -m rag.ingestion
    python -m rag.ingestion --embed-model models/finetuned_embedder
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is on path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("rag.ingestion")
settings = get_settings()

CHUNKS_JSON_PATH    = Path(settings.FAISS_INDEX_PATH).with_name("policy_chunks.json")
CHUNKS_PICKLE_PATH  = Path(settings.FAISS_INDEX_PATH).with_name("policy_chunks.pkl")


def _extract_text_from_file(path: Path) -> str:
    """Extract plain text from a PDF or TXT file."""
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _load_documents() -> List[Dict[str, Any]]:
    """Load all .txt and .pdf files from the policy docs directory."""
    docs_dir = Path(settings.POLICY_DOCS_DIR)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Policy docs directory not found: {docs_dir}")

    exts = {".txt", ".pdf"}
    files = [p for p in sorted(docs_dir.rglob("*")) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No .pdf or .txt files found in {docs_dir}")

    docs = []
    for p in files:
        text = _extract_text_from_file(p)
        docs.append({"file_name": p.name, "file_path": str(p), "text": text})
        logger.info(f"Loaded {p.name} ({len(text)} chars)", extra={"agent_name": "ingestion"})

    logger.info(f"Loaded {len(docs)} documents from {docs_dir}", extra={"agent_name": "ingestion"})
    return docs


def _chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into overlapping fixed-size chunks with metadata."""
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    step = size - overlap

    chunks: List[Dict[str, Any]] = []
    idx = 0
    for doc in docs:
        text = doc["text"]
        words = text.split()
        start = 0
        while start < len(words):
            window = words[start: start + size]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": idx,
                    "source_file": doc["file_name"],
                    "file_path": doc["file_path"],
                    "estimated_section": _infer_section(chunk_text),
                })
                idx += 1
            start += step

    logger.info(f"Created {len(chunks)} chunks", extra={"agent_name": "ingestion"})
    return chunks


def _infer_section(text: str) -> str:
    """Heuristic: extract nearest SECTION heading from chunk text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SECTION") or stripped.startswith("==="):
            return stripped[:80]
    return "General"


def _embed_chunks(chunks: List[Dict[str, Any]], embed_model: str | None = None) -> np.ndarray:
    """Embed all chunk texts using sentence-transformers; returns unit-normalised matrix."""
    model_name = embed_model or settings.EMBED_MODEL
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks with {model_name}",
                extra={"agent_name": "ingestion"})
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-length -> inner product == cosine sim
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create and populate a FAISS IndexFlatIP (exact inner product search)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}",
                extra={"agent_name": "ingestion"})
    return index


def build_index(embed_model: str | None = None) -> None:
    """
    Full pipeline: load -> chunk -> embed -> FAISS index -> save.
    Called when this module is executed directly.

    Args:
        embed_model: Override embedding model path (e.g. 'models/finetuned_embedder').
                     Defaults to settings.EMBED_MODEL.
    """
    # Ensure output directories exist
    index_path = Path(settings.FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_documents()
    chunks = _chunk_documents(docs)
    embeddings = _embed_chunks(chunks, embed_model=embed_model)
    index = _build_faiss_index(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}", extra={"agent_name": "ingestion"})

    # Save chunk metadata as JSON
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunk data (JSON) saved to {CHUNKS_JSON_PATH}", extra={"agent_name": "ingestion"})

    # Save chunk metadata as pickle (used by ml/finetune_embeddings.py)
    with open(str(CHUNKS_PICKLE_PATH), "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Chunk data (pickle) saved to {CHUNKS_PICKLE_PATH}", extra={"agent_name": "ingestion"})

    print(f"\nIndex built with {len(chunks)} chunks. FAISS index -> {index_path}")


def load_index() -> tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Load the pre-built FAISS index and companion chunk JSON from disk.
    Returns (faiss_index, chunks_list).
    """
    index_path = Path(settings.FAISS_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run: python -m rag.ingestion"
        )
    if not CHUNKS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Chunk JSON not found at {CHUNKS_JSON_PATH}. Run: python -m rag.ingestion"
        )

    index = faiss.read_index(str(index_path))
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(
        f"Loaded FAISS index ({index.ntotal} vectors) and {len(chunks)} chunks",
        extra={"agent_name": "ingestion"},
    )
    return index, chunks


def ingest_documents() -> List[Dict[str, Any]]:
    """
    Load and chunk all policy documents. Returns the list of chunk dicts.
    Used by ml/finetune_embeddings.py to get chunks without rebuilding the index.
    """
    docs = _load_documents()
    return _chunk_documents(docs)


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="Build FAISS index from policy PDFs")
    _parser.add_argument(
        "--embed-model", default=None,
        help="Embedding model path or name (default: settings.EMBED_MODEL). "
             "Use 'models/finetuned_embedder' after running ml/finetune_embeddings.py."
    )
    _args = _parser.parse_args()
    build_index(embed_model=_args.embed_model)
