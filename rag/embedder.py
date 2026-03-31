"""
embedder.py — Embedding model wrapper.
Uses all-MiniLM-L6-v2: fast, free, 384-dim, excellent for semantic search.
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Returns initialized HuggingFace embeddings.
    Model: all-MiniLM-L6-v2
      - 384 dimensions
      - ~80MB, runs on CPU
      - Excellent semantic similarity for English academic text
      - normalize_embeddings=True ensures cosine similarity works correctly
    """
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    print(f"[Embedder] Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[Embedder] Embedding model ready.")
    return embeddings
