"""
vector_store.py — ChromaDB vector store management.
Handles building, persisting, and loading the course catalog index.
"""
import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma


class CourseVectorStore:
    """
    ChromaDB-backed vector store for the KUK course catalog.
    Supports cosine similarity search with metadata filtering.
    """

    def __init__(
        self,
        persist_directory: str = "vectorstore/chroma_db",
        collection_name: str = "kuk_course_catalog",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._store: Optional[Chroma] = None

    def build(self, chunks: List[Document], embeddings) -> "CourseVectorStore":
        """
        Build ChromaDB index from document chunks.
        Persists to disk for reuse across sessions.
        """
        os.makedirs(self.persist_directory, exist_ok=True)
        print(
            f"[VectorStore] Building ChromaDB with {len(chunks)} chunks..."
        )

        self._store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[VectorStore] Index built and persisted → {self.persist_directory}"
        )
        return self

    def load(self, embeddings) -> "CourseVectorStore":
        """Load existing ChromaDB index from disk."""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"ChromaDB not found at {self.persist_directory}. "
                "Run `python main.py --build-index` first."
            )
        print(f"[VectorStore] Loading ChromaDB from {self.persist_directory}...")
        self._store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name,
        )
        count = self._store._collection.count()
        print(f"[VectorStore] Loaded {count} chunks from ChromaDB.")
        return self

    def as_retriever(self, k: int = 6, score_threshold: float = 0.3):
        """
        Return a LangChain retriever with similarity score threshold.
        k=6 balances recall vs. context window usage.
        """
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Call build() or load() first.")

        return self._store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

    def similarity_search_with_score(self, query: str, k: int = 6):
        """Direct similarity search returning (doc, score) tuples."""
        if self._store is None:
            raise RuntimeError("Vector store not initialized.")
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def index_exists(self) -> bool:
        """Check if an existing index is available."""
        chroma_files = os.path.join(self.persist_directory, "chroma.sqlite3")
        return os.path.exists(chroma_files)
