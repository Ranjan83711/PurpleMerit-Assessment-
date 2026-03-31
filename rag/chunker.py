"""
chunker.py — Split cleaned documents into retrieval-optimized chunks.

Strategy:
- Chunk size: 800 chars (~200 tokens) — large enough to capture full course
  descriptions with prerequisites, small enough for precise retrieval.
- Overlap: 150 chars (~18%) — ensures prerequisite conditions straddling
  chunk boundaries are captured in both adjacent chunks.
- RecursiveCharacterTextSplitter: respects paragraph → sentence → word
  boundaries, keeping course entries intact.
- Each chunk inherits source metadata + gets a unique chunk_id for citation.
"""
import json
import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CatalogChunker:
    """
    Splits catalog documents into overlapping chunks with citation metadata.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, adding chunk-level metadata.
        Each chunk gets a unique chunk_id: {source_name}_p{page}_c{chunk_idx}
        """
        all_chunks = []
        chunk_counter = 0

        for doc in documents:
            splits = self.splitter.split_documents([doc])
            for i, chunk in enumerate(splits):
                source_name = chunk.metadata.get("source_name", "kuk_catalog")
                page = chunk.metadata.get("page", "?")

                chunk_id = f"{source_name}_p{page}_c{i}"
                chunk.metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_counter,
                        "chunk_within_page": i,
                        "char_count": len(chunk.page_content),
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        print(
            f"[Chunker] {len(documents)} pages → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def save_chunks(self, chunks: List[Document], output_path: str):
        """Save chunks JSON for inspection and reproducibility."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [
            {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Chunker] Saved {len(chunks)} chunks → {output_path}")

    def load_chunks(self, input_path: str) -> List[Document]:
        """Load previously saved chunks (skip re-ingestion)."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
