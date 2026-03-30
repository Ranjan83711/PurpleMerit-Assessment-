"""
Vector Store Module
Manages ChromaDB for storing and querying document embeddings.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """ChromaDB vector store for course catalog documents."""
    
    def __init__(
        self,
        persist_directory: str = "./vectorstore/chroma_db",
        collection_name: str = "kuk_course_catalog",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize ChromaDB client
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get embedding function
        from rag.embedder import get_embedding_function
        self.embedding_fn = get_embedding_function(embedding_model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(
            f"ChromaDB initialized: collection='{collection_name}', "
            f"docs={self.collection.count()}, path='{persist_directory}'"
        )
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', and metadata
            batch_size: Number of documents to add per batch
        """
        if not chunks:
            logger.warning("No chunks to add.")
            return
        
        # Check for existing documents to avoid duplicates
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.info(f"Collection already has {existing_count} documents. Adding new ones...")
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            
            if not text.strip():
                continue
            
            ids.append(chunk_id)
            documents.append(text)
            metadatas.append({
                "source_file": chunk.get("source_file", "unknown"),
                "page_number": chunk.get("page_number", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "content_type": chunk.get("content_type", "general"),
                "section_heading": chunk.get("section_heading", ""),
                "word_count": chunk.get("word_count", 0),
            })
        
        # Add in batches
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            try:
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
                total_added += len(batch_ids)
                logger.info(f"  Added batch {i // batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                logger.error(f"  Failed to add batch: {e}")
        
        logger.info(f"✅ Total documents in collection: {self.collection.count()}")
    
    def query(
        self,
        query_text: str,
        n_results: int = 8,
        content_type_filter: Optional[str] = None,
        source_file_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: Search query
            n_results: Number of results to return
            content_type_filter: Filter by content type
            source_file_filter: Filter by source file
        
        Returns:
            List of result dicts with text, metadata, and distance
        """
        where_filter = None
        if content_type_filter:
            where_filter = {"content_type": content_type_filter}
        elif source_file_filter:
            where_filter = {"source_file": source_file_filter}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
        
        # Format results
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "relevance_score": 1 - (results["distances"][0][i] if results["distances"] else 0),
                })
        
        return formatted
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Sample some documents for stats
        if count > 0:
            sample = self.collection.peek(min(10, count))
            content_types = set()
            source_files = set()
            for meta in sample.get("metadatas", []):
                content_types.add(meta.get("content_type", "unknown"))
                source_files.add(meta.get("source_file", "unknown"))
            
            return {
                "total_documents": count,
                "sample_content_types": list(content_types),
                "sample_source_files": list(source_files),
                "collection_name": self.collection_name,
            }
        
        return {"total_documents": 0, "collection_name": self.collection_name}
    
    def reset_collection(self):
        """Delete and recreate the collection."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset complete.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.model_config import (
        CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
        EMBEDDING_MODEL_NAME, PROCESSED_DATA_DIR,
    )
    
    store = ChromaVectorStore(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )
    
    # Load chunks if available
    chunks_path = PROCESSED_DATA_DIR / "chunks.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        store.add_documents(chunks)
        stats = store.get_collection_stats()
        print(f"\n📊 Vector Store Stats: {json.dumps(stats, indent=2)}")
    else:
        print(f"No chunks found at {chunks_path}. Run chunker.py first.")
        stats = store.get_collection_stats()
        print(f"Current stats: {json.dumps(stats, indent=2)}")
