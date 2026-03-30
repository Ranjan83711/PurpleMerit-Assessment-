"""
Embedding Module
Creates embeddings using sentence-transformers (free, local, no API key).
Model: all-MiniLM-L6-v2 (fast, 384 dimensions, good quality)
"""

from typing import List
from loguru import logger


class EmbeddingModel:
    """Wrapper around sentence-transformers for generating embeddings."""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        if cls._instance is None or cls._model is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        if self._model is not None:
            return
        
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"  Model loaded. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension


def get_embedding_function(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
    """Get a ChromaDB-compatible embedding function."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
    )


if __name__ == "__main__":
    model = EmbeddingModel()
    
    test_texts = [
        "CS101 Introduction to Computer Science requires no prerequisites.",
        "CS301 Data Structures requires CS201 as a prerequisite.",
        "The minimum grade required is C+ for all prerequisite courses.",
    ]
    
    embeddings = model.embed_texts(test_texts)
    print(f"\n📊 Embedding Summary:")
    print(f"   Model: {model.model_name}")
    print(f"   Dimension: {model.dimension}")
    print(f"   Texts embedded: {len(embeddings)}")
    print(f"   Sample vector (first 5): {embeddings[0][:5]}")
