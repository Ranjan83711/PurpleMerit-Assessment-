"""
retriever.py — Retrieval logic with citation formatting.
Wraps ChromaDB search with citation metadata extraction.
"""
from typing import List, Dict, Any, Tuple
from langchain.schema import Document


class CatalogRetriever:
    """
    Retrieves relevant catalog chunks and formats them with citations.
    Used directly by CrewAI tools and also standalone.
    """

    def __init__(self, vector_store, k: int = 6, score_threshold: float = 0.3):
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        Returns list of dicts with content + citation info.
        """
        results = self.vector_store.similarity_search_with_score(
            query=query, k=self.k
        )

        retrieved = []
        for doc, score in results:
            if score >= self.score_threshold:
                retrieved.append(
                    {
                        "content": doc.page_content,
                        "citation": self._format_citation(doc),
                        "metadata": doc.metadata,
                        "score": round(score, 4),
                    }
                )

        # Sort by score descending
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved

    def retrieve_as_context(self, query: str) -> Tuple[str, List[str]]:
        """
        Returns (formatted_context_string, list_of_citations).
        Ready for injection into prompts.
        """
        results = self.retrieve(query)

        if not results:
            return "No relevant information found in the catalog.", []

        context_parts = []
        citations = []

        for i, r in enumerate(results, start=1):
            context_parts.append(
                f"[CHUNK {i} | {r['citation']}]\n{r['content']}"
            )
            citations.append(r["citation"])

        context = "\n\n---\n\n".join(context_parts)
        return context, citations

    def _format_citation(self, doc: Document) -> str:
        """
        Format a citation string from document metadata.
        Format: KUK_Catalog_2011 | Page 42 | Chunk kuk_prospectus_2011_p42_c0
        """
        meta = doc.metadata
        source_name = meta.get("source_name", "KUK_Catalog")
        page = meta.get("page", "?")
        chunk_id = meta.get("chunk_id", "unknown")
        return f"{source_name} | Page {page} | Chunk {chunk_id}"
