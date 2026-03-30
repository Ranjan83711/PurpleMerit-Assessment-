"""
Retriever Module
Handles document retrieval from ChromaDB with context enrichment.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


class CatalogRetriever:
    """Retrieves relevant catalog chunks for student queries."""
    
    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: ChromaVectorStore instance. If None, creates one.
        """
        if vector_store is None:
            from rag.vector_store import ChromaVectorStore
            from configs.model_config import (
                CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
            )
            self.vector_store = ChromaVectorStore(
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_model_name=EMBEDDING_MODEL_NAME,
            )
        else:
            self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        content_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question or search query
            top_k: Number of results
            content_type: Optional filter (prerequisite, course_description, etc.)
        
        Returns:
            List of relevant document chunks with metadata
        """
        results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
            content_type_filter=content_type,
        )
        
        # Add citation info to each result
        for i, result in enumerate(results):
            meta = result.get("metadata", {})
            result["citation"] = self._format_citation(meta, i + 1)
            result["rank"] = i + 1
        
        return results
    
    def retrieve_for_prerequisite(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Specialized retrieval for prerequisite queries.
        Retrieves from prerequisite and course_description content types.
        """
        # Get prerequisite-specific chunks
        prereq_results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
        )
        
        # Also search for related course descriptions
        # Extract course names/codes from query for targeted search
        course_query = query + " prerequisite requirement eligibility"
        course_results = self.vector_store.query(
            query_text=course_query,
            n_results=top_k // 2,
        )
        
        # Merge and deduplicate
        seen_texts = set()
        merged = []
        for result in prereq_results + course_results:
            text_key = result["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                merged.append(result)
        
        # Re-rank by relevance score
        merged.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Add citations
        for i, result in enumerate(merged[:top_k]):
            meta = result.get("metadata", {})
            result["citation"] = self._format_citation(meta, i + 1)
            result["rank"] = i + 1
        
        return merged[:top_k]
    
    def retrieve_for_planning(self, query: str, top_k: int = 12) -> List[Dict]:
        """
        Specialized retrieval for course planning queries.
        Retrieves program requirements, prerequisites, and course descriptions.
        """
        # Broad search for planning context
        results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
        )
        
        # Also get program requirement chunks
        program_query = query + " degree program requirement credits elective"
        program_results = self.vector_store.query(
            query_text=program_query,
            n_results=top_k // 2,
        )
        
        # Merge and deduplicate
        seen_texts = set()
        merged = []
        for result in results + program_results:
            text_key = result["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                merged.append(result)
        
        merged.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        for i, result in enumerate(merged[:top_k]):
            meta = result.get("metadata", {})
            result["citation"] = self._format_citation(meta, i + 1)
            result["rank"] = i + 1
        
        return merged[:top_k]
    
    def format_context_for_llm(self, results: List[Dict]) -> str:
        """
        Format retrieved results as context string for the LLM prompt.
        
        Returns a structured string with citations that the LLM can reference.
        """
        if not results:
            return "No relevant documents found in the catalog."
        
        context_parts = []
        for result in results:
            meta = result.get("metadata", {})
            citation = result.get("citation", "")
            text = result.get("text", "")
            score = result.get("relevance_score", 0)
            
            context_parts.append(
                f"[Source {result.get('rank', '?')}] "
                f"({meta.get('source_file', 'unknown')}, Page {meta.get('page_number', '?')}, "
                f"Type: {meta.get('content_type', 'general')}) "
                f"[Relevance: {score:.2f}]\n"
                f"{text}\n"
                f"Citation: {citation}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_citation(self, metadata: Dict, rank: int) -> str:
        """Format a citation string from metadata."""
        source = metadata.get("source_file", "Unknown Source")
        page = metadata.get("page_number", "?")
        section = metadata.get("section_heading", "")
        content_type = metadata.get("content_type", "general")
        
        citation = f"[{rank}] {source}, Page {page}"
        if section:
            citation += f", Section: {section}"
        citation += f" ({content_type})"
        
        return citation


if __name__ == "__main__":
    retriever = CatalogRetriever()
    
    test_queries = [
        "What are the prerequisites for Computer Science courses?",
        "What are the degree requirements for B.Tech?",
        "What is the grading policy?",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.retrieve(query, top_k=3)
        for r in results:
            print(f"  [{r['rank']}] Score: {r['relevance_score']:.3f} | {r['citation']}")
            print(f"      {r['text'][:150]}...")
