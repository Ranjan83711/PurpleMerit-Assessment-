"""
Vector Search Tool for CrewAI Agents
Allows agents to search the ChromaDB catalog vector store.
"""

import sys
from pathlib import Path
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

sys.path.insert(0, str(Path(__file__).parent.parent))


class VectorSearchInput(BaseModel):
    """Input schema for vector search tool."""
    query: str = Field(..., description="The search query to find relevant catalog information")
    top_k: int = Field(default=8, description="Number of results to return")
    search_type: str = Field(
        default="general",
        description="Type of search: 'general', 'prerequisite', or 'planning'"
    )


class VectorSearchTool(BaseTool):
    """Tool to search the KUK course catalog vector store."""
    
    name: str = "catalog_search"
    description: str = (
        "Search the Kurukshetra University (KUK) course catalog for information about "
        "courses, prerequisites, degree requirements, academic policies, and more. "
        "Returns relevant text excerpts with citations (source file, page number, section). "
        "Use search_type='prerequisite' for prerequisite queries, 'planning' for course "
        "planning, or 'general' for other queries."
    )
    args_schema: Type[BaseModel] = VectorSearchInput
    
    _retriever: object = None
    
    def __init__(self, retriever=None, **kwargs):
        super().__init__(**kwargs)
        if retriever:
            self._retriever = retriever
        else:
            from rag.retriever import CatalogRetriever
            self._retriever = CatalogRetriever()
    
    def _run(self, query: str, top_k: int = 8, search_type: str = "general") -> str:
        """Execute the vector search."""
        try:
            if search_type == "prerequisite":
                results = self._retriever.retrieve_for_prerequisite(query, top_k=top_k)
            elif search_type == "planning":
                results = self._retriever.retrieve_for_planning(query, top_k=top_k)
            else:
                results = self._retriever.retrieve(query, top_k=top_k)
            
            if not results:
                return "No relevant information found in the catalog for this query."
            
            return self._retriever.format_context_for_llm(results)
        
        except Exception as e:
            return f"Error searching catalog: {str(e)}"
