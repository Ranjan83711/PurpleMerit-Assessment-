"""
vector_search_tool.py — CrewAI-compatible tool for vector search.
Wraps CatalogRetriever as a CrewAI BaseTool.
"""
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class VectorSearchInput(BaseModel):
    query: str = Field(description="Search query to find relevant catalog information")
    k: int = Field(default=6, description="Number of chunks to retrieve")


class VectorSearchTool(BaseTool):
    """
    CrewAI tool for semantic search over the KUK course catalog ChromaDB.
    Returns relevant text chunks with citations.
    """

    name: str = "catalog_vector_search"
    description: str = (
        "Search the KUK course catalog vector database for information about courses, "
        "prerequisites, program requirements, and academic policies. "
        "Returns relevant text excerpts with citations (source, page, chunk ID). "
        "Use this for ANY question about course requirements or prerequisites."
    )
    args_schema: Type[BaseModel] = VectorSearchInput

    # Injected at runtime
    retriever: object = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str, k: int = 6) -> str:
        """Execute semantic search and return formatted results with citations."""
        if self.retriever is None:
            return "ERROR: Retriever not initialized. Contact system administrator."

        results = self.retriever.retrieve(query)

        if not results:
            return (
                "NO_RESULTS: No relevant information found in the catalog for this query. "
                "The information may not be in the available catalog documents."
            )

        output_parts = [f"Found {len(results)} relevant catalog excerpts:\n"]
        for i, r in enumerate(results, start=1):
            output_parts.append(
                f"[{i}] CITATION: {r['citation']}\n"
                f"    RELEVANCE_SCORE: {r['score']}\n"
                f"    CONTENT:\n{r['content']}\n"
            )

        return "\n".join(output_parts)
