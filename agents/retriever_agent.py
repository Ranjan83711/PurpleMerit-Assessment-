"""
retriever_agent.py — Catalog Retriever Agent: semantic search over ChromaDB.
"""
from crewai import Agent


def create_retriever_agent(llm, tools: list) -> Agent:
    """
    Catalog Retriever Agent: Searches the KUK vector store.
    - Runs targeted semantic queries
    - Returns cited excerpts
    - Assesses coverage (FULL / PARTIAL / NOT_FOUND)
    """
    return Agent(
        role="Catalog Retrieval Specialist",
        goal=(
            "Retrieve the most relevant course catalog information for the student's query. "
            "Always include citations (source, page, chunk ID) for every piece of information. "
            "If information is not in the catalog, explicitly report NOT_FOUND."
        ),
        backstory=(
            "You are a specialist in KUK academic catalogs with deep knowledge of how "
            "to search for course prerequisites, program requirements, and academic policies. "
            "You run multiple targeted search queries to ensure comprehensive retrieval, "
            "and you never invent information — only what the catalog says."
        ),
        tools=tools,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=5,
    )
