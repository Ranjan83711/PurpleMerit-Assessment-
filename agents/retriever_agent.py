"""
Retriever Agent
Searches catalog vector store for relevant course and program information.
"""

from crewai import Agent
from rag.prompt_templates import RETRIEVER_AGENT_SYSTEM_PROMPT


def create_retriever_agent(llm, tools=None) -> Agent:
    """
    Create the Catalog Retriever Agent.
    
    Role: Search the vector store for relevant catalog information
    including prerequisites, course descriptions, program requirements,
    and academic policies.
    """
    agent_tools = tools or []
    
    return Agent(
        role="Catalog Research Specialist",
        goal=(
            "Search the Kurukshetra University course catalog thoroughly to find "
            "ALL relevant information for the student's query. This includes "
            "prerequisites, co-requisites, program requirements, course descriptions, "
            "credit requirements, and academic policies. Return comprehensive "
            "results with full citation information (source file, page number, section)."
        ),
        backstory=(
            "You are a meticulous academic researcher who knows how to navigate "
            "university catalogs and find exactly the right information. You "
            "understand that prerequisite chains can be complex and that programs "
            "have many overlapping requirements. You always search broadly to "
            "ensure no relevant information is missed, and you ALWAYS include "
            "citation details for every piece of information you find."
        ),
        tools=agent_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        system_prompt=RETRIEVER_AGENT_SYSTEM_PROMPT,
    )
