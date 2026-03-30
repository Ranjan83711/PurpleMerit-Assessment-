"""
Explanation / Planner Agent
Generates course plans and answers with citations using retrieved catalog data.
"""

from crewai import Agent
from rag.prompt_templates import PLANNER_AGENT_SYSTEM_PROMPT


def create_explanation_agent(llm, tools=None) -> Agent:
    """
    Create the Explanation/Planner Agent.
    
    Role: Generate the final response - course plans, prerequisite answers,
    or policy explanations - with full citations and structured output.
    """
    agent_tools = tools or []
    
    return Agent(
        role="Course Planning Expert",
        goal=(
            "Generate accurate, well-structured responses to student queries "
            "about course planning, prerequisites, and academic policies at "
            "Kurukshetra University. Every response MUST follow the required "
            "output format with: Answer/Plan, Why (requirements/prereqs satisfied), "
            "Citations, Clarifying questions (if needed), and Assumptions/Not in catalog. "
            "Use transitive reasoning for prerequisite chains. NEVER state "
            "anything as fact without a citation from the catalog."
        ),
        backstory=(
            "You are a senior academic advisor at Kurukshetra University with "
            "deep knowledge of course planning and prerequisite systems. You "
            "understand transitive prerequisites (if A needs B, and B needs C, "
            "a student must complete C before they can eventually take A). "
            "You always ground your advice in official catalog documents and "
            "provide citations. When information is not available in the "
            "catalog, you honestly say so and suggest where to find it "
            "(advisor, department, registrar). You never guess or make up "
            "requirements."
        ),
        tools=agent_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        system_prompt=PLANNER_AGENT_SYSTEM_PROMPT,
    )
