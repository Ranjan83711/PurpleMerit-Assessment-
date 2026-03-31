"""
explanation_agent.py — Planner/Explanation Agent: produces the course plan.
"""
from crewai import Agent


def create_explanation_agent(llm) -> Agent:
    """
    Planner / Explanation Agent: Synthesizes evidence into a course plan.
    - Produces the final Answer/Plan output
    - Justifies every course recommendation with citations
    - Lists all risks and assumptions
    """
    return Agent(
        role="Course Planning Advisor",
        goal=(
            "Produce a detailed, citation-backed course plan for the student's next term. "
            "Every recommended course must have a clear justification grounded in the catalog. "
            "Flag all assumptions and information not verifiable from the catalog."
        ),
        backstory=(
            "You are a senior academic advisor at KUK who specializes in helping students "
            "navigate complex degree requirements. You create clear, actionable semester plans "
            "that respect prerequisites, credit limits, and program rules. "
            "You never recommend a course without confirming the student meets its prerequisites, "
            "and you always tell students when you're making an assumption."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=4,
    )
