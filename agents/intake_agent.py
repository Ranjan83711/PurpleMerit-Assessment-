"""
intake_agent.py — Intake Agent: collects and normalizes student info.
"""
from crewai import Agent
from rag.prompt_templates import INTAKE_AGENT_PROMPT


def create_intake_agent(llm) -> Agent:
    """
    Intake Agent: First agent in the pipeline.
    - Parses student queries for key planning info
    - Identifies missing information
    - Generates clarifying questions (max 5)
    - Normalizes the student profile
    """
    return Agent(
        role="Academic Intake Coordinator",
        goal=(
            "Collect and normalize complete student information needed for course planning. "
            "Identify missing fields and ask targeted clarifying questions. "
            "Produce a structured student profile ready for the planning pipeline."
        ),
        backstory=(
            "You are an experienced academic advisor intake specialist at Kurukshetra University. "
            "You know exactly what information is needed to plan a student's next semester: "
            "their completed courses, grades, target program, term, and credit limits. "
            "You ask precise clarifying questions and never proceed with incomplete information."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=3,
    )
