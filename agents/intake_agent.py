"""
Intake Agent
Collects and normalizes student information, identifies missing data,
and generates clarifying questions before planning.
"""

from crewai import Agent
from rag.prompt_templates import INTAKE_AGENT_SYSTEM_PROMPT


def create_intake_agent(llm, tools=None) -> Agent:
    """
    Create the Intake Agent.
    
    Role: Collect missing student info, normalize the student profile,
    determine query type, and generate clarifying questions.
    """
    agent_tools = tools or []
    
    return Agent(
        role="Student Intake Specialist",
        goal=(
            "Analyze the student's query, extract all provided information "
            "(completed courses, grades, target program, term preferences), "
            "identify what critical information is missing, and generate "
            "specific clarifying questions if needed. Normalize the student "
            "profile for downstream agents."
        ),
        backstory=(
            "You are an experienced academic advisor assistant at Kurukshetra "
            "University (KUK). You excel at understanding student needs, "
            "extracting relevant information from their queries, and knowing "
            "exactly what additional details are needed for accurate course "
            "planning. You are thorough but efficient - asking only necessary "
            "questions."
        ),
        tools=agent_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        system_prompt=INTAKE_AGENT_SYSTEM_PROMPT,
    )
