"""
Rule Extractor Agent
Extracts and structures academic rules, prerequisites, and policies.
"""

from crewai import Agent


def create_rule_extractor_agent(llm, tools=None) -> Agent:
    """
    Create the Rule Extractor Agent.
    
    Role: Extract structured rules from catalog text - prerequisite chains,
    either/or requirements, grade requirements, co-requisites, exceptions.
    """
    agent_tools = tools or []
    
    return Agent(
        role="Academic Rule Extractor",
        goal=(
            "Extract and structure all academic rules from the retrieved catalog "
            "information. This includes: prerequisite chains (A → B → C), "
            "either/or requirements, minimum grade requirements, co-requisites, "
            "exceptions (instructor consent), credit requirements, and any "
            "special conditions. Present rules in a clear, structured format "
            "that the planner can use for decision-making."
        ),
        backstory=(
            "You are an expert at reading academic catalogs and extracting "
            "the precise rules that govern course enrollment and program "
            "completion. You understand the nuances of academic language - "
            "the difference between 'recommended' and 'required', between "
            "'prerequisite' and 'co-requisite', between 'or' and 'and'. "
            "You pay special attention to transitive prerequisites: if "
            "Course C requires Course B, and Course B requires Course A, "
            "then a student needs Course A before they can eventually take "
            "Course C."
        ),
        tools=agent_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
