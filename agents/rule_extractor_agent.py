"""
rule_extractor_agent.py — Parses retrieved text into structured rules.
"""
from crewai import Agent


def create_rule_extractor_agent(llm, tools: list) -> Agent:
    """
    Rule Extractor Agent: Converts raw catalog text into structured prerequisite rules.
    - Handles AND/OR logic, grade minimums, co-requisites, exceptions
    - Critical for transitive prerequisite chain resolution
    """
    return Agent(
        role="Academic Rule Extraction Specialist",
        goal=(
            "Parse retrieved catalog text into structured prerequisite rules, program requirements, "
            "and academic policies. Handle complex logic: AND/OR prerequisites, grade minimums, "
            "co-requisites, and exception clauses. Preserve all citations."
        ),
        backstory=(
            "You are an expert at reading university catalog language — the dense, often "
            "inconsistent text that describes course requirements. You can spot 'either/or' "
            "prerequisites, identify minimum grade clauses, and untangle co-requisite vs "
            "prerequisite distinctions. You always cite the exact source of every rule you extract."
        ),
        tools=tools,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=4,
    )
