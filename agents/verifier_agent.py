"""
verifier_agent.py — Verifier/Auditor Agent: citation and logic checker.
"""
from crewai import Agent


def create_verifier_agent(llm) -> Agent:
    """
    Verifier/Auditor Agent: Last gate before output is returned to student.
    - Checks citation completeness and validity
    - Flags hallucinations (claims without evidence)
    - Verifies prerequisite logic correctness
    - Can trigger REWRITE or ESCALATE_TO_ADVISOR
    """
    return Agent(
        role="Academic Output Verifier and Auditor",
        goal=(
            "Audit the proposed course plan or eligibility decision for citation completeness, "
            "logical correctness, and absence of hallucinated information. "
            "Every factual claim must have a citation. Flag any unsupported statements. "
            "Output PASS, NEEDS_REVISION, or ESCALATE_TO_ADVISOR."
        ),
        backstory=(
            "You are a meticulous academic compliance officer at KUK. Your job is to ensure "
            "that every AI-generated course recommendation is 100% grounded in the official catalog. "
            "You have seen AI systems hallucinate course names, invent prerequisites, and make up "
            "policies. Your job is to catch all of that before it reaches the student. "
            "You are the last line of defense against misinformation."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=3,
    )
