"""
tasks.py — CrewAI task definitions for the course planning pipeline.
Tasks are chained: each task's output feeds into the next.
"""
from crewai import Task
from rag.prompt_templates import (
    INTAKE_AGENT_PROMPT,
    RETRIEVER_AGENT_PROMPT,
    RULE_EXTRACTOR_PROMPT,
    PLANNER_AGENT_PROMPT,
    VERIFIER_AGENT_PROMPT,
)


def create_intake_task(agent, student_query: str) -> Task:
    """Task 1: Parse and normalize student information."""
    return Task(
        description=f"""
Analyze the following student query and extract all available information.
Identify what is missing and generate clarifying questions if needed.

STUDENT QUERY:
{student_query}

{INTAKE_AGENT_PROMPT}
""",
        expected_output=(
            "A structured student profile with: completed_courses, target_program, "
            "target_term, max_credits, grades (if provided), catalog_year (if mentioned). "
            "Plus a list of missing fields and clarifying questions (max 5) if info is incomplete. "
            "STATUS must be either COMPLETE or NEEDS_INFO."
        ),
        agent=agent,
    )


def create_retrieval_task(agent, student_query: str, student_profile: str = "") -> Task:
    """Task 2: Retrieve relevant catalog information."""
    return Task(
        description=f"""
Search the KUK course catalog vector database for all information relevant to this student's
course planning needs. Run multiple targeted queries to get comprehensive coverage.

STUDENT QUERY: {student_query}
STUDENT PROFILE: {student_profile}

Search for:
1. Prerequisites and requirements for any courses mentioned
2. Program/degree requirements for the student's major
3. Relevant academic policies (credit limits, grading rules, repeat policies)
4. Any related courses in the prerequisite chain

{RETRIEVER_AGENT_PROMPT}

CRITICAL: Every piece of information MUST include its citation (source | page | chunk_id).
If information is not found, explicitly say "NOT FOUND IN CATALOG".
""",
        expected_output=(
            "A list of retrieved catalog excerpts, each with: "
            "CONTENT (exact text), CITATION (source | page | chunk_id), RELEVANCE (HIGH/MEDIUM/LOW). "
            "Plus COVERAGE_ASSESSMENT (FULL/PARTIAL/NOT_FOUND) and MISSING_INFORMATION list."
        ),
        agent=agent,
    )


def create_rule_extraction_task(agent, retrieved_evidence: str = "") -> Task:
    """Task 3: Parse retrieved text into structured rules."""
    return Task(
        description=f"""
Parse the retrieved catalog text into structured prerequisite rules, program requirements,
and academic policies.

RETRIEVED EVIDENCE:
{retrieved_evidence if retrieved_evidence else "[Will be provided from previous task output]"}

{RULE_EXTRACTOR_PROMPT}

Pay special attention to:
- AND vs OR prerequisites (e.g., "Course A AND Course B" vs "Course A OR Course B")
- Co-requisites (must be taken concurrently vs. can be completed before)
- Grade minimums (e.g., "C or better in prerequisite")
- Exceptions ("with instructor consent", "with departmental approval")
- Either/or program requirements (e.g., "complete either X or Y")

Preserve ALL citations from the retrieved evidence.
""",
        expected_output=(
            "Structured rules in YAML-like format: prerequisites (with AND/OR logic, "
            "min_grade, exceptions, citations), co-requisites, program_requirements, "
            "and academic policies. Every rule must have its citation."
        ),
        agent=agent,
    )


def create_planning_task(
    agent,
    student_profile: str,
    structured_rules: str = "",
    retrieved_evidence: str = "",
    query_type: str = "plan",  # "plan" or "eligibility"
) -> Task:
    """Task 4: Generate course plan or eligibility decision."""

    if query_type == "eligibility":
        task_desc = f"""
Based on the student profile and extracted catalog rules, determine course ELIGIBILITY.

STUDENT PROFILE:
{student_profile}

STRUCTURED RULES (from catalog):
{structured_rules if structured_rules else "[From previous task]"}

CATALOG EVIDENCE:
{retrieved_evidence if retrieved_evidence else "[From previous task]"}

{PLANNER_AGENT_PROMPT}

For eligibility, respond with:
- DECISION: ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO
- Step-by-step prerequisite chain reasoning
- Each step must cite catalog evidence
- NEXT STEP: what the student should do

Use the EXACT output format:
Answer / Plan:
Why (requirements/prereqs satisfied):
Citations:
Clarifying questions (if needed):
Assumptions / Not in catalog:
"""
    else:
        task_desc = f"""
Based on the student profile and catalog rules, create an optimized course plan for next term.

STUDENT PROFILE:
{student_profile}

STRUCTURED RULES (from catalog):
{structured_rules if structured_rules else "[From previous task]"}

CATALOG EVIDENCE:
{retrieved_evidence if retrieved_evidence else "[From previous task]"}

{PLANNER_AGENT_PROMPT}

Create a term plan that:
1. Lists eligible courses (all prerequisites met)
2. Prioritizes required courses over electives
3. Respects max credit limits
4. Provides justification + citation for EVERY course

Use the EXACT output format:
Answer / Plan:
Why (requirements/prereqs satisfied):
Citations:
Clarifying questions (if needed):
Assumptions / Not in catalog:
"""

    return Task(
        description=task_desc,
        expected_output=(
            "A structured course plan (or eligibility decision) using the exact format: "
            "Answer/Plan, Why, Citations, Clarifying questions, Assumptions/Not in catalog. "
            "Every course recommendation must have justification AND citation. "
            "Risks and assumptions must be explicitly listed."
        ),
        agent=agent,
    )


def create_verification_task(agent, plan_output: str = "") -> Task:
    """Task 5: Audit the plan for citation completeness and logic."""
    return Task(
        description=f"""
Audit the following course plan or eligibility decision for:
1. Citation completeness — every factual claim has a citation
2. Logic correctness — prerequisite chains are followed correctly
3. Hallucination — no invented courses, policies, or requirements
4. Safe abstention — "not in catalog" items are flagged

PLAN TO AUDIT:
{plan_output if plan_output else "[From previous task output]"}

{VERIFIER_AGENT_PROMPT}

Be strict: if a claim lacks a citation, flag it.
If a claim cannot be verified from the catalog evidence in prior tasks, flag it as HALLUCINATION.
Output a clear PASS / NEEDS_REVISION / ESCALATE_TO_ADVISOR verdict.
""",
        expected_output=(
            "Verification report with: status (PASS/FAIL/NEEDS_REVISION), "
            "citation_coverage (X/Y), list of unsupported_claims, logic_errors, "
            "hallucination_flags, and recommended_action (APPROVE/REWRITE/ESCALATE_TO_ADVISOR)."
        ),
        agent=agent,
    )
