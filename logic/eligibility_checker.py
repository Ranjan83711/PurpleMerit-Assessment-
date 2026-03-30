"""
Eligibility Checker Module
Evaluates student eligibility based on parsed rules and profile.
"""

from typing import Dict, List, Any


class EligibilityResult:
    def __init__(self, eligible: bool, reason: str, satisfied: List[str] = None, missing: List[str] = None):
        self.eligible = eligible
        self.reason = reason
        self.satisfied = satisfied or []
        self.missing = missing or []


def check_eligibility(student_profile: Dict[str, Any], course_rules: Dict[str, Any]) -> EligibilityResult:
    """
    Check if a student is eligible for a course based on their profile and course rules.
    """
    completed_courses = {c.upper() for c in student_profile.get("completed_courses", [])}
    
    # We will use the LLM planner agent to handle complex transitive logic, 
    # but this provides a structured fallback/utility for simpler explicit logic.
    
    reqs = course_rules.get("prerequisites", [])
    if not reqs:
        return EligibilityResult(True, "No prerequisites required.")

    satisfied = []
    missing = []
    
    # Very basic static check (the Planner Agent does the heavy lifting via LLM)
    for req in reqs:
        if req.upper() in completed_courses:
            satisfied.append(req)
        else:
            missing.append(req)
            
    if missing:
        return EligibilityResult(
            False, 
            f"Missing prerequisites: {', '.join(missing)}",
            satisfied=satisfied,
            missing=missing
        )
        
    return EligibilityResult(
        True,
        "All explicitly requested prerequisites met in basic check.",
        satisfied=satisfied
    )
