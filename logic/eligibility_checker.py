"""
eligibility_checker.py — High-level eligibility checking interface.
Combines retrieval + transitive reasoning for prerequisite decisions.
"""
from typing import List, Dict, Any, Optional
from rag.prompt_templates import ELIGIBILITY_CHECK_PROMPT


class EligibilityChecker:
    """
    Checks course eligibility using RAG + transitive reasoning.
    """

    def __init__(self, llm, retriever, transitive_engine):
        self.llm = llm
        self.retriever = retriever
        self.transitive_engine = transitive_engine

    def check(
        self,
        course_name: str,
        completed_courses: List[str],
        grades: Optional[Dict[str, str]] = None,
        additional_context: str = "",
    ) -> Dict[str, Any]:
        """
        Full eligibility check for a course.
        Returns structured result with decision, evidence, and citations.
        """
        grades = grades or {}

        # Use transitive engine for multi-hop reasoning
        chain_result = self.transitive_engine.check_prerequisite_chain(
            target_course=course_name,
            completed_courses=completed_courses,
            grades=grades,
        )

        return chain_result

    def check_multiple(
        self,
        courses: List[str],
        completed_courses: List[str],
        grades: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Check eligibility for multiple courses at once."""
        results = []
        for course in courses:
            result = self.check(course, completed_courses, grades)
            results.append(result)
        return results
