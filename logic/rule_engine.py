"""
rule_engine.py — Extract and apply academic rules from catalog text.
"""
import re
from typing import List, Dict, Any, Optional


class RuleEngine:
    """
    Extracts structured rules from catalog text and evaluates them
    against student data.
    """

    def extract_credit_requirements(self, text: str) -> Dict[str, Any]:
        """Extract credit hour requirements from text."""
        rules = {}

        # Total credits
        total_patterns = [
            r"(\d+)\s*credit\s*hours?\s*(?:are\s*)?required",
            r"minimum\s*(?:of\s*)?(\d+)\s*credits?",
            r"(\d+)\s*credits?\s*(?:must\s*be\s*)?completed",
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rules["total_credits_required"] = int(match.group(1))
                break

        # Per-term limits
        term_patterns = [
            r"(?:maximum|max\.?)\s*(?:of\s*)?(\d+)\s*credits?\s*per\s*(?:term|semester)",
            r"(\d+)\s*credits?\s*per\s*(?:term|semester)\s*(?:maximum|limit)",
        ]
        for pattern in term_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rules["max_credits_per_term"] = int(match.group(1))
                break

        return rules

    def extract_grade_requirements(self, text: str) -> List[Dict[str, str]]:
        """Extract minimum grade requirements."""
        requirements = []

        patterns = [
            r"(?:minimum|min\.?)\s*grade\s*(?:of\s*)?([A-F][+\-]?)",
            r"grade\s*(?:of\s*)?([A-F][+\-]?)\s*or\s*(?:better|higher|above)",
            r"([A-F][+\-]?)\s*or\s*better\s*in",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                requirements.append({
                    "min_grade": match.group(1).upper(),
                    "context": text[max(0, match.start()-50):match.end()+50].strip(),
                })

        return requirements

    def check_credit_eligibility(
        self,
        completed_credits: int,
        required_credits: int,
    ) -> bool:
        """Check if student has enough credits."""
        return completed_credits >= required_credits

    def check_grade_requirement(
        self,
        student_grade: str,
        min_grade: str,
    ) -> bool:
        """Check if student's grade meets minimum requirement."""
        grade_order = {
            "A+": 10, "A": 9, "A-": 8,
            "B+": 7, "B": 6, "B-": 5,
            "C+": 4, "C": 3, "C-": 2,
            "D": 1, "F": 0,
        }
        student_val = grade_order.get(student_grade.upper(), -1)
        min_val = grade_order.get(min_grade.upper(), -1)
        return student_val >= min_val
