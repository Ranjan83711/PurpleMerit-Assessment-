"""
Parser Tool for CrewAI Agents
Parses and structures student input and query information.
"""

import re
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class ParseStudentInput(BaseModel):
    """Input schema for parser tool."""
    raw_input: str = Field(..., description="Raw student query or input text to parse")


class StudentProfileParserTool(BaseTool):
    """Tool to parse and structure student input into a profile."""
    
    name: str = "parse_student_profile"
    description: str = (
        "Parse a student's query or message to extract structured information: "
        "completed courses, target courses, program/major, grades, term, etc. "
        "Returns a structured profile that other agents can use."
    )
    args_schema: Type[BaseModel] = ParseStudentInput
    
    def _run(self, raw_input: str) -> str:
        """Parse student input into structured profile."""
        profile = {
            "query_type": self._detect_query_type(raw_input),
            "completed_courses": self._extract_courses(raw_input, "completed"),
            "target_courses": self._extract_courses(raw_input, "target"),
            "program": self._extract_program(raw_input),
            "grades": self._extract_grades(raw_input),
            "term": self._extract_term(raw_input),
            "max_courses": self._extract_max_courses(raw_input),
            "missing_info": [],
        }
        
        # Identify missing critical info
        if not profile["completed_courses"]:
            profile["missing_info"].append("completed courses")
        if not profile["program"]:
            profile["missing_info"].append("target program/major/degree")
        
        if profile["query_type"] == "course_planning":
            if not profile["term"]:
                profile["missing_info"].append("target term (e.g., Fall/Spring)")
            if not profile["max_courses"]:
                profile["missing_info"].append("maximum courses or credits for the term")
        
        # Format as string
        result = "PARSED STUDENT PROFILE:\n"
        result += f"Query Type: {profile['query_type']}\n"
        result += f"Completed Courses: {profile['completed_courses'] or 'Not specified'}\n"
        result += f"Target Courses: {profile['target_courses'] or 'Not specified'}\n"
        result += f"Program/Major: {profile['program'] or 'Not specified'}\n"
        result += f"Grades: {profile['grades'] or 'Not specified'}\n"
        result += f"Term: {profile['term'] or 'Not specified'}\n"
        result += f"Max Courses: {profile['max_courses'] or 'Not specified'}\n"
        
        if profile["missing_info"]:
            result += f"\nMISSING INFORMATION: {', '.join(profile['missing_info'])}\n"
            result += "Consider asking the student for this information."
        
        return result
    
    def _detect_query_type(self, text: str) -> str:
        """Detect the type of query."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["prerequisite", "prereq", "can i take", "eligible", "before enrolling", "need before"]):
            return "prerequisite_check"
        elif any(kw in text_lower for kw in ["plan", "schedule", "next term", "next semester", "suggest courses", "what should i take"]):
            return "course_planning"
        elif any(kw in text_lower for kw in ["policy", "grading", "gpa", "repeat", "credit limit", "academic"]):
            return "policy_question"
        elif any(kw in text_lower for kw in ["requirement", "degree", "major", "minor", "credits needed"]):
            return "program_requirement"
        else:
            return "general_question"
    
    def _extract_courses(self, text: str, context: str) -> list:
        """Extract course codes/names from text."""
        # Match patterns like CS101, MATH 120, BCA-101, MCA301, etc.
        course_pattern = r'\b[A-Z]{2,5}[-\s]?\d{2,4}[A-Z]?\b'
        courses = re.findall(course_pattern, text.upper())
        return list(set(courses)) if courses else []
    
    def _extract_program(self, text: str) -> str:
        """Extract program/degree/major from text."""
        text_lower = text.lower()
        programs = [
            "b.tech", "btech", "b.sc", "bsc", "b.a", "ba", "b.com", "bcom",
            "m.tech", "mtech", "m.sc", "msc", "m.a", "ma", "m.com", "mcom",
            "bca", "mca", "bba", "mba", "phd", "ph.d",
            "computer science", "information technology", "electronics",
            "mathematics", "physics", "chemistry", "commerce",
        ]
        for prog in programs:
            if prog in text_lower:
                return prog.upper()
        return ""
    
    def _extract_grades(self, text: str) -> dict:
        """Extract grade information."""
        # Look for patterns like "CS101: A", "got B+ in MATH120"
        grade_pattern = r'([A-Z]{2,5}[-\s]?\d{2,4})\s*[:=\-]\s*([A-F][+-]?)'
        matches = re.findall(grade_pattern, text.upper())
        return dict(matches) if matches else {}
    
    def _extract_term(self, text: str) -> str:
        """Extract target term."""
        text_lower = text.lower()
        if "fall" in text_lower:
            return "Fall"
        elif "spring" in text_lower:
            return "Spring"
        elif "summer" in text_lower:
            return "Summer"
        elif "winter" in text_lower:
            return "Winter"
        return ""
    
    def _extract_max_courses(self, text: str) -> str:
        """Extract maximum courses or credits."""
        # Look for patterns like "5 courses", "18 credits", "max 6"
        match = re.search(r'(\d+)\s*(courses?|subjects?|credits?|papers?)', text.lower())
        if match:
            return f"{match.group(1)} {match.group(2)}"
        match = re.search(r'max(?:imum)?\s*(\d+)', text.lower())
        if match:
            return match.group(1)
        return ""
