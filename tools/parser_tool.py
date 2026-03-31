"""
parser_tool.py — CrewAI tool for parsing structured prerequisite rules
from natural language catalog text using the LLM.
"""
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class ParsePrereqInput(BaseModel):
    text: str = Field(description="Raw catalog text containing prerequisite information to parse")
    course_name: str = Field(default="", description="Name of the course being analyzed")


class PrerequisiteParserTool(BaseTool):
    """
    Uses LLM to parse complex prerequisite language into structured rules.
    Handles 'either/or', co-requisites, grade requirements, exceptions.
    """

    name: str = "prerequisite_parser"
    description: str = (
        "Parse complex prerequisite language from catalog text into structured rules. "
        "Use this when catalog text contains 'either/or', grade requirements, co-requisites, "
        "or exception clauses that need structured interpretation. "
        "Input raw catalog text; output structured prerequisite rules."
    )
    args_schema: Type[BaseModel] = ParsePrereqInput

    llm: object = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, text: str, course_name: str = "") -> str:
        """Parse prerequisite text into structured format."""
        if self.llm is None:
            return self._rule_based_parse(text)

        prompt = f"""Parse the following catalog text and extract prerequisite rules in structured format.

Course: {course_name or "Unknown"}
Catalog Text: {text}

Extract and output:
PREREQUISITES:
  - type: [required/optional/either_or]
    courses: [list of course names/codes]
    logic: [AND/OR]
    min_grade: [grade or "not specified"]
    
CO_REQUISITES:
  - [concurrent courses if any, or "none"]

GRADE_REQUIREMENTS:
  - [minimum grades if specified, or "none"]

EXCEPTIONS:
  - [exception clauses like "with instructor consent", or "none"]

CREDIT_REQUIREMENTS:
  - [credit hour requirements if any, or "none"]

If the text does not specify a field, write "not specified" for that field.
Do NOT invent rules not present in the text."""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return self._rule_based_parse(text)

    def _rule_based_parse(self, text: str) -> str:
        """Fallback rule-based parser for common prerequisite patterns."""
        import re
        lines = []

        # Look for prerequisite keywords
        prereq_patterns = [
            r"prerequisite[s]?[:\s]+(.+?)(?:\.|$)",
            r"pre-requisite[s]?[:\s]+(.+?)(?:\.|$)",
            r"requires?[:\s]+(.+?)(?:\.|$)",
            r"must have completed[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in prereq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                lines.append(f"PREREQUISITES found: {matches}")

        # Co-requisites
        coreq_patterns = [r"co-?requisite[s]?[:\s]+(.+?)(?:\.|$)"]
        for pattern in coreq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                lines.append(f"CO_REQUISITES found: {matches}")

        # Grade requirements
        grade_patterns = [r"minimum grade[:\s]+([A-F][+\-]?)", r"grade of ([A-F][+\-]?) or better"]
        for pattern in grade_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                lines.append(f"GRADE_REQUIREMENTS: {matches}")

        # Exceptions
        exception_patterns = [r"(instructor['\s]?s? consent)", r"(departmental approval)", r"(permission of)"]
        for pattern in exception_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                lines.append(f"EXCEPTIONS: {matches}")

        return "\n".join(lines) if lines else "No structured prerequisites detected in text."
