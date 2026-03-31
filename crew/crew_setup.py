"""
crew_setup.py — CrewAI orchestration for the Course Planning Assistant.

Pipeline:
  [Intake Agent] → [Retriever Agent] → [Rule Extractor Agent]
       → [Planner/Explanation Agent] → [Verifier Agent]

Two modes:
  - ELIGIBILITY: Check if student can take a specific course
  - PLAN: Generate a full term course plan
"""
import os
from typing import Dict, Any, Optional
from crewai import Crew, Process

from agents.intake_agent import create_intake_agent
from agents.retriever_agent import create_retriever_agent
from agents.rule_extractor_agent import create_rule_extractor_agent
from agents.explanation_agent import create_explanation_agent
from agents.verifier_agent import create_verifier_agent

from crew.tasks import (
    create_intake_task,
    create_retrieval_task,
    create_rule_extraction_task,
    create_planning_task,
    create_verification_task,
)

from tools.vector_search_tool import VectorSearchTool
from tools.pdf_tool import PDFPageTool
from tools.parser_tool import PrerequisiteParserTool


class CoursePlanningCrew:
    """
    Orchestrates the 5-agent CrewAI pipeline for course planning and eligibility checking.
    """

    def __init__(self, llm, retriever, pdf_path: str = "data/raw/kuk_prospectus_2011.pdf"):
        self.llm = llm
        self.retriever = retriever
        self.pdf_path = pdf_path

        # Initialize tools
        self.vector_search_tool = VectorSearchTool(retriever=retriever)
        self.pdf_tool = PDFPageTool(pdf_path=pdf_path)
        self.parser_tool = PrerequisiteParserTool(llm=llm)

        self.retriever_tools = [self.vector_search_tool, self.pdf_tool]
        self.extractor_tools = [self.vector_search_tool, self.parser_tool]

        # Initialize agents
        self.intake_agent = create_intake_agent(llm)
        self.retriever_agent = create_retriever_agent(llm, self.retriever_tools)
        self.rule_extractor_agent = create_rule_extractor_agent(llm, self.extractor_tools)
        self.planner_agent = create_explanation_agent(llm)
        self.verifier_agent = create_verifier_agent(llm)

    def run_eligibility_check(
        self,
        student_query: str,
        student_profile: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run the full eligibility check pipeline.
        Returns structured result with decision, reasoning, and citations.
        """
        profile_str = self._format_profile(student_profile) if student_profile else ""

        # Define tasks
        intake_task = create_intake_task(self.intake_agent, student_query)
        retrieval_task = create_retrieval_task(
            self.retriever_agent, student_query, profile_str
        )
        rule_task = create_rule_extraction_task(self.rule_extractor_agent)
        planning_task = create_planning_task(
            self.planner_agent,
            student_profile=profile_str or student_query,
            query_type="eligibility",
        )
        verification_task = create_verification_task(self.verifier_agent)

        # Build and run crew
        crew = Crew(
            agents=[
                self.intake_agent,
                self.retriever_agent,
                self.rule_extractor_agent,
                self.planner_agent,
                self.verifier_agent,
            ],
            tasks=[
                intake_task,
                retrieval_task,
                rule_task,
                planning_task,
                verification_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return self._parse_crew_result(result, "eligibility")

    def run_course_plan(
        self,
        student_query: str,
        student_profile: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run the full course planning pipeline.
        Returns a structured semester plan with justifications and citations.
        """
        profile_str = self._format_profile(student_profile) if student_profile else student_query

        intake_task = create_intake_task(self.intake_agent, student_query)
        retrieval_task = create_retrieval_task(
            self.retriever_agent, student_query, profile_str
        )
        rule_task = create_rule_extraction_task(self.rule_extractor_agent)
        planning_task = create_planning_task(
            self.planner_agent,
            student_profile=profile_str,
            query_type="plan",
        )
        verification_task = create_verification_task(self.verifier_agent)

        crew = Crew(
            agents=[
                self.intake_agent,
                self.retriever_agent,
                self.rule_extractor_agent,
                self.planner_agent,
                self.verifier_agent,
            ],
            tasks=[
                intake_task,
                retrieval_task,
                rule_task,
                planning_task,
                verification_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return self._parse_crew_result(result, "plan")

    def run_general_query(self, query: str) -> Dict[str, Any]:
        """
        Run a general catalog query (prerequisite question, policy lookup, etc.)
        Uses a lighter 3-agent pipeline: Retriever → Planner → Verifier.
        """
        retrieval_task = create_retrieval_task(self.retriever_agent, query)
        rule_task = create_rule_extraction_task(self.rule_extractor_agent)
        planning_task = create_planning_task(
            self.planner_agent,
            student_profile=query,
            query_type="eligibility",
        )
        verification_task = create_verification_task(self.verifier_agent)

        crew = Crew(
            agents=[
                self.retriever_agent,
                self.rule_extractor_agent,
                self.planner_agent,
                self.verifier_agent,
            ],
            tasks=[retrieval_task, rule_task, planning_task, verification_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return self._parse_crew_result(result, "general")

    def _format_profile(self, profile: Dict) -> str:
        """Format a student profile dict into a readable string."""
        if not profile:
            return ""
        lines = ["STUDENT PROFILE:"]
        for key, value in profile.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _parse_crew_result(self, result, query_type: str) -> Dict[str, Any]:
        """Parse CrewAI result into a structured response dict."""
        # CrewAI returns the last task's output
        raw_output = str(result)

        # Extract key sections
        sections = {
            "answer": self._extract_section(raw_output, "Answer / Plan:"),
            "why": self._extract_section(raw_output, "Why (requirements/prereqs satisfied):"),
            "citations": self._extract_section(raw_output, "Citations:"),
            "clarifying_questions": self._extract_section(raw_output, "Clarifying questions"),
            "assumptions": self._extract_section(raw_output, "Assumptions / Not in catalog:"),
            "verification": self._extract_section(raw_output, "VERIFICATION_RESULT:"),
        }

        # Determine decision
        decision = "NEED MORE INFO"
        raw_upper = raw_output.upper()
        if "ELIGIBLE" in raw_upper and "NOT ELIGIBLE" not in raw_upper:
            decision = "ELIGIBLE"
        elif "NOT ELIGIBLE" in raw_upper:
            decision = "NOT ELIGIBLE"
        elif "PASS" in raw_upper and "VERIFICATION" in raw_upper:
            decision = "APPROVED"

        return {
            "query_type": query_type,
            "decision": decision,
            "raw_output": raw_output,
            "sections": sections,
            "formatted": self._format_final_output(sections),
        }

    def _extract_section(self, text: str, section_header: str) -> str:
        """Extract a section from structured text output."""
        lines = text.split("\n")
        result_lines = []
        in_section = False

        for line in lines:
            if section_header.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                # Stop at next major section header
                if any(
                    header in line
                    for header in [
                        "Answer / Plan:",
                        "Why (",
                        "Citations:",
                        "Clarifying questions",
                        "Assumptions /",
                        "VERIFICATION_RESULT:",
                        "DECISION:",
                    ]
                ) and line.strip() != "":
                    break
                result_lines.append(line)

        return "\n".join(result_lines).strip()

    def _format_final_output(self, sections: Dict[str, str]) -> str:
        """Format sections into the required output structure."""
        parts = []

        if sections.get("answer"):
            parts.append(f"**Answer / Plan:**\n{sections['answer']}")
        if sections.get("why"):
            parts.append(f"\n**Why (requirements/prereqs satisfied):**\n{sections['why']}")
        if sections.get("citations"):
            parts.append(f"\n**Citations:**\n{sections['citations']}")
        if sections.get("clarifying_questions"):
            parts.append(f"\n**Clarifying questions (if needed):**\n{sections['clarifying_questions']}")
        if sections.get("assumptions"):
            parts.append(f"\n**Assumptions / Not in catalog:**\n{sections['assumptions']}")
        if sections.get("verification"):
            parts.append(f"\n**Verification:**\n{sections['verification']}")

        return "\n".join(parts) if parts else "No structured output available."
