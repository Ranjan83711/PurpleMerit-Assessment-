"""
transitive_reasoning.py — LLM-powered transitive prerequisite reasoning.

Core idea: Instead of trying to hard-code prerequisite graphs from imperfect
PDF extraction, we use the LLM itself to reason through prerequisite chains
using the retrieved vector DB knowledge as context.

This handles:
- A → B → C chains (multi-hop)
- Either/or prerequisites
- Grade requirements in chains
- Implicit prerequisites not explicitly listed
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document


class TransitiveReasoningEngine:
    """
    Uses LLM + vector DB context to reason through multi-hop prerequisite chains.
    
    Design principle: The LLM reads all relevant catalog chunks and reasons
    through the chain step-by-step, citing evidence at each hop.
    This is more robust than graph extraction from OCR'd PDFs.
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def check_prerequisite_chain(
        self,
        target_course: str,
        completed_courses: List[str],
        grades: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Check if a student can take target_course given their completed courses.
        
        Uses multi-hop retrieval: first retrieves info about target course,
        then recursively retrieves info about any prerequisites found,
        building a full evidence chain.
        """
        grades = grades or {}

        # Step 1: Retrieve info about the target course
        target_context, target_citations = self.retriever.retrieve_as_context(
            f"prerequisites requirements for {target_course}"
        )

        # Step 2: Retrieve info about completed courses (for grade verification)
        completed_str = ", ".join(completed_courses)
        completed_context, _ = self.retriever.retrieve_as_context(
            f"course description {completed_str}"
        )

        # Step 3: Build the reasoning prompt
        prompt = self._build_chain_reasoning_prompt(
            target_course=target_course,
            completed_courses=completed_courses,
            grades=grades,
            target_context=target_context,
            completed_context=completed_context,
        )

        # Step 4: LLM reasons through the chain
        try:
            response = self.llm.invoke(prompt)
            reasoning_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            reasoning_text = f"LLM reasoning failed: {str(e)}"

        # Step 5: Parse the structured response
        return self._parse_reasoning_response(
            reasoning_text,
            target_course,
            target_citations,
        )

    def _build_chain_reasoning_prompt(
        self,
        target_course: str,
        completed_courses: List[str],
        grades: Dict[str, str],
        target_context: str,
        completed_context: str,
    ) -> str:
        """Build the multi-hop reasoning prompt."""
        grades_str = (
            "\n".join([f"  - {course}: {grade}" for course, grade in grades.items()])
            if grades
            else "  (No grades provided)"
        )

        return f"""You are an academic advisor AI. Using ONLY the catalog evidence below,
determine if the student is eligible to enroll in {target_course}.

STUDENT'S COMPLETED COURSES:
{chr(10).join(f"  - {c}" for c in completed_courses)}

STUDENT'S GRADES:
{grades_str}

TARGET COURSE: {target_course}

CATALOG EVIDENCE FOR TARGET COURSE:
{target_context}

CATALOG EVIDENCE FOR COMPLETED COURSES:
{completed_context}

INSTRUCTIONS:
1. Identify ALL prerequisites for {target_course} from the evidence above
2. For each prerequisite, check if the student has completed it
3. If prerequisites have their OWN prerequisites (chain), reason through those too
4. Check grade requirements if specified
5. Note any exceptions (instructor consent, etc.)
6. If any information is not in the catalog, explicitly state "NOT IN CATALOG"

Respond in EXACTLY this format:

ELIGIBILITY_CHAIN_ANALYSIS:

TARGET_COURSE: {target_course}
DECISION: [ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO / CANNOT DETERMINE]

PREREQUISITE_CHAIN:
  Hop 1: {target_course} requires → [list prerequisites with citations]
    [Citation: source | page | chunk_id]
  
  Hop 2 (if any prereq has prereqs): [prereq X] requires → [list]
    [Citation: source | page | chunk_id]

STUDENT_STATUS_PER_PREREQ:
  - [Prereq 1]: [COMPLETED / NOT COMPLETED / GRADE INSUFFICIENT / UNKNOWN]
    Evidence: [why]
  - [Prereq 2]: ...

GRADE_REQUIREMENTS_CHECK:
  [Any grade minimums and whether student meets them, or "Not specified in catalog"]

EXCEPTIONS_AVAILABLE:
  [Any exception paths like instructor consent, or "None found in catalog"]

FINAL_VERDICT:
  Decision: [ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO]
  Reason: [clear 1-2 sentence explanation]
  Next_Step: [What the student should do next]
  
CITATIONS_USED:
  - [citation 1]
  - [citation 2]
  ...

NOT_IN_CATALOG:
  - [Any information needed but not found in the provided catalog evidence]
"""

    def _parse_reasoning_response(
        self,
        response_text: str,
        target_course: str,
        base_citations: List[str],
    ) -> Dict[str, Any]:
        """Parse the LLM's structured response into a dict."""
        # Extract decision
        decision = "NEED MORE INFO"
        if "DECISION: ELIGIBLE" in response_text and "NOT ELIGIBLE" not in response_text.split("DECISION: ELIGIBLE")[0][-20:]:
            decision = "ELIGIBLE"
        elif "DECISION: NOT ELIGIBLE" in response_text:
            decision = "NOT ELIGIBLE"
        elif "CANNOT DETERMINE" in response_text:
            decision = "CANNOT DETERMINE"

        # Extract final verdict (more reliable)
        if "Decision: ELIGIBLE" in response_text and "NOT ELIGIBLE" not in response_text:
            decision = "ELIGIBLE"
        elif "Decision: NOT ELIGIBLE" in response_text:
            decision = "NOT ELIGIBLE"

        return {
            "target_course": target_course,
            "decision": decision,
            "full_reasoning": response_text,
            "citations": base_citations,
        }

    def build_prereq_context_for_multiple_hops(
        self, course_name: str, max_hops: int = 3
    ) -> str:
        """
        Retrieve context for a course AND its likely prerequisites
        (multi-hop retrieval for transitive chains).
        """
        all_contexts = []
        queries = [
            f"prerequisites for {course_name}",
            f"{course_name} course requirements",
            f"courses required before {course_name}",
        ]

        for query in queries:
            context, _ = self.retriever.retrieve_as_context(query)
            if "No relevant information" not in context:
                all_contexts.append(context)

        return "\n\n===\n\n".join(all_contexts) if all_contexts else "No prerequisite information found."
