"""
test_logic.py — Unit tests for logic/reasoning components.
Run: pytest tests/test_logic.py -v
"""
import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.rule_engine import RuleEngine
from logic.transitive_reasoning import TransitiveReasoningEngine


class TestRuleEngine:
    def test_extract_credit_requirements(self):
        engine = RuleEngine()
        text = "A minimum of 160 credit hours are required to complete the program."
        rules = engine.extract_credit_requirements(text)
        assert rules.get("total_credits_required") == 160

    def test_extract_grade_requirements(self):
        engine = RuleEngine()
        text = "Students must achieve a minimum grade of C or better in prerequisite courses."
        reqs = engine.extract_grade_requirements(text)
        assert len(reqs) > 0
        assert any(r["min_grade"].upper() == "C" for r in reqs)

    def test_grade_check_pass(self):
        engine = RuleEngine()
        assert engine.check_grade_requirement("B", "C") is True
        assert engine.check_grade_requirement("A", "B") is True
        assert engine.check_grade_requirement("C", "C") is True

    def test_grade_check_fail(self):
        engine = RuleEngine()
        assert engine.check_grade_requirement("D", "C") is False
        assert engine.check_grade_requirement("F", "D") is False
        assert engine.check_grade_requirement("C", "B") is False

    def test_credit_eligibility(self):
        engine = RuleEngine()
        assert engine.check_credit_eligibility(160, 160) is True
        assert engine.check_credit_eligibility(170, 160) is True
        assert engine.check_credit_eligibility(150, 160) is False


class TestTransitiveReasoning:
    def test_eligible_decision_parsing(self):
        """Test that ELIGIBLE is correctly extracted from LLM response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="DECISION: ELIGIBLE\nFINAL_VERDICT:\n  Decision: ELIGIBLE\n  Reason: All prerequisites met."
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve_as_context.return_value = ("Sample context", ["cite1"])

        engine = TransitiveReasoningEngine(mock_llm, mock_retriever)
        result = engine.check_prerequisite_chain(
            "Mathematics-II", ["Mathematics-I"], {}
        )
        assert result["decision"] == "ELIGIBLE"

    def test_not_eligible_decision_parsing(self):
        """Test that NOT ELIGIBLE is correctly extracted."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="DECISION: NOT ELIGIBLE\nFINAL_VERDICT:\n  Decision: NOT ELIGIBLE\n  Reason: Missing Mathematics-I."
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve_as_context.return_value = ("Sample context", ["cite1"])

        engine = TransitiveReasoningEngine(mock_llm, mock_retriever)
        result = engine.check_prerequisite_chain("Mathematics-II", [], {})
        assert result["decision"] == "NOT ELIGIBLE"

    def test_citations_included(self):
        """Test that citations from retrieval are passed through."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="DECISION: NEED MORE INFO")
        mock_retriever = MagicMock()
        mock_retriever.retrieve_as_context.return_value = (
            "Some context",
            ["kuk_catalog | Page 10 | chunk_id_1"],
        )

        engine = TransitiveReasoningEngine(mock_llm, mock_retriever)
        result = engine.check_prerequisite_chain("Advanced Math", ["Calculus"], {})
        assert len(result["citations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
