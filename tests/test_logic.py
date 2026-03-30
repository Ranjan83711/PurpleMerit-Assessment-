"""
Unit tests for logic components.
"""

from logic.transitive_reasoning import resolve_prerequisite_chain
from logic.eligibility_checker import check_eligibility

def test_transitive_reasoning():
    prereq_map = {
        'CS301': ['CS201'],
        'CS201': ['CS101', 'MATH120'],
        'CS101': [],
        'MATH120': []
    }
    required = resolve_prerequisite_chain('CS301', prereq_map)
    assert 'CS101' in required
    assert 'MATH120' in required
    assert 'CS201' in required
    
def test_eligibility_basic():
    profile = {"completed_courses": ["CS101", "MATH120"]}
    rules = {"prerequisites": ["CS101", "MATH120"]}
    res = check_eligibility(profile, rules)
    assert res.eligible == True
    
    rules_missing = {"prerequisites": ["CS101", "CS201"]}
    res2 = check_eligibility(profile, rules_missing)
    assert res2.eligible == False
    assert "CS201" in res2.missing
