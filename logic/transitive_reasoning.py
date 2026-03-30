"""
Transitive Reasoning Module
Processes dependency chains (A -> B -> C)
"""

from typing import Dict, List, Set

def resolve_prerequisite_chain(course: str, prereq_map: Dict[str, List[str]]) -> Set[str]:
    """
    Given a target course and a mapping of course -> direct prerequisites,
    return the full set of transitive prerequisites.
    
    Args:
        course: Target course name (e.g. 'CS301')
        prereq_map: Direct prereqs map, e.g. {'CS301': ['CS201'], 'CS201': ['CS101']}
        
    Returns:
        Set of all courses needed directly or indirectly.
    """
    all_prereqs = set()
    visited = set()
    
    def dfs(current_course):
        if current_course in visited:
            return
        visited.add(current_course)
        
        direct_reqs = prereq_map.get(current_course, [])
        for req in direct_reqs:
            all_prereqs.add(req)
            dfs(req)
            
    dfs(course)
    return all_prereqs
