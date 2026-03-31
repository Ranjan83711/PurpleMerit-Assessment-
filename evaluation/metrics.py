"""
metrics.py — Standalone metric computation utilities.
Used by evaluator.py and also importable for custom analysis.
"""
from typing import List, Dict, Any


def citation_coverage_rate(results: List[Dict]) -> float:
    """% of responses that include at least one citation."""
    if not results:
        return 0.0
    cited = sum(1 for r in results if r.get("has_citation", False))
    return round(cited / len(results), 4)


def eligibility_correctness(results: List[Dict]) -> float:
    """
    Average score on prerequisite_check queries.
    Rubric:
      - Correct decision (ELIGIBLE/NOT ELIGIBLE match) = 1.0
      - NEED_MORE_INFO when expected ELIGIBLE/NOT ELIGIBLE = 0.5
      - Wrong direction (ELIGIBLE vs NOT ELIGIBLE) = 0.0
    """
    prereq_results = [r for r in results if r.get("query_type") == "prerequisite_check"]
    if not prereq_results:
        return 0.0
    return round(sum(r.get("score", 0) for r in prereq_results) / len(prereq_results), 4)


def abstention_accuracy(results: List[Dict]) -> float:
    """% of 'not_in_docs' queries correctly abstained from answering."""
    not_in_docs = [r for r in results if r.get("query_type") == "not_in_docs"]
    if not not_in_docs:
        return 0.0
    correct = sum(1 for r in not_in_docs if r.get("abstained_correctly", False))
    return round(correct / len(not_in_docs), 4)


def chain_reasoning_quality(results: List[Dict]) -> float:
    """Average score on prerequisite_chain queries (multi-hop)."""
    chain_results = [r for r in results if r.get("query_type") == "prerequisite_chain"]
    if not chain_results:
        return 0.0
    return round(sum(r.get("score", 0) for r in chain_results) / len(chain_results), 4)


def average_latency(results: List[Dict]) -> float:
    """Average response latency in seconds."""
    times = [r.get("elapsed_seconds", 0) for r in results if "elapsed_seconds" in r]
    return round(sum(times) / len(times), 2) if times else 0.0


def full_report(results: List[Dict]) -> Dict[str, Any]:
    """Compute and return all metrics as a single dict."""
    return {
        "total_queries": len(results),
        "citation_coverage_rate": citation_coverage_rate(results),
        "eligibility_correctness": eligibility_correctness(results),
        "abstention_accuracy": abstention_accuracy(results),
        "chain_reasoning_quality": chain_reasoning_quality(results),
        "average_latency_seconds": average_latency(results),
        "errors": sum(1 for r in results if r.get("actual_decision") == "ERROR"),
    }


def print_report(metrics: Dict[str, Any]):
    """Pretty-print a metrics report."""
    print("\n" + "=" * 55)
    print("📊  EVALUATION METRICS REPORT")
    print("=" * 55)
    print(f"  Total queries evaluated : {metrics['total_queries']}")
    print(f"  Errors                  : {metrics['errors']}")
    print("-" * 55)
    print(f"  Citation coverage rate  : {metrics['citation_coverage_rate']:.1%}")
    print(f"  Eligibility correctness : {metrics['eligibility_correctness']:.1%}")
    print(f"  Abstention accuracy     : {metrics['abstention_accuracy']:.1%}")
    print(f"  Chain reasoning quality : {metrics['chain_reasoning_quality']:.1%}")
    print(f"  Avg latency (sec)       : {metrics['average_latency_seconds']:.2f}s")
    print("=" * 55 + "\n")
