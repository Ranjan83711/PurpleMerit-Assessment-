"""
evaluator.py — Runs the 25-query test suite and computes evaluation metrics.

Metrics computed:
  1. Citation coverage rate   — % of responses that include at least one citation
  2. Eligibility correctness  — % of prereq decisions matching expected
  3. Abstention accuracy      — % of "not in docs" queries correctly refused
  4. Chain reasoning quality  — manual rubric score for multi-hop queries
"""
import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class CourseAssistantEvaluator:
    """
    Evaluates the course planning assistant against the 25-query test set.
    """

    # Rubric for eligibility correctness
    ELIGIBILITY_RUBRIC = {
        "ELIGIBLE_match": 1.0,
        "NOT_ELIGIBLE_match": 1.0,
        "NEED_MORE_INFO_match": 1.0,
        "ELIGIBLE_vs_NOT_ELIGIBLE": 0.0,   # Wrong direction = 0
        "ELIGIBLE_vs_NEED_MORE_INFO": 0.5,  # Partial credit
        "NOT_ELIGIBLE_vs_NEED_MORE_INFO": 0.5,
        "ABSTAIN_match": 1.0,
        "ABSTAIN_vs_answer": 0.0,           # Should have abstained = 0
    }

    def __init__(self, crew, test_queries_path: str, output_path: str):
        self.crew = crew
        self.test_queries_path = test_queries_path
        self.output_path = output_path
        self.results = []

    def load_test_queries(self) -> List[Dict]:
        with open(self.test_queries_path, "r") as f:
            return json.load(f)

    def run(self, max_queries: Optional[int] = None, delay_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Run all test queries through the crew pipeline.
        delay_seconds: pause between queries to avoid Groq rate limits.
        """
        queries = self.load_test_queries()
        if max_queries:
            queries = queries[:max_queries]

        print(f"\n{'='*60}")
        print(f"EVALUATION RUN — {len(queries)} queries")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        for i, query_obj in enumerate(queries, start=1):
            print(f"\n[{i}/{len(queries)}] Running: {query_obj['id']} ({query_obj['type']})")
            print(f"Query: {query_obj['query'][:80]}...")

            start_time = time.time()
            try:
                # Route to appropriate crew method
                if query_obj["type"] in ("prerequisite_check", "prerequisite_chain"):
                    result = self.crew.run_eligibility_check(query_obj["query"])
                elif query_obj["type"] == "not_in_docs":
                    result = self.crew.run_general_query(query_obj["query"])
                else:
                    result = self.crew.run_general_query(query_obj["query"])

                elapsed = time.time() - start_time
                eval_result = self._evaluate_single(query_obj, result, elapsed)
                self.results.append(eval_result)
                print(f"  Decision: {result.get('decision', 'UNKNOWN')} | "
                      f"Expected: {query_obj['expected_decision']} | "
                      f"Score: {eval_result['score']:.1f} | "
                      f"Time: {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  ERROR: {str(e)[:100]}")
                self.results.append({
                    "query_id": query_obj["id"],
                    "query_type": query_obj["type"],
                    "query": query_obj["query"],
                    "expected": query_obj["expected_decision"],
                    "actual_decision": "ERROR",
                    "score": 0.0,
                    "has_citation": False,
                    "abstained_correctly": False,
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                })

            # Rate limit protection for Groq
            if i < len(queries):
                time.sleep(delay_seconds)

        # Compute summary metrics
        summary = self._compute_metrics()
        self._save_results(summary)
        self._print_report(summary)
        return summary

    def _evaluate_single(
        self, query_obj: Dict, result: Dict, elapsed: float
    ) -> Dict[str, Any]:
        """Evaluate a single query result against expected output."""
        raw = result.get("raw_output", "")
        decision = result.get("decision", "UNKNOWN")
        expected = query_obj["expected_decision"]

        # Check citation presence
        has_citation = (
            "Page" in raw
            or "Chunk" in raw
            or "Source" in raw
            or "citation" in raw.lower()
            or "| p" in raw.lower()
        )

        # Check abstention correctness
        abstained = (
            "not have that information" in raw.lower()
            or "not in catalog" in raw.lower()
            or "not found" in raw.lower()
            or "cannot determine" in raw.lower()
            or "abstain" in raw.lower()
            or "schedule of classes" in raw.lower()
            or "advisor" in raw.lower()
        )
        abstained_correctly = (
            abstained if expected == "ABSTAIN" else (not abstained if expected != "ABSTAIN" else True)
        )

        # Compute score
        score = self._compute_score(expected, decision, has_citation, abstained)

        return {
            "query_id": query_obj["id"],
            "query_type": query_obj["type"],
            "query": query_obj["query"],
            "expected": expected,
            "actual_decision": decision,
            "has_citation": has_citation,
            "abstained_correctly": abstained_correctly if expected == "ABSTAIN" else None,
            "score": score,
            "elapsed_seconds": round(elapsed, 2),
            "raw_output_preview": raw[:500],
        }

    def _compute_score(
        self, expected: str, actual: str, has_citation: bool, abstained: bool
    ) -> float:
        """Score a single response (0.0 to 1.0)."""
        # Abstention queries
        if expected == "ABSTAIN":
            return 1.0 if abstained else 0.0

        # Factual/chain queries — check citation presence as primary signal
        if expected in ("FACTUAL_ANSWER", "CHAIN_ANALYSIS"):
            return 1.0 if has_citation else 0.5

        # Eligibility queries — check decision accuracy
        if expected == actual:
            base_score = 1.0
        elif expected == "ELIGIBLE" and actual == "NOT ELIGIBLE":
            base_score = 0.0
        elif expected == "NOT ELIGIBLE" and actual == "ELIGIBLE":
            base_score = 0.0
        elif "NEED_MORE_INFO" in expected or "NEED MORE INFO" in actual:
            base_score = 0.5
        else:
            base_score = 0.3

        # Citation bonus/penalty
        citation_modifier = 0.0 if has_citation else -0.1
        return max(0.0, min(1.0, base_score + citation_modifier))

    def _compute_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all results."""
        total = len(self.results)
        if total == 0:
            return {"error": "No results"}

        # Citation coverage
        cited = sum(1 for r in self.results if r.get("has_citation"))
        citation_rate = cited / total

        # By type
        by_type = {}
        for q_type in ["prerequisite_check", "prerequisite_chain", "program_requirement", "not_in_docs"]:
            type_results = [r for r in self.results if r["query_type"] == q_type]
            if type_results:
                avg_score = sum(r["score"] for r in type_results) / len(type_results)
                by_type[q_type] = {
                    "count": len(type_results),
                    "avg_score": round(avg_score, 3),
                    "citation_rate": round(
                        sum(1 for r in type_results if r.get("has_citation")) / len(type_results), 3
                    ),
                }

        # Abstention accuracy
        not_in_docs = [r for r in self.results if r["query_type"] == "not_in_docs"]
        abstention_accuracy = (
            sum(1 for r in not_in_docs if r.get("abstained_correctly")) / len(not_in_docs)
            if not_in_docs else 0.0
        )

        # Eligibility correctness
        eligibility = [r for r in self.results if r["query_type"] == "prerequisite_check"]
        eligibility_correctness = (
            sum(r["score"] for r in eligibility) / len(eligibility) if eligibility else 0.0
        )

        # Overall
        overall_score = sum(r["score"] for r in self.results) / total
        avg_latency = sum(r.get("elapsed_seconds", 0) for r in self.results) / total

        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": total,
            "overall_score": round(overall_score, 3),
            "citation_coverage_rate": round(citation_rate, 3),
            "eligibility_correctness": round(eligibility_correctness, 3),
            "abstention_accuracy": round(abstention_accuracy, 3),
            "avg_latency_seconds": round(avg_latency, 2),
            "by_type": by_type,
            "individual_results": self.results,
        }

    def _save_results(self, summary: Dict):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[Evaluator] Results saved → {self.output_path}")

    def _print_report(self, summary: Dict):
        """Print a formatted evaluation report."""
        print(f"\n{'='*60}")
        print("EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"Total Queries:          {summary['total_queries']}")
        print(f"Overall Score:          {summary['overall_score']:.1%}")
        print(f"Citation Coverage:      {summary['citation_coverage_rate']:.1%}")
        print(f"Eligibility Correctness:{summary['eligibility_correctness']:.1%}")
        print(f"Abstention Accuracy:    {summary['abstention_accuracy']:.1%}")
        print(f"Avg Latency:            {summary['avg_latency_seconds']:.1f}s")
        print(f"\nBy Query Type:")
        for q_type, stats in summary.get("by_type", {}).items():
            print(f"  {q_type:<25} score={stats['avg_score']:.1%}  "
                  f"cited={stats['citation_rate']:.1%}  n={stats['count']}")
        print(f"{'='*60}\n")
