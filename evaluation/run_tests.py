"""
run_tests.py — Entry point for running the evaluation suite.
Usage: python evaluation/run_tests.py [--max N] [--delay SECONDS]
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path="env/.env")

from configs.model_config import get_groq_llm, get_embedding_model
from rag.vector_store import CourseVectorStore
from rag.retriever import CatalogRetriever
from crew.crew_setup import CoursePlanningCrew
from evaluation.evaluator import CourseAssistantEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run Course Planning Assistant Evaluation")
    parser.add_argument("--max", type=int, default=None, help="Max number of queries to run")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between queries (seconds)")
    args = parser.parse_args()

    print("Loading models and vector store...")
    llm = get_groq_llm()
    embeddings = get_embedding_model()

    vs = CourseVectorStore()
    vs.load(embeddings)

    retriever = CatalogRetriever(vs, k=6)
    crew = CoursePlanningCrew(llm=llm, retriever=retriever)

    evaluator = CourseAssistantEvaluator(
        crew=crew,
        test_queries_path="data/evaluation/test_queries.json",
        output_path="data/evaluation/eval_results.json",
    )

    print(f"Running evaluation (max={args.max or 'all'}, delay={args.delay}s)...")
    summary = evaluator.run(max_queries=args.max, delay_seconds=args.delay)

    print("\nTop failures:")
    failures = [r for r in summary["individual_results"] if r["score"] < 0.5]
    for f in failures[:5]:
        print(f"  [{f['query_id']}] Score={f['score']:.1f} | "
              f"Expected={f['expected']} | Got={f['actual_decision']}")
        print(f"  Query: {f['query'][:80]}")


if __name__ == "__main__":
    main()
