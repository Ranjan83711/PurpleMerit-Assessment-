"""
Evaluator Module
Runs the evaluation set (25 queries) against the RAG system.
"""

import json
from pathlib import Path
from crew.crew_setup import CoursePlannerCrew

def load_eval_data(eval_dir: Path):
    queries_file = eval_dir / "test_queries.json"
    if not queries_file.exists():
        # Create minimal template
        queries = [
            {"id": 1, "query": "Can I take CS301 if I have completed CS101 and MATH120?", "type": "prereq"},
            {"id": 2, "query": "What do I need before taking Database Systems?", "type": "prereq"},
            {"id": 3, "query": "If I want to take Machine Learning, what is the full list of prior courses I need starting from my first year?", "type": "transitive"},
            {"id": 4, "query": "What are the core requirements for a BCA degree?", "type": "program"},
            {"id": 5, "query": "Are courses available in the evening?", "type": "not_in_docs"}
        ]
        queries_file.parent.mkdir(parents=True, exist_ok=True)
        with open(queries_file, 'w') as f:
            json.dump(queries, f, indent=4)
        return queries
    else:
        with open(queries_file, 'r') as f:
            return json.load(f)

def run_evaluation(api_key: str):
    eval_dir = Path(__file__).parent.parent / "data" / "evaluation"
    queries = load_eval_data(eval_dir)
    
    crew_app = CoursePlannerCrew(groq_api_key=api_key)
    
    results = []
    
    print(f"Running evaluation on {len(queries)} test queries...")
    for q in queries:
        print(f"\n[{q['type']}] Query {q['id']}: {q['query']}")
        try:
            # We assume crewapp returns a string
            response = crew_app.run(q['query'])
            res_obj = {"id": q['id'], "query": q['query'], "type": q['type'], "response": str(response)}
            results.append(res_obj)
            print("Response Length:", len(str(response)))
        except Exception as e:
            print("Error:", e)
            results.append({"id": q['id'], "query": q['query'], "type": q['type'], "error": str(e)})
            
    # Save results
    with open(eval_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nEvaluation completed. Results saved to data/evaluation/results.json")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    key = os.getenv("GROQ_API_KEY")
    if key and key != "your_groq_api_key_here":
        run_evaluation(key)
    else:
        print("Set GROQ_API_KEY in .env to run evaluations.")
