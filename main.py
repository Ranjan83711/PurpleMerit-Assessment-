"""
main.py — Entry point for the KUK Course Planning RAG Assistant.

Commands:
  python main.py --build-index       Build ChromaDB index from PDF
  python main.py --chat              Interactive CLI chat
  python main.py --evaluate          Run the 25-query evaluation suite
  python main.py --query "..."       Single query from command line
"""
import os
import sys
import argparse

# Always run relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "env", ".env"))


def build_index():
    """Full pipeline: Load PDF → Clean → Chunk → Embed → Store in ChromaDB."""
    from rag.loader import CatalogLoader
    from rag.cleaner import TextCleaner
    from rag.chunker import CatalogChunker
    from rag.vector_store import CourseVectorStore
    from configs.model_config import get_embedding_model

    pdf_path = os.getenv("PDF_PATH", "data/raw/kuk_prospectus_2011.pdf")
    chunk_size = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))

    print("\n" + "="*60)
    print("BUILDING RAG INDEX")
    print("="*60)

    # Step 1: Load
    loader = CatalogLoader(pdf_path)
    raw_docs = loader.load()
    loader.save_raw(raw_docs, "data/processed/raw_text.json")

    # Step 2: Clean
    cleaner = TextCleaner()
    clean_docs = cleaner.clean(raw_docs)
    cleaner.save_cleaned(clean_docs, "data/processed/cleaned_text.json")

    # Step 3: Chunk
    chunker = CatalogChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(clean_docs)
    chunker.save_chunks(chunks, "data/processed/chunks.json")

    # Step 4: Embed + Store
    embeddings = get_embedding_model()
    vs = CourseVectorStore(
        persist_directory=os.getenv("CHROMA_DB_PATH", "vectorstore/chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "kuk_course_catalog"),
    )
    vs.build(chunks, embeddings)

    print("\n✅ Index built successfully!")
    print(f"   Pages loaded:  {len(raw_docs)}")
    print(f"   Pages cleaned: {len(clean_docs)}")
    print(f"   Chunks stored: {len(chunks)}")
    print(f"   ChromaDB path: {os.getenv('CHROMA_DB_PATH', 'vectorstore/chroma_db')}")


def load_crew():
    """Load the crew pipeline (LLM + vector store + agents)."""
    from configs.model_config import get_groq_llm, get_embedding_model
    from rag.vector_store import CourseVectorStore
    from rag.retriever import CatalogRetriever
    from crew.crew_setup import CoursePlanningCrew

    print("Loading LLM (Groq llama-3.3-70b-versatile)...")
    llm = get_groq_llm()

    print("Loading embeddings (all-MiniLM-L6-v2)...")
    embeddings = get_embedding_model()

    print("Loading ChromaDB vector store...")
    vs = CourseVectorStore(
        persist_directory=os.getenv("CHROMA_DB_PATH", "vectorstore/chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "kuk_course_catalog"),
    )
    vs.load(embeddings)

    retriever = CatalogRetriever(vs, k=int(os.getenv("RETRIEVER_K", 6)))
    pdf_path = os.path.join(PROJECT_ROOT, os.getenv("PDF_PATH", "data/raw/kuk_prospectus_2011.pdf"))
    crew = CoursePlanningCrew(llm=llm, retriever=retriever, pdf_path=pdf_path)
    return crew


def run_chat():
    """Interactive CLI chat mode."""
    crew = load_crew()

    print("\n" + "="*60)
    print("KUK COURSE PLANNING ASSISTANT")
    print("Type 'quit' to exit | 'plan' for course plan | 'check' for eligibility")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Detect query type
            lower = user_input.lower()
            if any(k in lower for k in ["plan", "schedule", "next semester", "next term", "what courses"]):
                result = crew.run_course_plan(user_input)
            elif any(k in lower for k in ["can i take", "eligible", "prerequisite", "do i need"]):
                result = crew.run_eligibility_check(user_input)
            else:
                result = crew.run_general_query(user_input)

            print("\n" + "="*60)
            print("ASSISTANT:")
            print("="*60)
            print(result.get("formatted", result.get("raw_output", "No output")))
            print("="*60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def run_single_query(query: str):
    """Run a single query and print result."""
    crew = load_crew()
    lower = query.lower()

    if any(k in lower for k in ["plan", "schedule", "next semester"]):
        result = crew.run_course_plan(query)
    elif any(k in lower for k in ["can i take", "eligible", "prerequisite"]):
        result = crew.run_eligibility_check(query)
    else:
        result = crew.run_general_query(query)

    print("\n" + "="*60)
    print(result.get("formatted", result.get("raw_output", "")))
    print("="*60)


def run_evaluation(max_queries=None):
    """Run the 25-query evaluation suite."""
    from evaluation.evaluator import CourseAssistantEvaluator
    crew = load_crew()
    evaluator = CourseAssistantEvaluator(
        crew=crew,
        test_queries_path="data/evaluation/test_queries.json",
        output_path="data/evaluation/eval_results.json",
    )
    evaluator.run(max_queries=max_queries, delay_seconds=3.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KUK Course Planning RAG Assistant"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the ChromaDB index from the KUK catalog PDF",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive CLI chat",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the 25-query evaluation suite",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Max queries for evaluation run",
    )

    args = parser.parse_args()

    if args.build_index:
        build_index()
    elif args.chat:
        run_chat()
    elif args.evaluate:
        run_evaluation(max_queries=args.max_eval)
    elif args.query:
        run_single_query(args.query)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Add your GROQ_API_KEY to env/.env")
        print("  2. Place KUK PDF at data/raw/kuk_prospectus_2011.pdf")
        print("  3. python main.py --build-index")
        print("  4. python main.py --chat")
        print("  5. streamlit run app/streamlit_app.py")