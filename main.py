"""
Entry point for the AI Course Planner project.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import validate_config

def main():
    parser = argparse.ArgumentParser(description="AI Course Planner RAG System for KUK")
    parser.add_argument("--build", action="store_true", help="Build the vector DB index")
    parser.add_argument("--query", type=str, help="Run a manual query through the CLI")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation module")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit App")
    args = parser.parse_args()
    
    # 1. Start by Validating Configuration
    logger.info("Initializing Agentic Course Planner...")
    is_valid = validate_config()
    if not is_valid:
        logger.error("Configuration error. Please check your .env file.")
        sys.exit(1)
        
    # 2. Build Pipeline (Document ingestion -> vectorstore)
    if args.build:
        from rag.loader import load_all_pdfs, save_extracted_text
        from rag.cleaner import clean_pages, save_cleaned_text
        from rag.chunker import chunk_pages, save_chunks
        from rag.vector_store import ChromaVectorStore
        from configs.model_config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNKING_SETTINGS
        
        logger.info("Starting Build Process...")
        
        # Step 2a: Load PDF text
        if not RAW_DATA_DIR.exists():
            logger.error(f"Raw data directory missing. Please add PDFs to {RAW_DATA_DIR}")
            sys.exit(1)
            
        pages = load_all_pdfs(str(RAW_DATA_DIR))
        save_extracted_text(pages, str(PROCESSED_DATA_DIR / "extracted_text.json"))
        
        # Step 2b: Clean Text
        cleaned_pages = clean_pages(pages)
        save_cleaned_text(cleaned_pages, str(PROCESSED_DATA_DIR / "cleaned_text.json"))
        
        # Step 2c: Chunk
        chunks = chunk_pages(cleaned_pages)
        save_chunks(chunks, str(PROCESSED_DATA_DIR / "chunks.json"))
        
        # Step 2d: Embed and Store
        store = ChromaVectorStore()
        # Ensure fresh index
        store.reset_collection() 
        store.add_documents(chunks)
        
        logger.info("🚀 Vector DB indexing completed successfully!")

    # 3. CLI Query Execution
    elif args.query:
        logger.info(f"Executing manual query: '{args.query}'")
        from crew.crew_setup import CoursePlannerCrew
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_groq_api_key_here":
            logger.error("GROQ_API_KEY must be configured in .env to run queries.")
            sys.exit(1)
            
        crew_app = CoursePlannerCrew(groq_api_key=key)
        try:
            result = crew_app.run(args.query)
            print("\n" + "="*80)
            print("🤖 CrewAI Assistant Output:")
            print("="*80)
            
            # Crew Output property fallback
            output = result.raw if hasattr(result, 'raw') else str(result)
            print(output)
            print("="*80)
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")

    # 4. Run Evauation
    elif args.eval:
        from evaluation.evaluator import run_evaluation
        from dotenv import load_dotenv
        import os
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_groq_api_key_here":
            logger.error("GROQ_API_KEY must be configured in .env to run evaluations.")
            sys.exit(1)
        run_evaluation(key)

    # 5. UI Launcher
    elif args.ui:
        import subprocess
        logger.info("Starting Streamlit Web Application...")
        app_path = str(PROJECT_ROOT / "app" / "streamlit_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

    # No arguments fallback
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
