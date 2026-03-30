"""
Model Configuration Module
Centralizes all model and environment configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from loguru import logger

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config file not found at {config_path}, using defaults.")
    return {}


# Load YAML config
CONFIG = load_config()

# ============================================
# Groq LLM Settings
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv(
    "GROQ_MODEL_NAME",
    CONFIG.get("llm", {}).get("model_name", "llama-3.3-70b-versatile"),
)
GROQ_TEMPERATURE = float(
    os.getenv("GROQ_TEMPERATURE", CONFIG.get("llm", {}).get("temperature", 0.1))
)
GROQ_MAX_TOKENS = int(
    os.getenv("GROQ_MAX_TOKENS", CONFIG.get("llm", {}).get("max_tokens", 4096))
)

# ============================================
# Embedding Model Settings
# ============================================
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    CONFIG.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2"),
)
EMBEDDING_DEVICE = CONFIG.get("embedding", {}).get("device", "cpu")
EMBEDDING_BATCH_SIZE = CONFIG.get("embedding", {}).get("batch_size", 64)

# ============================================
# ChromaDB Settings
# ============================================
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(
        PROJECT_ROOT
        / CONFIG.get("vector_store", {}).get("persist_directory", "vectorstore/chroma_db")
    ),
)
CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    CONFIG.get("vector_store", {}).get("collection_name", "kuk_course_catalog"),
)

# ============================================
# RAG Pipeline Settings
# ============================================
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", CONFIG.get("rag", {}).get("chunk_size", 1000))
)
CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", CONFIG.get("rag", {}).get("chunk_overlap", 200))
)
MIN_CHUNK_LENGTH = CONFIG.get("rag", {}).get("min_chunk_length", 50)
MAX_CHUNK_LENGTH = CONFIG.get("rag", {}).get("max_chunk_length", 2000)

# ============================================
# Retriever Settings
# ============================================
RETRIEVER_TOP_K = int(
    os.getenv("RETRIEVER_TOP_K", CONFIG.get("retriever", {}).get("top_k", 8))
)
SIMILARITY_METRIC = os.getenv(
    "SIMILARITY_METRIC",
    CONFIG.get("retriever", {}).get("similarity_metric", "cosine"),
)
SCORE_THRESHOLD = float(
    CONFIG.get("retriever", {}).get("score_threshold", 0.3)
)

# ============================================
# Data Paths
# ============================================
RAW_DATA_DIR = PROJECT_ROOT / CONFIG.get("data", {}).get("raw_dir", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / CONFIG.get("data", {}).get(
    "processed_dir", "data/processed"
)
EVALUATION_DIR = PROJECT_ROOT / CONFIG.get("data", {}).get(
    "evaluation_dir", "data/evaluation"
)

# ============================================
# CrewAI Settings
# ============================================
CREW_VERBOSE = CONFIG.get("crew", {}).get("verbose", True)
CREW_PROCESS = CONFIG.get("crew", {}).get("process", "sequential")
CREW_MAX_ITERATIONS = CONFIG.get("crew", {}).get("max_iterations", 10)

# ============================================
# Logging
# ============================================
LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO")
LOG_FORMAT = CONFIG.get("logging", {}).get("format", "{time} | {level} | {message}")
LOG_FILE = PROJECT_ROOT / CONFIG.get("logging", {}).get("file", "logs/app.log")


def validate_config():
    """Validate critical configuration values."""
    errors = []

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        errors.append(
            "GROQ_API_KEY is not set. Get one from https://console.groq.com/keys"
        )

    if not RAW_DATA_DIR.exists():
        logger.warning(f"Raw data directory does not exist: {RAW_DATA_DIR}")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    logger.info("✅ Configuration validated successfully.")
    return True


def get_llm_config() -> dict:
    """Get LLM configuration as a dictionary."""
    return {
        "provider": "groq",
        "model": GROQ_MODEL_NAME,
        "api_key": GROQ_API_KEY,
        "temperature": GROQ_TEMPERATURE,
        "max_tokens": GROQ_MAX_TOKENS,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    print(f"Groq Model: {GROQ_MODEL_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"ChromaDB Dir: {CHROMA_PERSIST_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Retriever Top-K: {RETRIEVER_TOP_K}")
    print(f"API Key Set: {'Yes' if GROQ_API_KEY else 'No'}")
    validate_config()
