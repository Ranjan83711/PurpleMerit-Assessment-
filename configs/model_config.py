"""
Model configuration and LLM factory for CrewAI + Groq integration.
"""
import os
from dotenv import load_dotenv

# Load from env/ directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"))


def get_groq_llm(temperature: float = 0.1, model: str = None):
    """
    Returns a LangChain-compatible ChatGroq LLM instance.
    Uses llama-3.3-70b-versatile by default (free, fast, high-quality).
    """
    from langchain_groq import ChatGroq

    model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Please add your key to env/.env\n"
            "Get a free key at: https://console.groq.com"
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=2048,
        # Groq is extremely fast — no need for async tricks
    )


def get_embedding_model():
    """Returns the HuggingFace embedding model (all-MiniLM-L6-v2)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
