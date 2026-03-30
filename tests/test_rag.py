"""
Unit tests for the RAG components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.cleaner import clean_text
from rag.chunker import split_into_chunks

def test_clean_text():
    raw = "Course  Name: \n\n\nCS101.   \f\x00"
    cleaned = clean_text(raw)
    assert "CS101" in cleaned
    assert "\f" not in cleaned


def test_chunker_basic():
    long_text = "A" * 1200
    chunks = split_into_chunks(long_text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    assert len(chunks[0]) <= 1100  # leeway
