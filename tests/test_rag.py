"""
test_rag.py — Unit tests for RAG pipeline components.
Run: pytest tests/test_rag.py -v
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.cleaner import TextCleaner
from rag.chunker import CatalogChunker
from rag.retriever import CatalogRetriever
from langchain.schema import Document


class TestTextCleaner:
    def test_removes_extra_whitespace(self):
        cleaner = TextCleaner()
        text = "Hello   world\n\n\n\nThis is a test"
        result = cleaner._clean_text(text)
        assert "\n\n\n" not in result
        assert "   " not in result

    def test_removes_null_bytes(self):
        cleaner = TextCleaner()
        text = "Hello\x00World"
        result = cleaner._clean_text(text)
        assert "\x00" not in result

    def test_preserves_content(self):
        cleaner = TextCleaner()
        text = "Prerequisites: Mathematics-I. Students must have completed Physics-I."
        result = cleaner._clean_text(text)
        assert "Prerequisites" in result
        assert "Mathematics-I" in result

    def test_filters_short_pages(self):
        cleaner = TextCleaner()
        docs = [
            Document(page_content="hi", metadata={"page": 1}),
            Document(
                page_content="This is a real course description with prerequisites and content.",
                metadata={"page": 2},
            ),
        ]
        result = cleaner.clean(docs)
        assert len(result) == 1
        assert result[0].metadata["page"] == 2


class TestCatalogChunker:
    def test_chunk_count_increases(self):
        chunker = CatalogChunker(chunk_size=100, chunk_overlap=20)
        long_text = "This is a sentence about a course. " * 50
        docs = [Document(page_content=long_text, metadata={"page": 1, "source_name": "kuk"})]
        chunks = chunker.chunk(docs)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        chunker = CatalogChunker(chunk_size=200, chunk_overlap=50)
        docs = [
            Document(
                page_content="Course description text that is long enough to test. " * 10,
                metadata={"page": 5, "source_name": "kuk_catalog"},
            )
        ]
        chunks = chunker.chunk(docs)
        assert all("chunk_id" in c.metadata for c in chunks)
        assert all("page" in c.metadata for c in chunks)
        assert chunks[0].metadata["page"] == 5

    def test_chunk_id_format(self):
        chunker = CatalogChunker(chunk_size=200, chunk_overlap=50)
        docs = [
            Document(
                page_content="Test content for chunking. " * 10,
                metadata={"page": 3, "source_name": "kuk_test"},
            )
        ]
        chunks = chunker.chunk(docs)
        assert chunks[0].metadata["chunk_id"].startswith("kuk_test_p3_c0")


class TestCatalogRetriever:
    def test_citation_format(self):
        mock_store = MagicMock()
        mock_doc = Document(
            page_content="Sample content",
            metadata={
                "source_name": "kuk_prospectus_2011",
                "page": 42,
                "chunk_id": "kuk_prospectus_2011_p42_c0",
            },
        )
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.85)]

        retriever = CatalogRetriever(mock_store, k=6)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert "kuk_prospectus_2011" in results[0]["citation"]
        assert "Page 42" in results[0]["citation"]

    def test_score_threshold_filtering(self):
        mock_store = MagicMock()
        high_score_doc = Document(
            page_content="Relevant content",
            metadata={"source_name": "kuk", "page": 1, "chunk_id": "kuk_p1_c0"},
        )
        low_score_doc = Document(
            page_content="Irrelevant content",
            metadata={"source_name": "kuk", "page": 2, "chunk_id": "kuk_p2_c0"},
        )
        mock_store.similarity_search_with_score.return_value = [
            (high_score_doc, 0.8),
            (low_score_doc, 0.1),
        ]

        retriever = CatalogRetriever(mock_store, k=6, score_threshold=0.3)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0]["content"] == "Relevant content"

    def test_no_results_returns_empty(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []

        retriever = CatalogRetriever(mock_store, k=6)
        context, citations = retriever.retrieve_as_context("query with no results")

        assert "No relevant information" in context
        assert citations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
