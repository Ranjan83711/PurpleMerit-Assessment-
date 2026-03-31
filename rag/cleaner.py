"""
cleaner.py — Clean and normalize extracted PDF text.
Academic catalogs have lots of headers, footers, page numbers, and OCR noise.
"""
import re
import json
from typing import List
from langchain.schema import Document


class TextCleaner:
    """
    Cleans raw PDF text for better chunking and retrieval.
    Handles KUK catalog-specific noise patterns.
    """

    # Patterns to remove (page headers/footers, watermarks, etc.)
    NOISE_PATTERNS = [
        r"^\s*\d+\s*$",                          # Standalone page numbers
        r"Kurukshetra University.*?\n",           # Repeated header lines
        r"K\.U\.K\s*\n",
        r"www\.kuk\.ac\.in\s*",
        r"(?i)confidential\s*",
        r"\f",                                    # Form feed characters
        r"[ \t]{3,}",                             # 3+ consecutive spaces → single space
    ]

    def clean(self, documents: List[Document]) -> List[Document]:
        """Clean all documents, return cleaned copies with updated metadata."""
        cleaned = []
        for doc in documents:
            text = doc.page_content
            text = self._clean_text(text)
            if len(text.strip()) > 80:  # Only keep pages with meaningful content
                cleaned_doc = Document(
                    page_content=text,
                    metadata={**doc.metadata, "cleaned": True},
                )
                cleaned.append(cleaned_doc)

        print(
            f"[Cleaner] {len(documents)} pages → {len(cleaned)} pages after cleaning"
        )
        return cleaned

    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps to a single text block."""
        # Fix common PDF encoding issues
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove noise patterns
        for pattern in self.NOISE_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.MULTILINE)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)          # Max 2 consecutive newlines
        text = re.sub(r"[ \t]+", " ", text)              # Normalize spaces
        text = re.sub(r" \n", "\n", text)                # Trailing spaces before newline

        # Fix split words (common in columnar PDFs)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        return text.strip()

    def save_cleaned(self, documents: List[Document], output_path: str):
        """Save cleaned documents for inspection."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Cleaner] Saved cleaned text → {output_path}")
