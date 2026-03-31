"""
loader.py — Ingest PDFs (and optionally HTML/text) into raw Document objects.
Uses pdfplumber for better table/layout extraction from academic catalogs.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from pypdf import PdfReader
from langchain.schema import Document


class CatalogLoader:
    """
    Loads KUK (or any university) catalog PDFs into LangChain Documents.
    Preserves page number metadata for citation tracking.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.source_name = Path(pdf_path).stem

    def load(self) -> List[Document]:
        """
        Load PDF pages as Document objects with rich metadata.
        Uses pdfplumber for better text extraction, falls back to pypdf.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}\n"
                f"Please place your KUK prospectus PDF at: {self.pdf_path}"
            )

        documents = []
        print(f"[Loader] Loading PDF: {self.pdf_path}")

        try:
            docs = self._load_with_pdfplumber()
        except Exception as e:
            print(f"[Loader] pdfplumber failed ({e}), falling back to pypdf...")
            docs = self._load_with_pypdf()

        print(f"[Loader] Loaded {len(docs)} pages from {self.source_name}")
        return docs

    def _load_with_pdfplumber(self) -> List[Document]:
        """Extract text using pdfplumber (better for tables & columns)."""
        documents = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                # Also extract tables as text
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            if row:
                                text += "\n" + " | ".join(
                                    str(cell) if cell else "" for cell in row
                                )

                text = text.strip()
                if len(text) > 50:  # Skip near-empty pages
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": self.pdf_path,
                                "source_name": self.source_name,
                                "page": page_num,
                                "total_pages": len(pdf.pages),
                                "doc_type": "university_catalog",
                            },
                        )
                    )
        return documents

    def _load_with_pypdf(self) -> List[Document]:
        """Fallback: extract using pypdf."""
        documents = []
        reader = PdfReader(self.pdf_path)
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if len(text) > 50:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": self.pdf_path,
                            "source_name": self.source_name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "doc_type": "university_catalog",
                        },
                    )
                )
        return documents

    def save_raw(self, documents: List[Document], output_path: str):
        """Persist raw extracted text for inspection/debugging."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Loader] Saved raw text → {output_path}")
