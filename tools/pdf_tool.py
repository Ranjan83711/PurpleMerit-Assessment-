"""
pdf_tool.py — CrewAI tool for direct page lookup in the PDF.
Used when an agent needs to re-verify a specific page.
"""
import os
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import pdfplumber

# Always resolve relative to project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PageLookupInput(BaseModel):
    page_number: int = Field(description="Page number to retrieve from the catalog PDF")


class PDFPageTool(BaseTool):
    """
    Retrieves raw text from a specific page of the KUK catalog PDF.
    Use to verify or expand on information from vector search results.
    """

    name: str = "catalog_page_lookup"
    description: str = (
        "Look up the raw text of a specific page in the KUK course catalog PDF. "
        "Use this when you need to verify information or get more context from a page "
        "that was referenced in a vector search citation."
    )
    args_schema: Type[BaseModel] = PageLookupInput

    pdf_path: str = "data/raw/kuk_prospectus_2011.pdf"

    def _run(self, page_number: int) -> str:
        """Return text from the specified PDF page."""
        # Resolve path: if relative, make it absolute from project root
        pdf_path = self.pdf_path
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(_PROJECT_ROOT, pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return f"ERROR: Page {page_number} does not exist. PDF has {len(pdf.pages)} pages."
                page = pdf.pages[page_number - 1]
                text = page.extract_text() or ""
                return (
                    f"PAGE {page_number} CONTENT:\n"
                    f"{'='*50}\n"
                    f"{text}\n"
                    f"{'='*50}\n"
                    f"CITATION: KUK_Catalog | Page {page_number}"
                )
        except FileNotFoundError:
            return f"ERROR: PDF not found at {pdf_path}. Please ensure the KUK catalog PDF is uploaded."
        except Exception as e:
            return f"ERROR reading page {page_number}: {str(e)}"