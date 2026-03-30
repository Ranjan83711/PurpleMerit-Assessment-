"""
PDF Tool for CrewAI Agents
Allows agents to load and search raw PDF content directly.
"""

import sys
from pathlib import Path
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

sys.path.insert(0, str(Path(__file__).parent.parent))


class PDFSearchInput(BaseModel):
    """Input schema for PDF search tool."""
    filename: str = Field(..., description="Name of the PDF file to search")
    search_term: str = Field(..., description="Term to search for in the PDF")


class PDFSearchTool(BaseTool):
    """Tool to search within specific PDF files."""
    
    name: str = "pdf_search"
    description: str = (
        "Search for specific terms within a PDF file from the catalog. "
        "Useful for finding exact text, tables, or specific sections."
    )
    args_schema: Type[BaseModel] = PDFSearchInput
    
    _data_dir: str = ""
    
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        if data_dir:
            self._data_dir = data_dir
        else:
            from configs.model_config import RAW_DATA_DIR
            self._data_dir = str(RAW_DATA_DIR)
    
    def _run(self, filename: str, search_term: str) -> str:
        """Search within a PDF file for the given term."""
        try:
            from rag.loader import load_pdf
            
            pdf_path = Path(self._data_dir) / filename
            if not pdf_path.exists():
                # Try to find the file
                available = list(Path(self._data_dir).glob("*.pdf"))
                available_names = [f.name for f in available]
                return (
                    f"File '{filename}' not found. "
                    f"Available PDFs: {available_names}"
                )
            
            pages = load_pdf(str(pdf_path))
            search_lower = search_term.lower()
            
            matches = []
            for page in pages:
                if search_lower in page["text"].lower():
                    # Extract relevant snippet
                    text = page["text"]
                    idx = text.lower().find(search_lower)
                    start = max(0, idx - 200)
                    end = min(len(text), idx + len(search_term) + 200)
                    snippet = text[start:end]
                    
                    matches.append(
                        f"[Page {page['page_number']}] ...{snippet}..."
                    )
            
            if not matches:
                return f"No matches found for '{search_term}' in {filename}."
            
            return f"Found {len(matches)} matches in {filename}:\n\n" + "\n\n".join(matches[:5])
        
        except Exception as e:
            return f"Error searching PDF: {str(e)}"
