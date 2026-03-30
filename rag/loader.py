"""
PDF Loader Module
Extracts text from PDF files using pdfplumber with PyMuPDF fallback.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def extract_text_pdfplumber(pdf_path: str) -> List[Dict]:
    """Extract text from PDF using pdfplumber (better for tables/structured content)."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed.")
    
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "page_number": i + 1,
                    "text": text.strip(),
                    "source_file": os.path.basename(pdf_path),
                    "source_path": str(pdf_path),
                })
    return pages


def extract_text_pymupdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF using PyMuPDF (faster, good for most PDFs)."""
    if fitz is None:
        raise ImportError("PyMuPDF is not installed.")
    
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        if text.strip():
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
                "source_file": os.path.basename(pdf_path),
                "source_path": str(pdf_path),
            })
    doc.close()
    return pages


def load_pdf(pdf_path: str, method: str = "pdfplumber") -> List[Dict]:
    """
    Load a single PDF and extract text page by page.
    
    Args:
        pdf_path: Path to the PDF file
        method: Extraction method - 'pdfplumber' or 'pymupdf'
    
    Returns:
        List of dicts with page_number, text, source_file, source_path
    """
    pdf_path = str(Path(pdf_path).resolve())
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return []
    
    logger.info(f"Loading PDF: {os.path.basename(pdf_path)} (method: {method})")
    
    try:
        if method == "pdfplumber" and pdfplumber is not None:
            pages = extract_text_pdfplumber(pdf_path)
        elif method == "pymupdf" and fitz is not None:
            pages = extract_text_pymupdf(pdf_path)
        else:
            # Fallback: try both
            try:
                pages = extract_text_pdfplumber(pdf_path)
            except Exception:
                pages = extract_text_pymupdf(pdf_path)
        
        logger.info(f"  Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
        return pages
    
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return []


def load_all_pdfs(data_dir: str, method: str = "pdfplumber") -> List[Dict]:
    """
    Load all PDFs from a directory.
    
    Args:
        data_dir: Directory containing PDF files
        method: Extraction method
    
    Returns:
        Combined list of all pages from all PDFs
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    all_pages = []
    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file), method=method)
        all_pages.extend(pages)
    
    logger.info(f"Total pages extracted: {len(all_pages)}")
    return all_pages


def save_extracted_text(pages: List[Dict], output_path: str):
    """Save extracted text to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved extracted text to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.model_config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    pages = load_all_pdfs(str(RAW_DATA_DIR))
    if pages:
        save_extracted_text(pages, str(PROCESSED_DATA_DIR / "extracted_text.json"))
        
        # Print summary
        total_words = sum(len(p["text"].split()) for p in pages)
        print(f"\n📊 Extraction Summary:")
        print(f"   Total pages: {len(pages)}")
        print(f"   Total words: {total_words:,}")
        
        sources = set(p["source_file"] for p in pages)
        for src in sources:
            src_pages = [p for p in pages if p["source_file"] == src]
            src_words = sum(len(p["text"].split()) for p in src_pages)
            print(f"   {src}: {len(src_pages)} pages, {src_words:,} words")
