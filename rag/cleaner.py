"""
Text Cleaner Module
Cleans extracted PDF text for better chunking and retrieval.
"""

import re
import json
from pathlib import Path
from typing import List, Dict
from loguru import logger


def clean_text(text: str) -> str:
    """
    Clean raw extracted PDF text.
    
    Operations:
    - Remove excessive whitespace 
    - Fix broken words from PDF line breaks
    - Normalize unicode characters
    - Remove page headers/footers patterns
    - Preserve meaningful structure (headings, lists, tables)
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\xa0', ' ')
    
    # Remove common PDF artifacts
    text = re.sub(r'\f', '\n', text)  # Form feed
    text = re.sub(r'\x00', '', text)  # Null bytes
    
    # Fix hyphenated line breaks (e.g., "pre-\nrequisite" -> "prerequisite")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Collapse multiple blank lines into max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces (but preserve indentation structure)
    text = re.sub(r'[ \t]{3,}', '  ', text)
    
    # Remove common header/footer patterns
    text = re.sub(r'Page\s+\d+\s*(of\s+\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up stray punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text.strip()


def extract_section_heading(text: str) -> str:
    """Try to extract a section heading from the beginning of text."""
    lines = text.strip().split('\n')
    for line in lines[:3]:
        line = line.strip()
        # Heuristic: headings are usually short, uppercase, or title-case
        if line and len(line) < 120:
            if line.isupper() or (len(line.split()) <= 10 and not line.endswith('.')):
                return line
    return ""


def add_metadata(page: Dict) -> Dict:
    """Enrich page data with additional metadata."""
    text = page.get("text", "")
    
    # Extract potential section heading
    heading = extract_section_heading(text)
    
    # Detect content type
    content_type = "general"
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in ["prerequisite", "pre-requisite", "prereq", "co-requisite"]):
        content_type = "prerequisite"
    elif any(kw in text_lower for kw in ["course description", "course title", "credit hours", "credits"]):
        content_type = "course_description"
    elif any(kw in text_lower for kw in ["degree requirement", "program requirement", "major requirement", "minor requirement"]):
        content_type = "program_requirement"
    elif any(kw in text_lower for kw in ["grading policy", "academic policy", "grade point", "gpa", "repeat", "credit limit"]):
        content_type = "academic_policy"
    elif any(kw in text_lower for kw in ["eligibility", "admission", "enrollment"]):
        content_type = "eligibility"
    elif any(kw in text_lower for kw in ["syllabus", "scheme of examination", "examination"]):
        content_type = "syllabus"
    
    page["section_heading"] = heading
    page["content_type"] = content_type
    page["word_count"] = len(text.split())
    
    return page


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Clean all extracted pages.
    
    Args:
        pages: List of page dicts from loader
    
    Returns:
        Cleaned and enriched page dicts
    """
    logger.info(f"Cleaning {len(pages)} pages...")
    
    cleaned = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        if cleaned_text and len(cleaned_text.split()) >= 10:  # Min 10 words
            page["text"] = cleaned_text
            page = add_metadata(page)
            cleaned.append(page)
    
    removed = len(pages) - len(cleaned)
    if removed:
        logger.info(f"  Removed {removed} near-empty pages")
    
    logger.info(f"  {len(cleaned)} pages after cleaning")
    return cleaned


def save_cleaned_text(pages: List[Dict], output_path: str):
    """Save cleaned text to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved cleaned text to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.model_config import PROCESSED_DATA_DIR
    
    input_path = PROCESSED_DATA_DIR / "extracted_text.json"
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        
        cleaned = clean_pages(pages)
        save_cleaned_text(cleaned, str(PROCESSED_DATA_DIR / "cleaned_text.json"))
        
        # Summary
        content_types = {}
        for p in cleaned:
            ct = p.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1
        
        print(f"\n📊 Cleaning Summary:")
        print(f"   Pages cleaned: {len(cleaned)}")
        print(f"   Content types:")
        for ct, count in sorted(content_types.items()):
            print(f"     {ct}: {count}")
    else:
        print(f"No extracted text found at {input_path}. Run loader.py first.")
