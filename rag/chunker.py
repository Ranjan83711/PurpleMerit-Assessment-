"""
Text Chunking Module
Splits cleaned text into overlapping chunks for embedding and retrieval.

Strategy:
- Chunk size: ~1000 characters (configurable)
- Overlap: ~200 characters to preserve context across chunk boundaries
- Respects paragraph boundaries where possible
- Each chunk retains source metadata (file, page, section)
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict
from loguru import logger


def generate_chunk_id(text: str, source: str, page: int, index: int) -> str:
    """Generate a unique chunk ID."""
    content = f"{source}_{page}_{index}_{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_length: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks, respecting paragraph boundaries.
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_length: Minimum chunk length to keep
    
    Returns:
        List of text chunks
    """
    if not text or len(text) < min_chunk_length:
        return [text] if text and text.strip() else []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, save current and start new
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Keep overlap from end of current chunk
            if chunk_overlap > 0:
                overlap_text = current_chunk[-chunk_overlap:]
                # Try to start at a word boundary
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # Handle very long paragraphs (longer than chunk_size)
        while len(current_chunk) > chunk_size * 1.5:
            # Find a good split point
            split_point = chunk_size
            # Try to split at sentence boundary
            for sep in ['. ', '.\n', '; ', '\n']:
                idx = current_chunk.rfind(sep, 0, chunk_size + 100)
                if idx > chunk_size * 0.5:
                    split_point = idx + len(sep)
                    break
            
            chunks.append(current_chunk[:split_point].strip())
            
            # Apply overlap
            if chunk_overlap > 0:
                overlap_start = max(0, split_point - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
            else:
                current_chunk = current_chunk[split_point:]
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out chunks that are too small
    chunks = [c for c in chunks if len(c) >= min_chunk_length]
    
    return chunks


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_length: int = 50,
) -> List[Dict]:
    """
    Chunk all pages into retrieval-ready documents.
    
    Args:
        pages: List of cleaned page dicts
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        min_chunk_length: Minimum chunk length
    
    Returns:
        List of chunk dicts with metadata
    """
    logger.info(f"Chunking {len(pages)} pages (size={chunk_size}, overlap={chunk_overlap})...")
    
    all_chunks = []
    chunk_counter = 0
    
    for page in pages:
        text = page.get("text", "")
        source_file = page.get("source_file", "unknown")
        page_number = page.get("page_number", 0)
        content_type = page.get("content_type", "general")
        section_heading = page.get("section_heading", "")
        
        text_chunks = split_into_chunks(text, chunk_size, chunk_overlap, min_chunk_length)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = generate_chunk_id(chunk_text, source_file, page_number, i)
            
            chunk_doc = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": i,
                "total_chunks_in_page": len(text_chunks),
                "content_type": content_type,
                "section_heading": section_heading,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            }
            all_chunks.append(chunk_doc)
            chunk_counter += 1
    
    logger.info(f"  Created {chunk_counter} chunks from {len(pages)} pages")
    
    # Statistics
    avg_words = sum(c["word_count"] for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"  Average words per chunk: {avg_words:.0f}")
    
    return all_chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """Save chunks to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.model_config import PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
    
    input_path = PROCESSED_DATA_DIR / "cleaned_text.json"
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH)
        save_chunks(chunks, str(PROCESSED_DATA_DIR / "chunks.json"))
        
        print(f"\n📊 Chunking Summary:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
        
        content_types = {}
        for c in chunks:
            ct = c.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1
        for ct, count in sorted(content_types.items()):
            print(f"   {ct}: {count} chunks")
    else:
        print(f"No cleaned text found at {input_path}. Run cleaner.py first.")
