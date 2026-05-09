"""
Text chunking utilities for splitting documents into RAG-friendly segments.
"""
import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 0
) -> List[str]:
    """
    Split text into chunks that respect sentence boundaries.

    The function splits text at natural sentence boundaries (periods, exclamation
    marks, question marks, semicolons, newlines) while ensuring chunks don't
    exceed the specified maximum size.

    Args:
        text: The text to split into chunks.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between chunks (for context).

    Returns:
        List of text chunks, each at most chunk_size characters.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks = []
    split_pattern = r'.{1,%d}[。|\.|\!|\?|;|；|!|\?|\n]' % chunk_size

    segments = re.findall(split_pattern, text, flags=re.DOTALL)

    for segment in segments:
        stripped = segment.strip()
        if stripped:
            chunks.append(stripped)

    remaining = re.split(split_pattern, text, flags=re.DOTALL)
    for leftover in remaining:
        if leftover and len(leftover.strip()) > 0:
            if len(leftover) <= chunk_size:
                chunks.append(leftover.strip())
            else:
                for sub_chunk in _split_by_max_length(leftover, chunk_size):
                    if sub_chunk.strip():
                        chunks.append(sub_chunk.strip())

    return chunks


def _split_by_max_length(text: str, max_length: int) -> List[str]:
    """Split text by maximum length when no sentence boundaries found."""
    if not text:
        return []

    pattern = r'.{1,%d}' % max_length
    return re.findall(pattern, text, flags=re.DOTALL)


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500
) -> List[str]:
    """
    Split text into chunks by grouping sentences.

    This function first splits text into individual sentences, then groups
    them into chunks that don't exceed the maximum size.

    Args:
        text: The text to split into chunks.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    sentence_pattern = r'[^。\.!\?;；\n]+[。\.!\?;；\n]?'
    sentences = re.findall(sentence_pattern, text, flags=re.DOTALL)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for sub in _split_by_max_length(sentence, max_chunk_size):
                if sub.strip():
                    chunks.append(sub.strip())
        elif len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_with_overlap(
    text: str,
    chunk_size: int = 256,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.

    Args:
        text: The text to split into chunks.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks with overlap.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks
