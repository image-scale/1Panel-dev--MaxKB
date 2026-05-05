"""
Text chunking utilities for the KnowledgeBot platform.

Provides functions to split text into smaller chunks for processing and embedding.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledgebot.knowledge.documents import Document


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 512
    overlap: int = 50

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")


# Sentence boundary pattern: period, exclamation, or question mark followed by space or end
SENTENCE_BOUNDARY = re.compile(r'[.!?]\s+')


def _find_split_point(text: str, max_pos: int, min_pos: int) -> int:
    """
    Find the best split point in text, preferring sentence boundaries.

    Args:
        text: The text to search
        max_pos: Maximum position to consider
        min_pos: Minimum position to consider

    Returns:
        The best split position
    """
    # Look for sentence boundaries in the search range
    search_text = text[min_pos:max_pos]
    matches = list(SENTENCE_BOUNDARY.finditer(search_text))

    if matches:
        # Use the last sentence boundary in the range
        last_match = matches[-1]
        return min_pos + last_match.end()

    # No sentence boundary found, just split at max_pos
    return max_pos


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.

    The function tries to split at sentence boundaries (., !, ?) when possible
    to maintain semantic coherence.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    # Handle empty or whitespace-only text
    text = text.strip()
    if not text:
        return []

    # If text fits in one chunk, return it
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Calculate the end of this chunk
        end = start + chunk_size

        if end >= len(text):
            # Last chunk - take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find a good split point
        min_pos = max(start + chunk_size - overlap, start + 1)
        split_pos = _find_split_point(text, end, min_pos - start) + start

        # Ensure we don't exceed chunk_size
        if split_pos > end:
            split_pos = end

        # Extract chunk
        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Move start for next chunk, accounting for overlap
        start = split_pos - overlap
        if start < 0:
            start = 0

        # Avoid infinite loop
        if start >= split_pos:
            start = split_pos

    return chunks


def chunk_document(
    document: "Document",
    config: Optional[ChunkingConfig] = None
) -> List[str]:
    """
    Chunk a document's content.

    Args:
        document: The document to chunk
        config: Chunking configuration (uses defaults if None)

    Returns:
        List of text chunks from the document
    """
    if config is None:
        config = ChunkingConfig()

    return chunk_text(document.content, config.chunk_size, config.overlap)


def count_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> int:
    """
    Count how many chunks a text would produce without actually creating them.

    Args:
        text: The text to count chunks for
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks

    Returns:
        Number of chunks that would be produced
    """
    return len(chunk_text(text, chunk_size, overlap))


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    index: int
    start_pos: int
    end_pos: int

    @property
    def length(self) -> int:
        """Return the length of the chunk content."""
        return len(self.content)


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[TextChunk]:
    """
    Split text into chunks with position metadata.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of TextChunk objects with position information
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [TextChunk(content=text, index=0, start_pos=0, end_pos=len(text))]

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk_text_content = text[start:].strip()
            if chunk_text_content:
                chunks.append(TextChunk(
                    content=chunk_text_content,
                    index=index,
                    start_pos=start,
                    end_pos=len(text)
                ))
            break

        min_pos = max(start + chunk_size - overlap, start + 1)
        split_pos = _find_split_point(text, end, min_pos - start) + start

        if split_pos > end:
            split_pos = end

        chunk_text_content = text[start:split_pos].strip()
        if chunk_text_content:
            chunks.append(TextChunk(
                content=chunk_text_content,
                index=index,
                start_pos=start,
                end_pos=split_pos
            ))
            index += 1

        start = split_pos - overlap
        if start < 0:
            start = 0
        if start >= split_pos:
            start = split_pos

    return chunks
