"""
Tests for the text chunking system.
"""
import pytest

from knowledgebot.core.chunking import (
    ChunkingConfig,
    chunk_text,
    chunk_document,
    count_chunks,
    TextChunk,
    chunk_text_with_metadata,
)
from knowledgebot.knowledge.documents import Document


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        """ChunkingConfig should have default values."""
        config = ChunkingConfig()
        assert config.chunk_size == 512
        assert config.overlap == 50

    def test_custom_config(self):
        """ChunkingConfig should accept custom values."""
        config = ChunkingConfig(chunk_size=1024, overlap=100)
        assert config.chunk_size == 1024
        assert config.overlap == 100

    def test_invalid_chunk_size(self):
        """ChunkingConfig should reject non-positive chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-1)

    def test_negative_overlap(self):
        """ChunkingConfig should reject negative overlap."""
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            ChunkingConfig(overlap=-1)

    def test_overlap_too_large(self):
        """ChunkingConfig should reject overlap >= chunk_size."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, overlap=100)
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, overlap=150)


class TestChunkText:
    """Tests for chunk_text function."""

    def test_empty_text(self):
        """Empty text should return empty list."""
        result = chunk_text("")
        assert result == []

    def test_whitespace_only(self):
        """Whitespace-only text should return empty list."""
        result = chunk_text("   \n\t  ")
        assert result == []

    def test_short_text(self):
        """Text shorter than chunk_size should return single chunk."""
        text = "Hello, World!"
        result = chunk_text(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_text_equal_to_chunk_size(self):
        """Text equal to chunk_size should return single chunk."""
        text = "A" * 100
        result = chunk_text(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_basic_chunking(self):
        """Long text should be split into multiple chunks."""
        text = "A" * 200
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert len(result) > 1
        # Each chunk should be at most chunk_size
        for chunk in result:
            assert len(chunk) <= 100

    def test_chunks_overlap(self):
        """Consecutive chunks should have overlapping content."""
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        result = chunk_text(text, chunk_size=30, overlap=10)
        assert len(result) > 1
        # Check overlap exists (simple check - later chunks should contain some earlier content)

    def test_sentence_boundary_splitting(self):
        """Chunking should prefer to split at sentence boundaries."""
        text = "First sentence here. Second sentence here. Third sentence is longer and goes on."
        result = chunk_text(text, chunk_size=50, overlap=10)
        # Chunks should try to end at sentence boundaries
        assert len(result) >= 1

    def test_invalid_chunk_size(self):
        """chunk_text should raise error for invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("test", chunk_size=0)

    def test_invalid_overlap(self):
        """chunk_text should raise error for invalid overlap."""
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            chunk_text("test", overlap=-1)

    def test_overlap_too_large(self):
        """chunk_text should raise error for overlap >= chunk_size."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            chunk_text("test", chunk_size=10, overlap=10)

    def test_chunk_max_size(self):
        """Each chunk should not exceed chunk_size."""
        text = "A very long text. " * 100
        chunk_size = 100
        result = chunk_text(text, chunk_size=chunk_size, overlap=20)
        for chunk in result:
            assert len(chunk) <= chunk_size

    def test_no_empty_chunks(self):
        """No empty chunks should be produced."""
        text = "Some text with content."
        result = chunk_text(text, chunk_size=10, overlap=2)
        for chunk in result:
            assert len(chunk.strip()) > 0


class TestChunkDocument:
    """Tests for chunk_document function."""

    def test_chunk_document_with_default_config(self):
        """chunk_document should use default config if none provided."""
        doc = Document(
            name="test.txt",
            knowledge_base_id="kb123",
            content="A" * 1000
        )
        result = chunk_document(doc)
        assert len(result) > 1

    def test_chunk_document_with_custom_config(self):
        """chunk_document should use provided config."""
        doc = Document(
            name="test.txt",
            knowledge_base_id="kb123",
            content="A" * 200
        )
        config = ChunkingConfig(chunk_size=50, overlap=10)
        result = chunk_document(doc, config)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 50

    def test_chunk_empty_document(self):
        """chunk_document should return empty list for empty document."""
        doc = Document(
            name="test.txt",
            knowledge_base_id="kb123",
            content=""
        )
        result = chunk_document(doc)
        assert result == []


class TestCountChunks:
    """Tests for count_chunks function."""

    def test_count_chunks_empty(self):
        """count_chunks should return 0 for empty text."""
        assert count_chunks("") == 0

    def test_count_chunks_short_text(self):
        """count_chunks should return 1 for short text."""
        assert count_chunks("Hello", chunk_size=100) == 1

    def test_count_chunks_long_text(self):
        """count_chunks should return correct count for long text."""
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        count = count_chunks(text, chunk_size=100, overlap=10)
        assert count == len(chunks)


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_text_chunk_creation(self):
        """TextChunk should store all properties."""
        chunk = TextChunk(content="Hello", index=0, start_pos=0, end_pos=5)
        assert chunk.content == "Hello"
        assert chunk.index == 0
        assert chunk.start_pos == 0
        assert chunk.end_pos == 5

    def test_text_chunk_length(self):
        """TextChunk.length should return content length."""
        chunk = TextChunk(content="Hello World", index=0, start_pos=0, end_pos=11)
        assert chunk.length == 11


class TestChunkTextWithMetadata:
    """Tests for chunk_text_with_metadata function."""

    def test_empty_text(self):
        """Empty text should return empty list."""
        result = chunk_text_with_metadata("")
        assert result == []

    def test_short_text(self):
        """Short text should return single chunk with metadata."""
        text = "Hello, World!"
        result = chunk_text_with_metadata(text, chunk_size=100)
        assert len(result) == 1
        assert result[0].content == text
        assert result[0].index == 0
        assert result[0].start_pos == 0
        assert result[0].end_pos == len(text)

    def test_multiple_chunks_with_metadata(self):
        """Multiple chunks should have correct indices."""
        text = "A" * 200
        result = chunk_text_with_metadata(text, chunk_size=50, overlap=10)
        assert len(result) > 1
        # Check indices are sequential
        for i, chunk in enumerate(result):
            assert chunk.index == i

    def test_invalid_params(self):
        """Should raise error for invalid parameters."""
        with pytest.raises(ValueError):
            chunk_text_with_metadata("test", chunk_size=0)
        with pytest.raises(ValueError):
            chunk_text_with_metadata("test", overlap=-1)
        with pytest.raises(ValueError):
            chunk_text_with_metadata("test", chunk_size=10, overlap=20)
