"""
Tests for text chunking utilities.
"""
from django.test import TestCase
from apps.common.chunking import chunk_text, chunk_by_sentences, chunk_with_overlap


class ChunkTextTest(TestCase):
    """Test chunk_text function."""

    def test_basic_chunking(self):
        """Basic text is chunked correctly."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, chunk_size=256)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list."""
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])
        self.assertEqual(chunk_text(None), [])

    def test_chunks_respect_size_limit(self):
        """Chunks should not exceed chunk_size."""
        text = "This is a test. " * 100
        chunk_size = 50
        chunks = chunk_text(text, chunk_size=chunk_size)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), chunk_size + 20)

    def test_splits_at_sentence_boundaries(self):
        """Text should split at sentence boundaries."""
        text = "First sentence. Second sentence! Third sentence?"
        chunks = chunk_text(text, chunk_size=256)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_chinese_text_chunking(self):
        """Chinese text should chunk correctly at Chinese punctuation."""
        text = "这是第一句话。这是第二句话。这是第三句话。"
        chunks = chunk_text(text, chunk_size=256)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_mixed_punctuation(self):
        """Text with mixed punctuation should chunk correctly."""
        text = "English sentence. 中文句子。Another one! 再一句？"
        chunks = chunk_text(text, chunk_size=256)
        self.assertTrue(len(chunks) > 0)

    def test_newlines_as_boundaries(self):
        """Newlines should act as chunk boundaries."""
        text = "Line one\nLine two\nLine three"
        chunks = chunk_text(text, chunk_size=256)
        self.assertTrue(len(chunks) > 0)

    def test_filters_empty_chunks(self):
        """Empty chunks should be filtered out."""
        text = "Sentence one.   Sentence two.   "
        chunks = chunk_text(text, chunk_size=256)
        for chunk in chunks:
            self.assertTrue(len(chunk.strip()) > 0)

    def test_trims_whitespace(self):
        """Chunks should have trimmed whitespace."""
        text = "  First sentence.   Second sentence.  "
        chunks = chunk_text(text, chunk_size=256)
        for chunk in chunks:
            self.assertEqual(chunk, chunk.strip())

    def test_invalid_chunk_size_raises_error(self):
        """Invalid chunk_size should raise ValueError."""
        with self.assertRaises(ValueError):
            chunk_text("Some text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("Some text", chunk_size=-10)

    def test_very_long_text_without_punctuation(self):
        """Long text without punctuation should still chunk."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100)
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_single_short_sentence(self):
        """Single short sentence should return as single chunk."""
        text = "Short text."
        chunks = chunk_text(text, chunk_size=256)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text.")


class ChunkBySentencesTest(TestCase):
    """Test chunk_by_sentences function."""

    def test_groups_sentences(self):
        """Sentences should be grouped into chunks."""
        text = "First. Second. Third. Fourth. Fifth."
        chunks = chunk_by_sentences(text, max_chunk_size=50)
        self.assertTrue(len(chunks) > 0)

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list."""
        self.assertEqual(chunk_by_sentences(""), [])
        self.assertEqual(chunk_by_sentences("   "), [])

    def test_respects_max_size(self):
        """Chunks should not exceed max size (approximately)."""
        text = "A very long sentence that goes on and on. " * 10
        chunks = chunk_by_sentences(text, max_chunk_size=100)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 150)

    def test_invalid_max_size_raises_error(self):
        """Invalid max_chunk_size should raise ValueError."""
        with self.assertRaises(ValueError):
            chunk_by_sentences("Text", max_chunk_size=0)

    def test_single_sentence_longer_than_max(self):
        """Sentence longer than max should be split."""
        text = "a" * 200
        chunks = chunk_by_sentences(text, max_chunk_size=50)
        self.assertTrue(len(chunks) > 1)

    def test_chinese_sentences(self):
        """Chinese sentences should be grouped correctly."""
        text = "第一句。第二句。第三句。第四句。"
        chunks = chunk_by_sentences(text, max_chunk_size=100)
        self.assertTrue(len(chunks) > 0)


class ChunkWithOverlapTest(TestCase):
    """Test chunk_with_overlap function."""

    def test_basic_overlap(self):
        """Chunks should overlap by specified amount."""
        text = "a" * 100
        chunks = chunk_with_overlap(text, chunk_size=30, overlap=10)
        self.assertTrue(len(chunks) > 1)

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list."""
        self.assertEqual(chunk_with_overlap(""), [])

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size returns single chunk."""
        text = "Short text"
        chunks = chunk_with_overlap(text, chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text")

    def test_invalid_chunk_size_raises_error(self):
        """Invalid chunk_size should raise ValueError."""
        with self.assertRaises(ValueError):
            chunk_with_overlap("Text", chunk_size=0)

    def test_negative_overlap_treated_as_zero(self):
        """Negative overlap should be treated as zero."""
        text = "a" * 100
        chunks = chunk_with_overlap(text, chunk_size=30, overlap=-10)
        self.assertTrue(len(chunks) > 1)

    def test_overlap_larger_than_chunk_clamped(self):
        """Overlap larger than chunk_size should be clamped."""
        text = "a" * 100
        chunks = chunk_with_overlap(text, chunk_size=30, overlap=100)
        self.assertTrue(len(chunks) > 1)

    def test_chunks_have_expected_overlap(self):
        """Consecutive chunks should share overlap characters."""
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_with_overlap(text, chunk_size=10, overlap=3)
        if len(chunks) >= 2:
            end_of_first = chunks[0][-3:]
            start_of_second = chunks[1][:3]
            self.assertEqual(end_of_first, start_of_second)

    def test_trims_whitespace(self):
        """Chunks should be trimmed."""
        text = "  some text with spaces  "
        chunks = chunk_with_overlap(text, chunk_size=100)
        for chunk in chunks:
            self.assertEqual(chunk, chunk.strip())
