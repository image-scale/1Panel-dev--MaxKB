"""
Tests for vector and keyword search utilities.
"""
import math
from django.test import TestCase
from apps.common.search import (
    cosine_similarity, euclidean_distance, normalize_text, tokenize,
    keyword_match_score, embedding_search, keyword_search, blend_search,
    search, SearchMode, SearchResult
)


class CosineimilarityTest(TestCase):
    """Test cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=5)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0, places=5)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0, places=5)

    def test_empty_vectors(self):
        """Empty vectors return 0.0."""
        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([1.0], []), 0.0)

    def test_different_length_vectors(self):
        """Different length vectors return 0.0."""
        self.assertEqual(cosine_similarity([1.0, 2.0], [1.0]), 0.0)

    def test_zero_magnitude_vector(self):
        """Zero magnitude vector returns 0.0."""
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_similar_vectors(self):
        """Similar vectors have high similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec1, vec2)
        self.assertGreater(similarity, 0.99)

    def test_normalized_vectors(self):
        """Normalized vectors work correctly."""
        vec1 = [0.6, 0.8]
        vec2 = [0.8, 0.6]
        similarity = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.96, places=2)


class EuclideanDistanceTest(TestCase):
    """Test euclidean distance calculation."""

    def test_identical_vectors(self):
        """Identical vectors have distance 0.0."""
        vec = [1.0, 2.0, 3.0]
        self.assertEqual(euclidean_distance(vec, vec), 0.0)

    def test_simple_distance(self):
        """Simple distance calculation."""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        self.assertEqual(euclidean_distance(vec1, vec2), 5.0)

    def test_empty_vectors(self):
        """Empty vectors return infinity."""
        self.assertEqual(euclidean_distance([], []), float('inf'))

    def test_different_length_vectors(self):
        """Different length vectors return infinity."""
        self.assertEqual(euclidean_distance([1.0], [1.0, 2.0]), float('inf'))


class NormalizeTextTest(TestCase):
    """Test text normalization."""

    def test_lowercase(self):
        """Text is lowercased."""
        self.assertEqual(normalize_text("Hello WORLD"), "hello world")

    def test_whitespace_collapsed(self):
        """Multiple whitespaces are collapsed."""
        self.assertEqual(normalize_text("hello   world"), "hello world")

    def test_trim_whitespace(self):
        """Leading and trailing whitespace is trimmed."""
        self.assertEqual(normalize_text("  hello  "), "hello")

    def test_empty_string(self):
        """Empty string returns empty."""
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")


class TokenizeTest(TestCase):
    """Test tokenization."""

    def test_english_words(self):
        """English words are tokenized."""
        tokens = tokenize("hello world")
        self.assertEqual(tokens, ["hello", "world"])

    def test_chinese_characters(self):
        """Chinese characters are tokenized individually."""
        tokens = tokenize("你好世界")
        self.assertEqual(tokens, ["你", "好", "世", "界"])

    def test_mixed_text(self):
        """Mixed English and Chinese is handled."""
        tokens = tokenize("hello你好")
        self.assertIn("hello", tokens)
        self.assertIn("你", tokens)

    def test_empty_string(self):
        """Empty string returns empty list."""
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])

    def test_punctuation_removed(self):
        """Punctuation is not included in tokens."""
        tokens = tokenize("hello, world!")
        self.assertNotIn(",", tokens)
        self.assertNotIn("!", tokens)


class KeywordMatchScoreTest(TestCase):
    """Test keyword match scoring."""

    def test_exact_match(self):
        """Exact match has high score."""
        score = keyword_match_score("hello world", "hello world")
        self.assertGreater(score, 0.5)

    def test_partial_match(self):
        """Partial match has moderate score."""
        score = keyword_match_score("hello", "hello world")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_no_match(self):
        """No match has zero score."""
        score = keyword_match_score("hello", "goodbye")
        self.assertEqual(score, 0.0)

    def test_empty_query(self):
        """Empty query returns zero."""
        self.assertEqual(keyword_match_score("", "hello"), 0.0)

    def test_empty_text(self):
        """Empty text returns zero."""
        self.assertEqual(keyword_match_score("hello", ""), 0.0)

    def test_chinese_match(self):
        """Chinese text matching works."""
        score = keyword_match_score("你好", "你好世界")
        self.assertGreater(score, 0.0)


class EmbeddingSearchTest(TestCase):
    """Test embedding search function."""

    def setUp(self):
        """Set up test embeddings."""
        self.embeddings = [
            {
                'id': '1',
                'paragraph_id': 'p1',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'embedding': [1.0, 0.0, 0.0],
                'is_active': True,
            },
            {
                'id': '2',
                'paragraph_id': 'p2',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'embedding': [0.9, 0.1, 0.0],
                'is_active': True,
            },
            {
                'id': '3',
                'paragraph_id': 'p3',
                'document_id': 'd2',
                'knowledge_id': 'k1',
                'embedding': [0.0, 1.0, 0.0],
                'is_active': True,
            },
            {
                'id': '4',
                'paragraph_id': 'p4',
                'document_id': 'd2',
                'knowledge_id': 'k1',
                'embedding': [0.0, 0.0, 1.0],
                'is_active': False,
            },
        ]

    def test_finds_most_similar(self):
        """Most similar embedding is found first."""
        query = [1.0, 0.0, 0.0]
        results = embedding_search(query, self.embeddings, top_n=3)
        self.assertEqual(results[0]['paragraph_id'], 'p1')

    def test_returns_top_n(self):
        """Returns at most top_n results."""
        query = [1.0, 0.0, 0.0]
        results = embedding_search(query, self.embeddings, top_n=2)
        self.assertEqual(len(results), 2)

    def test_respects_threshold(self):
        """Respects similarity threshold."""
        query = [1.0, 0.0, 0.0]
        results = embedding_search(query, self.embeddings, top_n=10, similarity_threshold=0.999)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['paragraph_id'], 'p1')

    def test_excludes_inactive(self):
        """Excludes inactive embeddings."""
        query = [0.0, 0.0, 1.0]
        results = embedding_search(query, self.embeddings, top_n=10)
        paragraph_ids = [r['paragraph_id'] for r in results]
        self.assertNotIn('p4', paragraph_ids)

    def test_empty_embeddings(self):
        """Empty embeddings returns empty list."""
        results = embedding_search([1.0, 0.0], [], top_n=10)
        self.assertEqual(results, [])

    def test_empty_query(self):
        """Empty query returns empty list."""
        results = embedding_search([], self.embeddings, top_n=10)
        self.assertEqual(results, [])

    def test_includes_similarity_score(self):
        """Results include similarity score."""
        query = [1.0, 0.0, 0.0]
        results = embedding_search(query, self.embeddings, top_n=1)
        self.assertIn('similarity', results[0])
        self.assertGreater(results[0]['similarity'], 0.0)

    def test_deduplicates_by_paragraph(self):
        """Deduplicates results by paragraph ID."""
        embeddings = [
            {'id': '1', 'paragraph_id': 'p1', 'embedding': [1.0, 0.0], 'is_active': True},
            {'id': '2', 'paragraph_id': 'p1', 'embedding': [0.9, 0.1], 'is_active': True},
        ]
        results = embedding_search([1.0, 0.0], embeddings, top_n=10)
        self.assertEqual(len(results), 1)


class KeywordSearchTest(TestCase):
    """Test keyword search function."""

    def setUp(self):
        """Set up test paragraphs."""
        self.paragraphs = [
            {
                'id': 'p1',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'content': 'Python is a programming language',
                'title': 'Python Introduction',
                'is_active': True,
            },
            {
                'id': 'p2',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'content': 'Java is also a programming language',
                'title': 'Java Introduction',
                'is_active': True,
            },
            {
                'id': 'p3',
                'document_id': 'd2',
                'knowledge_id': 'k1',
                'content': 'Machine learning uses Python',
                'title': 'ML Basics',
                'is_active': True,
            },
            {
                'id': 'p4',
                'document_id': 'd2',
                'knowledge_id': 'k1',
                'content': 'This is inactive content',
                'title': 'Inactive',
                'is_active': False,
            },
        ]

    def test_finds_matching_content(self):
        """Finds paragraphs matching query."""
        results = keyword_search("Python", self.paragraphs, top_n=10)
        self.assertTrue(len(results) > 0)
        for result in results:
            self.assertIn(result['id'], ['p1', 'p3'])

    def test_returns_top_n(self):
        """Returns at most top_n results."""
        results = keyword_search("programming", self.paragraphs, top_n=1)
        self.assertEqual(len(results), 1)

    def test_excludes_inactive(self):
        """Excludes inactive paragraphs."""
        results = keyword_search("inactive", self.paragraphs, top_n=10)
        paragraph_ids = [r['id'] for r in results]
        self.assertNotIn('p4', paragraph_ids)

    def test_empty_query(self):
        """Empty query returns empty list."""
        results = keyword_search("", self.paragraphs, top_n=10)
        self.assertEqual(results, [])

    def test_no_matches(self):
        """No matches returns empty list."""
        results = keyword_search("xyz123", self.paragraphs, top_n=10)
        self.assertEqual(results, [])

    def test_includes_similarity_score(self):
        """Results include similarity score."""
        results = keyword_search("Python", self.paragraphs, top_n=1)
        self.assertIn('similarity', results[0])
        self.assertGreater(results[0]['similarity'], 0.0)

    def test_searches_title_and_content(self):
        """Searches both title and content."""
        results = keyword_search("Introduction", self.paragraphs, top_n=10)
        self.assertTrue(len(results) >= 2)


class BlendSearchTest(TestCase):
    """Test blend search function."""

    def setUp(self):
        """Set up test data."""
        self.embeddings = [
            {
                'id': '1',
                'paragraph_id': 'p1',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'embedding': [1.0, 0.0],
                'is_active': True,
            },
            {
                'id': '2',
                'paragraph_id': 'p2',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'embedding': [0.5, 0.5],
                'is_active': True,
            },
        ]
        self.paragraphs = [
            {
                'id': 'p1',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'content': 'Python programming',
                'title': 'Python',
                'is_active': True,
            },
            {
                'id': 'p2',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'content': 'Java programming',
                'title': 'Java',
                'is_active': True,
            },
        ]

    def test_combines_scores(self):
        """Combines embedding and keyword scores."""
        results = blend_search(
            "Python", [1.0, 0.0],
            self.embeddings, self.paragraphs,
            top_n=10
        )
        self.assertTrue(len(results) > 0)
        self.assertIn('similarity', results[0])

    def test_includes_both_scores(self):
        """Results include both embedding and keyword scores."""
        results = blend_search(
            "Python", [1.0, 0.0],
            self.embeddings, self.paragraphs,
            top_n=10
        )
        self.assertIn('embedding_score', results[0])
        self.assertIn('keyword_score', results[0])

    def test_falls_back_to_keyword(self):
        """Falls back to keyword search without vector."""
        results = blend_search(
            "Python", None,
            [], self.paragraphs,
            top_n=10
        )
        self.assertTrue(len(results) > 0)

    def test_falls_back_to_embedding(self):
        """Falls back to embedding search without query."""
        results = blend_search(
            "", [1.0, 0.0],
            self.embeddings, [],
            top_n=10
        )
        self.assertTrue(len(results) > 0)


class SearchFunctionTest(TestCase):
    """Test unified search function."""

    def setUp(self):
        """Set up test data."""
        self.embeddings = [
            {
                'id': '1',
                'paragraph_id': 'p1',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'embedding': [1.0, 0.0],
                'is_active': True,
                'source_type': 'p',
            },
            {
                'id': '2',
                'paragraph_id': 'p2',
                'document_id': 'd2',
                'knowledge_id': 'k2',
                'embedding': [0.0, 1.0],
                'is_active': True,
                'source_type': 'p',
            },
        ]
        self.paragraphs = [
            {
                'id': 'p1',
                'document_id': 'd1',
                'knowledge_id': 'k1',
                'content': 'First paragraph content',
                'title': 'First',
                'is_active': True,
            },
            {
                'id': 'p2',
                'document_id': 'd2',
                'knowledge_id': 'k2',
                'content': 'Second paragraph content',
                'title': 'Second',
                'is_active': True,
            },
        ]

    def test_embedding_mode(self):
        """Embedding mode uses vector search."""
        results = search(
            "first", [1.0, 0.0],
            self.embeddings, self.paragraphs,
            mode=SearchMode.EMBEDDING
        )
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)

    def test_keyword_mode(self):
        """Keyword mode uses text search."""
        results = search(
            "first", None,
            self.embeddings, self.paragraphs,
            mode=SearchMode.KEYWORDS
        )
        self.assertTrue(len(results) > 0)

    def test_blend_mode(self):
        """Blend mode combines both."""
        results = search(
            "first", [1.0, 0.0],
            self.embeddings, self.paragraphs,
            mode=SearchMode.BLEND
        )
        self.assertTrue(len(results) > 0)

    def test_filter_by_knowledge_id(self):
        """Filters by knowledge ID."""
        results = search(
            "first", [1.0, 0.0],
            self.embeddings, self.paragraphs,
            mode=SearchMode.EMBEDDING,
            knowledge_ids=['k1']
        )
        for r in results:
            self.assertEqual(r.knowledge_id, 'k1')

    def test_filter_by_document_id(self):
        """Filters by document ID."""
        results = search(
            "first", [1.0, 0.0],
            self.embeddings, self.paragraphs,
            mode=SearchMode.EMBEDDING,
            document_ids=['d1']
        )
        for r in results:
            self.assertEqual(r.document_id, 'd1')

    def test_exclude_paragraph_ids(self):
        """Excludes specified paragraphs."""
        results = search(
            "content", None,
            self.embeddings, self.paragraphs,
            mode=SearchMode.KEYWORDS,
            exclude_paragraph_ids=['p1']
        )
        for r in results:
            self.assertNotEqual(r.paragraph_id, 'p1')

    def test_search_result_to_dict(self):
        """SearchResult can convert to dict."""
        result = SearchResult(
            paragraph_id='p1',
            document_id='d1',
            knowledge_id='k1',
            content='Test content',
            title='Test',
            similarity=0.9
        )
        d = result.to_dict()
        self.assertEqual(d['paragraph_id'], 'p1')
        self.assertEqual(d['similarity'], 0.9)

    def test_empty_results_with_no_vector(self):
        """Embedding mode with no vector returns empty."""
        results = search(
            "first", None,
            self.embeddings, self.paragraphs,
            mode=SearchMode.EMBEDDING
        )
        self.assertEqual(results, [])


class SearchModeTest(TestCase):
    """Test SearchMode enum."""

    def test_search_mode_values(self):
        """SearchMode has correct values."""
        self.assertEqual(SearchMode.EMBEDDING.value, 'embedding')
        self.assertEqual(SearchMode.KEYWORDS.value, 'keywords')
        self.assertEqual(SearchMode.BLEND.value, 'blend')
