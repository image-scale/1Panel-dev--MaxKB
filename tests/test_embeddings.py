"""
Tests for the embedding service.
"""
import pytest
import math

from knowledgebot.core.embeddings import (
    EmbeddingProvider,
    SimpleEmbedding,
    normalize_embedding,
    cosine_similarity,
    euclidean_distance,
    get_embedding_provider,
    set_embedding_provider,
    reset_embedding_provider,
)


class TestSimpleEmbedding:
    """Tests for SimpleEmbedding provider."""

    def test_creation_default_dimension(self):
        """SimpleEmbedding should have default dimension."""
        provider = SimpleEmbedding()
        assert provider.dimension == 128

    def test_creation_custom_dimension(self):
        """SimpleEmbedding should accept custom dimension."""
        provider = SimpleEmbedding(dimension=256)
        assert provider.dimension == 256

    def test_invalid_dimension(self):
        """SimpleEmbedding should reject non-positive dimension."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            SimpleEmbedding(dimension=0)
        with pytest.raises(ValueError, match="dimension must be positive"):
            SimpleEmbedding(dimension=-1)

    def test_embed_text_returns_list_of_floats(self):
        """embed_text should return a list of floats."""
        provider = SimpleEmbedding(dimension=64)
        result = provider.embed_text("hello")
        assert isinstance(result, list)
        assert len(result) == 64
        assert all(isinstance(x, float) for x in result)

    def test_embed_text_correct_dimension(self):
        """embed_text should return vector of correct dimension."""
        for dim in [32, 64, 128, 256]:
            provider = SimpleEmbedding(dimension=dim)
            result = provider.embed_text("test text")
            assert len(result) == dim

    def test_embed_text_deterministic(self):
        """Same text should produce same embedding."""
        provider = SimpleEmbedding()
        result1 = provider.embed_text("hello world")
        result2 = provider.embed_text("hello world")
        assert result1 == result2

    def test_embed_text_different_texts(self):
        """Different texts should produce different embeddings."""
        provider = SimpleEmbedding()
        result1 = provider.embed_text("hello")
        result2 = provider.embed_text("world")
        assert result1 != result2

    def test_embed_text_normalized(self):
        """Embeddings should be normalized to unit length."""
        provider = SimpleEmbedding()
        result = provider.embed_text("test")
        # Check L2 norm is approximately 1
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    def test_embed_text_empty(self):
        """Empty text should return zero vector."""
        provider = SimpleEmbedding(dimension=64)
        result = provider.embed_text("")
        assert len(result) == 64
        assert all(x == 0.0 for x in result)

    def test_embed_text_whitespace_only(self):
        """Whitespace-only text should return zero vector."""
        provider = SimpleEmbedding(dimension=64)
        result = provider.embed_text("   \n\t  ")
        assert all(x == 0.0 for x in result)

    def test_embed_texts_multiple(self):
        """embed_texts should return list of embeddings."""
        provider = SimpleEmbedding(dimension=64)
        texts = ["hello", "world", "test"]
        results = provider.embed_texts(texts)
        assert len(results) == 3
        assert all(len(emb) == 64 for emb in results)

    def test_embed_texts_empty_list(self):
        """embed_texts with empty list should return empty list."""
        provider = SimpleEmbedding()
        results = provider.embed_texts([])
        assert results == []

    def test_embed_texts_matches_embed_text(self):
        """embed_texts results should match individual embed_text calls."""
        provider = SimpleEmbedding()
        texts = ["hello", "world"]
        batch_results = provider.embed_texts(texts)
        individual_results = [provider.embed_text(t) for t in texts]
        assert batch_results == individual_results


class TestNormalizeEmbedding:
    """Tests for normalize_embedding function."""

    def test_normalize_unit_vector(self):
        """Normalizing a unit vector should return same vector."""
        # A unit vector [1, 0, 0]
        vector = [1.0, 0.0, 0.0]
        result = normalize_embedding(vector)
        assert result == [1.0, 0.0, 0.0]

    def test_normalize_non_unit_vector(self):
        """Normalizing should produce unit length vector."""
        vector = [3.0, 4.0]  # Length = 5
        result = normalize_embedding(vector)
        assert abs(result[0] - 0.6) < 1e-6
        assert abs(result[1] - 0.8) < 1e-6
        # Check unit length
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """Normalizing zero vector should return zero vector."""
        vector = [0.0, 0.0, 0.0]
        result = normalize_embedding(vector)
        assert result == [0.0, 0.0, 0.0]

    def test_normalize_empty_vector(self):
        """Normalizing empty vector should return empty vector."""
        result = normalize_embedding([])
        assert result == []


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(v, v)
        assert abs(similarity - 1.0) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        similarity = cosine_similarity(v1, v2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        similarity = cosine_similarity(v1, v2)
        assert abs(similarity) < 1e-6

    def test_different_dimensions(self):
        """Vectors with different dimensions should raise error."""
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="same dimension"):
            cosine_similarity(v1, v2)

    def test_zero_vector(self):
        """Zero vector should return similarity 0."""
        v1 = [1.0, 2.0]
        v2 = [0.0, 0.0]
        similarity = cosine_similarity(v1, v2)
        assert similarity == 0.0

    def test_empty_vectors(self):
        """Empty vectors should return 0."""
        similarity = cosine_similarity([], [])
        assert similarity == 0.0


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        v = [1.0, 2.0, 3.0]
        distance = euclidean_distance(v, v)
        assert distance == 0.0

    def test_simple_distance(self):
        """Should compute correct Euclidean distance."""
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        distance = euclidean_distance(v1, v2)
        assert abs(distance - 5.0) < 1e-6

    def test_different_dimensions(self):
        """Vectors with different dimensions should raise error."""
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="same dimension"):
            euclidean_distance(v1, v2)

    def test_empty_vectors(self):
        """Empty vectors should return 0."""
        distance = euclidean_distance([], [])
        assert distance == 0.0


class TestGlobalProvider:
    """Tests for global embedding provider management."""

    def setup_method(self):
        """Reset provider before each test."""
        reset_embedding_provider()

    def test_get_provider_default(self):
        """get_embedding_provider should return default SimpleEmbedding."""
        provider = get_embedding_provider()
        assert isinstance(provider, SimpleEmbedding)
        assert provider.dimension == 128

    def test_set_custom_provider(self):
        """set_embedding_provider should set custom provider."""
        custom = SimpleEmbedding(dimension=256)
        set_embedding_provider(custom)
        provider = get_embedding_provider()
        assert provider is custom
        assert provider.dimension == 256

    def test_reset_provider(self):
        """reset_embedding_provider should clear the provider."""
        custom = SimpleEmbedding(dimension=256)
        set_embedding_provider(custom)
        reset_embedding_provider()
        # Getting provider again should create new default
        provider = get_embedding_provider()
        assert provider.dimension == 128


class TestEmbeddingProviderInterface:
    """Tests to verify EmbeddingProvider is a proper abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract EmbeddingProvider."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_simple_embedding_is_provider(self):
        """SimpleEmbedding should be instance of EmbeddingProvider."""
        provider = SimpleEmbedding()
        assert isinstance(provider, EmbeddingProvider)
