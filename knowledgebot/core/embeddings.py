"""
Embedding service for the KnowledgeBot platform.

Provides interfaces and implementations for creating vector embeddings from text.
"""
import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Embedding providers convert text into vector representations
    that can be used for semantic similarity search.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embedding vectors produced by this provider."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Create an embedding vector for a single text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass


class SimpleEmbedding(EmbeddingProvider):
    """
    Simple hash-based embedding provider for testing and development.

    This provider creates deterministic embeddings based on text hashes.
    It's not suitable for production use but provides consistent
    embeddings for testing the retrieval pipeline.
    """

    def __init__(self, dimension: int = 128):
        """
        Initialize the simple embedding provider.

        Args:
            dimension: The dimension of embedding vectors to produce
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def _hash_to_floats(self, text: str) -> List[float]:
        """Convert text to a list of floats via hashing."""
        # Create multiple hashes to fill the dimension
        floats = []
        seed = 0

        while len(floats) < self._dimension:
            # Create a hash with the current seed
            hash_input = f"{seed}:{text}".encode('utf-8')
            hash_bytes = hashlib.sha256(hash_input).digest()

            # Convert bytes to floats in range [-1, 1]
            for i in range(0, len(hash_bytes), 4):
                if len(floats) >= self._dimension:
                    break
                # Convert 4 bytes to a float
                value = int.from_bytes(hash_bytes[i:i+4], 'big')
                # Normalize to [-1, 1]
                normalized = (value / (2**32)) * 2 - 1
                floats.append(normalized)

            seed += 1

        return floats[:self._dimension]

    def embed_text(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimension

        # Normalize the text
        text = text.lower().strip()

        # Create hash-based embedding
        embedding = self._hash_to_floats(text)

        # Normalize to unit length
        return normalize_embedding(embedding)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]


def normalize_embedding(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length (L2 normalization).

    Args:
        vector: The vector to normalize

    Returns:
        The normalized vector with unit length, or zero vector if input is zero
    """
    if not vector:
        return vector

    # Calculate L2 norm
    norm = math.sqrt(sum(x * x for x in vector))

    if norm == 0:
        return vector

    # Normalize
    return [x / norm for x in vector]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in range [-1, 1], where 1 means identical direction

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectors must have same dimension: {len(v1)} != {len(v2)}")

    if not v1:
        return 0.0

    # Compute dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))

    # Compute magnitudes
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Euclidean distance (non-negative)

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vectors must have same dimension: {len(v1)} != {len(v2)}")

    if not v1:
        return 0.0

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# Global embedding provider instance
_embedding_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the configured embedding provider.

    Returns:
        The global embedding provider instance

    Note:
        If no provider has been set, creates a default SimpleEmbedding provider.
    """
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = SimpleEmbedding()
    return _embedding_provider


def set_embedding_provider(provider: EmbeddingProvider) -> None:
    """
    Set the global embedding provider.

    Args:
        provider: The embedding provider to use
    """
    global _embedding_provider
    _embedding_provider = provider


def reset_embedding_provider() -> None:
    """Reset the global embedding provider to None."""
    global _embedding_provider
    _embedding_provider = None
