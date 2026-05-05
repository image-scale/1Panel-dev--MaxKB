"""
Tests for the vector store.
"""
import pytest

from knowledgebot.core.vectorstore import (
    VectorEntry,
    SearchResult,
    VectorStore,
    get_vector_store,
    reset_vector_store,
)
from knowledgebot.core.embeddings import SimpleEmbedding


class TestVectorEntry:
    """Tests for VectorEntry."""

    def test_creation(self):
        """VectorEntry should be created with required fields."""
        entry = VectorEntry(
            vector=[0.1, 0.2, 0.3],
            text="test text",
            knowledge_base_id="kb123"
        )
        assert entry.vector == [0.1, 0.2, 0.3]
        assert entry.text == "test text"
        assert entry.knowledge_base_id == "kb123"
        assert entry.id is not None
        assert entry.is_active is True

    def test_with_metadata(self):
        """VectorEntry should accept metadata."""
        entry = VectorEntry(
            vector=[0.1, 0.2],
            text="test",
            knowledge_base_id="kb123",
            metadata={"doc_id": "doc1", "chunk_index": 0}
        )
        assert entry.metadata["doc_id"] == "doc1"
        assert entry.metadata["chunk_index"] == 0

    def test_to_dict(self):
        """to_dict should return dictionary representation."""
        entry = VectorEntry(
            vector=[0.1, 0.2],
            text="test",
            knowledge_base_id="kb123",
            metadata={"key": "value"}
        )
        data = entry.to_dict()
        assert data["id"] == entry.id
        assert data["text"] == "test"
        assert data["knowledge_base_id"] == "kb123"
        assert "vector" not in data  # Vector not included
        assert data["metadata"] == {"key": "value"}


class TestSearchResult:
    """Tests for SearchResult."""

    def test_creation(self):
        """SearchResult should store entry and score."""
        entry = VectorEntry(
            vector=[0.1, 0.2],
            text="test",
            knowledge_base_id="kb123"
        )
        result = SearchResult(entry=entry, score=0.95)
        assert result.entry is entry
        assert result.score == 0.95

    def test_to_dict(self):
        """to_dict should return dictionary representation."""
        entry = VectorEntry(
            vector=[0.1, 0.2],
            text="test",
            knowledge_base_id="kb123"
        )
        result = SearchResult(entry=entry, score=0.95)
        data = result.to_dict()
        assert data["id"] == entry.id
        assert data["text"] == "test"
        assert data["score"] == 0.95


class TestVectorStore:
    """Tests for VectorStore."""

    def setup_method(self):
        """Reset stores before each test."""
        reset_vector_store()

    def test_add_and_get(self):
        """Should add and retrieve entry."""
        store = VectorStore()
        entry = VectorEntry(
            vector=[0.1, 0.2],
            text="test",
            knowledge_base_id="kb123"
        )
        store.add(entry)

        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.text == "test"

    def test_add_vector(self):
        """add_vector should create and add entry."""
        store = VectorStore()
        entry = store.add_vector(
            vector=[0.1, 0.2],
            text="test text",
            knowledge_base_id="kb123",
            metadata={"key": "value"}
        )
        assert entry.text == "test text"
        assert store.get(entry.id) is not None

    def test_delete(self):
        """Should delete entry and return True."""
        store = VectorStore()
        entry = store.add_vector([0.1, 0.2], "test", "kb123")
        result = store.delete(entry.id)
        assert result is True
        assert store.get(entry.id) is None

    def test_delete_nonexistent(self):
        """Should return False when deleting nonexistent entry."""
        store = VectorStore()
        result = store.delete("nonexistent")
        assert result is False

    def test_delete_by_knowledge_base(self):
        """Should delete all entries for a knowledge base."""
        store = VectorStore()
        store.add_vector([0.1], "text1", "kb1")
        store.add_vector([0.2], "text2", "kb1")
        store.add_vector([0.3], "text3", "kb2")

        count = store.delete_by_knowledge_base("kb1")
        assert count == 2
        assert store.count("kb1") == 0
        assert store.count("kb2") == 1

    def test_delete_by_metadata(self):
        """Should delete entries matching metadata."""
        store = VectorStore()
        store.add_vector([0.1], "text1", "kb1", {"doc_id": "doc1"})
        store.add_vector([0.2], "text2", "kb1", {"doc_id": "doc1"})
        store.add_vector([0.3], "text3", "kb1", {"doc_id": "doc2"})

        count = store.delete_by_metadata("kb1", "doc_id", "doc1")
        assert count == 2
        assert store.count("kb1") == 1

    def test_search_basic(self):
        """Should return similar vectors."""
        store = VectorStore()
        provider = SimpleEmbedding(dimension=64)

        # Add some entries
        vec1 = provider.embed_text("hello world")
        vec2 = provider.embed_text("hello there")
        vec3 = provider.embed_text("completely different")

        store.add_vector(vec1, "hello world", "kb1")
        store.add_vector(vec2, "hello there", "kb1")
        store.add_vector(vec3, "completely different", "kb1")

        # Search
        query = provider.embed_text("hello")
        results = store.search(query, ["kb1"], top_k=3)

        assert len(results) > 0
        # Results should be sorted by score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_top_k(self):
        """Should respect top_k limit."""
        store = VectorStore()
        for i in range(10):
            store.add_vector([float(i) / 10], f"text{i}", "kb1")

        results = store.search([0.5], ["kb1"], top_k=3)
        assert len(results) <= 3

    def test_search_min_score(self):
        """Should filter by min_score."""
        store = VectorStore()
        provider = SimpleEmbedding(dimension=64)

        vec1 = provider.embed_text("exact match text")
        vec2 = provider.embed_text("completely unrelated content")

        store.add_vector(vec1, "exact match text", "kb1")
        store.add_vector(vec2, "completely unrelated", "kb1")

        query = provider.embed_text("exact match text")
        results = store.search(query, ["kb1"], top_k=10, min_score=0.99)

        # Should only return very similar entries
        assert all(r.score >= 0.99 for r in results)

    def test_search_multiple_kbs(self):
        """Should search across multiple knowledge bases."""
        store = VectorStore()
        provider = SimpleEmbedding(dimension=64)

        vec1 = provider.embed_text("hello")
        vec2 = provider.embed_text("hello")

        store.add_vector(vec1, "hello kb1", "kb1")
        store.add_vector(vec2, "hello kb2", "kb2")

        query = provider.embed_text("hello")
        results = store.search(query, ["kb1", "kb2"], top_k=10)

        assert len(results) == 2

    def test_search_inactive_excluded(self):
        """Inactive entries should be excluded from search."""
        store = VectorStore()
        entry = store.add_vector([0.1, 0.2], "test", "kb1")
        entry.is_active = False

        results = store.search([0.1, 0.2], ["kb1"], top_k=10)
        assert len(results) == 0

    def test_count(self):
        """Should count entries correctly."""
        store = VectorStore()
        store.add_vector([0.1], "text1", "kb1")
        store.add_vector([0.2], "text2", "kb1")
        store.add_vector([0.3], "text3", "kb2")

        assert store.count() == 3
        assert store.count("kb1") == 2
        assert store.count("kb2") == 1
        assert store.count("kb3") == 0

    def test_list_by_knowledge_base(self):
        """Should list all entries for a knowledge base."""
        store = VectorStore()
        store.add_vector([0.1], "text1", "kb1")
        store.add_vector([0.2], "text2", "kb1")
        store.add_vector([0.3], "text3", "kb2")

        entries = store.list_by_knowledge_base("kb1")
        assert len(entries) == 2
        texts = {e.text for e in entries}
        assert texts == {"text1", "text2"}

    def test_clear(self):
        """Should clear all entries."""
        store = VectorStore()
        store.add_vector([0.1], "text1", "kb1")
        store.add_vector([0.2], "text2", "kb2")

        store.clear()
        assert store.count() == 0


class TestGlobalVectorStore:
    """Tests for global vector store management."""

    def setup_method(self):
        """Reset before each test."""
        reset_vector_store()

    def test_get_store_default(self):
        """get_vector_store should return a VectorStore."""
        store = get_vector_store()
        assert isinstance(store, VectorStore)

    def test_get_store_singleton(self):
        """get_vector_store should return same instance."""
        store1 = get_vector_store()
        store2 = get_vector_store()
        assert store1 is store2

    def test_reset_store(self):
        """reset_vector_store should clear the singleton."""
        store1 = get_vector_store()
        store1.add_vector([0.1], "test", "kb1")
        reset_vector_store()
        store2 = get_vector_store()
        assert store2 is not store1
        assert store2.count() == 0
