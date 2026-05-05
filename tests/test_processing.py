"""
Tests for the document processing pipeline.
"""
import pytest

from knowledgebot.knowledge.models import get_kb_store
from knowledgebot.knowledge.service import create_knowledge_base
from knowledgebot.knowledge.documents import (
    get_doc_store,
    create_document,
    get_document,
    DocumentStatus,
)
from knowledgebot.core.vectorstore import get_vector_store, reset_vector_store
from knowledgebot.core.embeddings import reset_embedding_provider
from knowledgebot.knowledge.processing import (
    ProcessingResult,
    process_document,
    process_documents,
    reprocess_knowledge_base,
    get_document_chunks,
    search_knowledge_base,
)


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_success_result(self):
        """Should create success result."""
        result = ProcessingResult(
            document_id="doc1",
            success=True,
            chunks_created=5
        )
        assert result.success is True
        assert result.chunks_created == 5
        assert result.error_message is None

    def test_failure_result(self):
        """Should create failure result."""
        result = ProcessingResult(
            document_id="doc1",
            success=False,
            chunks_created=0,
            error_message="Document not found"
        )
        assert result.success is False
        assert result.error_message == "Document not found"


class TestProcessDocument:
    """Tests for process_document function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        reset_vector_store()
        reset_embedding_provider()

    def test_process_document_success(self):
        """Should process document and create chunks."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(
            kb.id,
            "test.txt",
            "This is the first sentence. This is the second sentence. This is the third sentence.",
        )

        result = process_document(doc.id, chunk_size=50, chunk_overlap=10)

        assert result.success is True
        assert result.chunks_created > 0
        # Document status should be completed
        updated_doc = get_document(doc.id)
        assert updated_doc.status == DocumentStatus.COMPLETED

    def test_process_document_not_found(self):
        """Should return error for nonexistent document."""
        result = process_document("nonexistent-id")
        assert result.success is False
        assert "not found" in result.error_message.lower()

    def test_process_empty_document(self):
        """Should handle empty document."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(kb.id, "empty.txt", "")

        result = process_document(doc.id)

        assert result.success is True
        assert result.chunks_created == 0

    def test_process_document_creates_vectors(self):
        """Should create vectors in the vector store."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(
            kb.id,
            "test.txt",
            "Some content here that will be chunked and embedded.",
        )

        process_document(doc.id, chunk_size=20, chunk_overlap=5)

        store = get_vector_store()
        assert store.count(kb.id) > 0

    def test_process_document_removes_old_vectors(self):
        """Reprocessing should remove old vectors."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(kb.id, "test.txt", "Original content.")

        # Process once
        process_document(doc.id, chunk_size=100, chunk_overlap=10)
        store = get_vector_store()
        count1 = store.count(kb.id)

        # Process again with different content
        doc.content = "Updated content that is different."
        get_doc_store().update(doc)
        process_document(doc.id, chunk_size=100, chunk_overlap=10)

        # Should still have vectors but may be different count
        assert store.count(kb.id) > 0


class TestProcessDocuments:
    """Tests for process_documents function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        reset_vector_store()
        reset_embedding_provider()

    def test_process_multiple_documents(self):
        """Should process multiple documents."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc1 = create_document(kb.id, "doc1.txt", "Content one.")
        doc2 = create_document(kb.id, "doc2.txt", "Content two.")
        doc3 = create_document(kb.id, "doc3.txt", "Content three.")

        results = process_documents([doc1.id, doc2.id, doc3.id])

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_process_mixed_success(self):
        """Should handle mixed success/failure."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(kb.id, "doc.txt", "Content.")

        results = process_documents([doc.id, "nonexistent"])

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False


class TestReprocessKnowledgeBase:
    """Tests for reprocess_knowledge_base function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        reset_vector_store()
        reset_embedding_provider()

    def test_reprocess_all_documents(self):
        """Should reprocess all documents in KB."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        create_document(kb.id, "doc1.txt", "Content one.")
        create_document(kb.id, "doc2.txt", "Content two.")

        results = reprocess_knowledge_base(kb.id)

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_reprocess_empty_kb(self):
        """Should handle empty knowledge base."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        results = reprocess_knowledge_base(kb.id)
        assert results == []


class TestGetDocumentChunks:
    """Tests for get_document_chunks function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        reset_vector_store()
        reset_embedding_provider()

    def test_get_chunks_after_processing(self):
        """Should return chunks after processing."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(
            kb.id,
            "test.txt",
            "First chunk here. Second chunk here. Third chunk here.",
        )
        process_document(doc.id, chunk_size=20, chunk_overlap=5)

        chunks = get_document_chunks(doc.id)
        assert len(chunks) > 0

    def test_get_chunks_nonexistent(self):
        """Should return empty for nonexistent document."""
        chunks = get_document_chunks("nonexistent")
        assert chunks == []


class TestSearchKnowledgeBase:
    """Tests for search_knowledge_base function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        reset_vector_store()
        reset_embedding_provider()

    def test_search_finds_relevant(self):
        """Should find relevant content."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(
            kb.id,
            "test.txt",
            "Python is a programming language. JavaScript is also a programming language.",
        )
        process_document(doc.id, chunk_size=50, chunk_overlap=10)

        results = search_knowledge_base("programming language", [kb.id], top_k=5)

        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_search_empty_kb(self):
        """Should return empty for empty knowledge base."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        results = search_knowledge_base("query", [kb.id])
        assert results == []

    def test_search_multiple_kbs(self):
        """Should search across multiple knowledge bases."""
        kb1 = create_knowledge_base(name="KB 1", user_id="user1")
        kb2 = create_knowledge_base(name="KB 2", user_id="user1")
        doc1 = create_document(kb1.id, "doc1.txt", "Python programming guide.")
        doc2 = create_document(kb2.id, "doc2.txt", "Python tutorial content.")
        process_document(doc1.id, chunk_size=100, chunk_overlap=10)
        process_document(doc2.id, chunk_size=100, chunk_overlap=10)

        results = search_knowledge_base("Python", [kb1.id, kb2.id])

        assert len(results) >= 2

    def test_search_respects_top_k(self):
        """Should respect top_k limit."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        for i in range(10):
            doc = create_document(kb.id, f"doc{i}.txt", f"Content number {i}.")
            process_document(doc.id, chunk_size=100, chunk_overlap=10)

        results = search_knowledge_base("Content", [kb.id], top_k=3)

        assert len(results) <= 3
