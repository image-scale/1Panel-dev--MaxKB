"""
Tests for the document management system.
"""
import pytest

from knowledgebot.knowledge.models import get_kb_store
from knowledgebot.knowledge.service import create_knowledge_base
from knowledgebot.knowledge.documents import (
    Document,
    DocumentStatus,
    FileType,
    DocumentStore,
    get_doc_store,
    create_document,
    get_document,
    list_documents,
    update_document,
    delete_document,
    get_document_content,
)


class TestDocumentStatus:
    """Tests for DocumentStatus enum."""

    def test_document_status_pending(self):
        """DocumentStatus should have PENDING value."""
        assert DocumentStatus.PENDING.value == "pending"

    def test_document_status_processing(self):
        """DocumentStatus should have PROCESSING value."""
        assert DocumentStatus.PROCESSING.value == "processing"

    def test_document_status_completed(self):
        """DocumentStatus should have COMPLETED value."""
        assert DocumentStatus.COMPLETED.value == "completed"

    def test_document_status_error(self):
        """DocumentStatus should have ERROR value."""
        assert DocumentStatus.ERROR.value == "error"


class TestFileType:
    """Tests for FileType enum."""

    def test_file_type_text(self):
        """FileType should have TEXT value."""
        assert FileType.TEXT.value == "text"

    def test_file_type_pdf(self):
        """FileType should have PDF value."""
        assert FileType.PDF.value == "pdf"

    def test_file_type_markdown(self):
        """FileType should have MARKDOWN value."""
        assert FileType.MARKDOWN.value == "markdown"


class TestDocumentModel:
    """Tests for the Document model."""

    def test_document_creation(self):
        """Document should be created with required fields."""
        doc = Document(name="test.txt", knowledge_base_id="kb123")
        assert doc.name == "test.txt"
        assert doc.knowledge_base_id == "kb123"
        assert doc.content == ""
        assert doc.file_type == FileType.TEXT
        assert doc.status == DocumentStatus.PENDING
        assert doc.id is not None
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_document_with_content(self):
        """Document should accept content."""
        content = "This is the document content."
        doc = Document(
            name="test.txt",
            knowledge_base_id="kb123",
            content=content
        )
        assert doc.content == content

    def test_document_char_length(self):
        """char_length should return content length."""
        content = "Hello, World!"
        doc = Document(
            name="test.txt",
            knowledge_base_id="kb123",
            content=content
        )
        assert doc.char_length == len(content)
        assert doc.char_length == 13

    def test_document_char_length_empty(self):
        """char_length should be 0 for empty content."""
        doc = Document(name="test.txt", knowledge_base_id="kb123")
        assert doc.char_length == 0

    def test_document_to_dict(self):
        """to_dict should return dictionary representation."""
        doc = Document(
            name="test.txt",
            knowledge_base_id="kb123",
            content="Test content",
            file_type=FileType.MARKDOWN,
            status=DocumentStatus.COMPLETED
        )
        data = doc.to_dict()
        assert data["id"] == doc.id
        assert data["name"] == "test.txt"
        assert data["knowledge_base_id"] == "kb123"
        assert data["file_type"] == "markdown"
        assert data["status"] == "completed"
        assert data["char_length"] == 12
        assert "content" not in data  # Content not included in dict
        assert "created_at" in data
        assert "updated_at" in data


class TestDocumentStore:
    """Tests for DocumentStore."""

    def setup_method(self):
        """Reset store before each test."""
        get_doc_store().clear()

    def test_add_and_get_by_id(self):
        """Should add and retrieve document by ID."""
        store = DocumentStore()
        doc = Document(name="test.txt", knowledge_base_id="kb123", content="Hello")
        store.add(doc)

        retrieved = store.get_by_id(doc.id)
        assert retrieved is not None
        assert retrieved.name == "test.txt"

    def test_get_by_id_not_found(self):
        """Should return None if document not found."""
        store = DocumentStore()
        retrieved = store.get_by_id("nonexistent")
        assert retrieved is None

    def test_list_by_knowledge_base(self):
        """Should list all documents for a knowledge base."""
        store = DocumentStore()
        doc1 = Document(name="doc1.txt", knowledge_base_id="kb123")
        doc2 = Document(name="doc2.txt", knowledge_base_id="kb123")
        doc3 = Document(name="doc3.txt", knowledge_base_id="kb456")
        store.add(doc1)
        store.add(doc2)
        store.add(doc3)

        kb123_docs = store.list_by_knowledge_base("kb123")
        assert len(kb123_docs) == 2
        names = {doc.name for doc in kb123_docs}
        assert names == {"doc1.txt", "doc2.txt"}

    def test_delete(self):
        """Should delete document and return True."""
        store = DocumentStore()
        doc = Document(name="test.txt", knowledge_base_id="kb123")
        store.add(doc)

        result = store.delete(doc.id)
        assert result is True
        assert store.get_by_id(doc.id) is None

    def test_delete_nonexistent(self):
        """Should return False when deleting nonexistent document."""
        store = DocumentStore()
        result = store.delete("nonexistent")
        assert result is False

    def test_delete_by_knowledge_base(self):
        """Should delete all documents for a knowledge base."""
        store = DocumentStore()
        doc1 = Document(name="doc1.txt", knowledge_base_id="kb123")
        doc2 = Document(name="doc2.txt", knowledge_base_id="kb123")
        doc3 = Document(name="doc3.txt", knowledge_base_id="kb456")
        store.add(doc1)
        store.add(doc2)
        store.add(doc3)

        count = store.delete_by_knowledge_base("kb123")
        assert count == 2
        assert len(store.list_by_knowledge_base("kb123")) == 0
        assert len(store.list_by_knowledge_base("kb456")) == 1


class TestCreateDocument:
    """Tests for create_document function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()

    def test_create_document_success(self):
        """create_document should create and return new document."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(
            knowledge_base_id=kb.id,
            name="test.txt",
            content="Test content"
        )
        assert doc is not None
        assert doc.name == "test.txt"
        assert doc.content == "Test content"
        assert doc.knowledge_base_id == kb.id

    def test_create_document_with_file_type(self):
        """create_document should accept file_type."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(
            knowledge_base_id=kb.id,
            name="test.md",
            content="# Test",
            file_type=FileType.MARKDOWN
        )
        assert doc.file_type == FileType.MARKDOWN

    def test_create_document_nonexistent_kb(self):
        """create_document should raise error for nonexistent knowledge base."""
        with pytest.raises(ValueError, match="does not exist"):
            create_document(
                knowledge_base_id="nonexistent",
                name="test.txt",
                content="Test"
            )

    def test_create_document_char_length_calculated(self):
        """char_length should be calculated from content."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        content = "This is a test document with some content."
        doc = create_document(
            knowledge_base_id=kb.id,
            name="test.txt",
            content=content
        )
        assert doc.char_length == len(content)


class TestGetDocument:
    """Tests for get_document function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()

    def test_get_document_exists(self):
        """get_document should return document if exists."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        created = create_document(kb.id, "test.txt", "Content")
        retrieved = get_document(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_document_not_found(self):
        """get_document should return None if not found."""
        result = get_document("nonexistent")
        assert result is None


class TestListDocuments:
    """Tests for list_documents function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()

    def test_list_documents_returns_kb_docs(self):
        """list_documents should return all documents for a KB."""
        kb1 = create_knowledge_base(name="KB 1", user_id="user123")
        kb2 = create_knowledge_base(name="KB 2", user_id="user123")
        create_document(kb1.id, "doc1.txt", "Content 1")
        create_document(kb1.id, "doc2.txt", "Content 2")
        create_document(kb2.id, "doc3.txt", "Content 3")

        docs = list_documents(kb1.id)
        assert len(docs) == 2
        names = {doc.name for doc in docs}
        assert names == {"doc1.txt", "doc2.txt"}

    def test_list_documents_empty(self):
        """list_documents should return empty list if no documents."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        docs = list_documents(kb.id)
        assert docs == []


class TestUpdateDocument:
    """Tests for update_document function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()

    def test_update_document_name(self):
        """update_document should update name."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(kb.id, "old.txt", "Content")
        updated = update_document(doc.id, name="new.txt")
        assert updated is not None
        assert updated.name == "new.txt"

    def test_update_document_status(self):
        """update_document should update status."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(kb.id, "test.txt", "Content")
        assert doc.status == DocumentStatus.PENDING

        updated = update_document(doc.id, status=DocumentStatus.COMPLETED)
        assert updated is not None
        assert updated.status == DocumentStatus.COMPLETED

    def test_update_document_content(self):
        """update_document should update content."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(kb.id, "test.txt", "Old content")
        updated = update_document(doc.id, content="New content")
        assert updated is not None
        assert updated.content == "New content"
        assert updated.char_length == len("New content")

    def test_update_document_not_found(self):
        """update_document should return None if not found."""
        result = update_document("nonexistent", name="new.txt")
        assert result is None


class TestDeleteDocument:
    """Tests for delete_document function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()

    def test_delete_document_success(self):
        """delete_document should delete and return True."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(kb.id, "test.txt", "Content")
        result = delete_document(doc.id)
        assert result is True
        assert get_document(doc.id) is None

    def test_delete_document_not_found(self):
        """delete_document should return False if not found."""
        result = delete_document("nonexistent")
        assert result is False


class TestGetDocumentContent:
    """Tests for get_document_content function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()

    def test_get_document_content_exists(self):
        """get_document_content should return content."""
        kb = create_knowledge_base(name="Test KB", user_id="user123")
        doc = create_document(kb.id, "test.txt", "Test content here")
        content = get_document_content(doc.id)
        assert content == "Test content here"

    def test_get_document_content_not_found(self):
        """get_document_content should return None if not found."""
        content = get_document_content("nonexistent")
        assert content is None
