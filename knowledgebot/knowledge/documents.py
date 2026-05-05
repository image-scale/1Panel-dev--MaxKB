"""
Document models and storage for the KnowledgeBot platform.
"""
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List


class DocumentStatus(Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class FileType(Enum):
    """Supported file types."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"


@dataclass
class Document:
    """
    Document model representing a document within a knowledge base.

    Attributes:
        id: Unique identifier for the document
        knowledge_base_id: ID of the knowledge base this document belongs to
        name: Name/title of the document
        content: Full text content of the document
        file_type: Type of the original file
        char_length: Length of the content in characters
        status: Processing status of the document
        meta: Additional metadata
        created_at: Timestamp when the document was created
        updated_at: Timestamp when the document was last updated
    """
    name: str
    knowledge_base_id: str
    content: str = ""
    file_type: FileType = FileType.TEXT
    status: DocumentStatus = DocumentStatus.PENDING
    meta: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def char_length(self) -> int:
        """Return the character length of the content."""
        return len(self.content)

    def to_dict(self) -> dict:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "knowledge_base_id": self.knowledge_base_id,
            "file_type": self.file_type.value,
            "char_length": self.char_length,
            "status": self.status.value,
            "meta": self.meta.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class DocumentStore:
    """
    In-memory storage for documents.
    In production, this would be backed by a database.
    """

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._kb_index: Dict[str, set] = {}  # kb_id -> set of doc_ids

    def add(self, doc: Document) -> Document:
        """Add a document to the store."""
        self._documents[doc.id] = doc

        # Update knowledge base index
        if doc.knowledge_base_id not in self._kb_index:
            self._kb_index[doc.knowledge_base_id] = set()
        self._kb_index[doc.knowledge_base_id].add(doc.id)

        return doc

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Get a document by its ID."""
        return self._documents.get(doc_id)

    def list_by_knowledge_base(self, kb_id: str) -> List[Document]:
        """List all documents for a knowledge base."""
        doc_ids = self._kb_index.get(kb_id, set())
        return [self._documents[doc_id] for doc_id in doc_ids if doc_id in self._documents]

    def update(self, doc: Document) -> Document:
        """Update a document in the store."""
        doc.updated_at = datetime.utcnow()
        self._documents[doc.id] = doc
        return doc

    def delete(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        doc = self._documents.get(doc_id)
        if doc:
            # Remove from main store
            del self._documents[doc_id]

            # Remove from knowledge base index
            if doc.knowledge_base_id in self._kb_index:
                self._kb_index[doc.knowledge_base_id].discard(doc_id)

            return True
        return False

    def delete_by_knowledge_base(self, kb_id: str) -> int:
        """Delete all documents for a knowledge base. Returns count deleted."""
        doc_ids = list(self._kb_index.get(kb_id, set()))
        count = 0
        for doc_id in doc_ids:
            if self.delete(doc_id):
                count += 1
        return count

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._documents.clear()
        self._kb_index.clear()


# Global document store instance
_doc_store = DocumentStore()


def get_doc_store() -> DocumentStore:
    """Get the global document store instance."""
    return _doc_store


# Document service functions

def create_document(
    knowledge_base_id: str,
    name: str,
    content: str,
    file_type: FileType = FileType.TEXT,
    meta: Optional[Dict] = None
) -> Document:
    """
    Create a new document in a knowledge base.

    Args:
        knowledge_base_id: ID of the knowledge base
        name: Name of the document
        content: Text content of the document
        file_type: Type of the file
        meta: Optional metadata

    Returns:
        The newly created Document object

    Raises:
        ValueError: If the knowledge base does not exist
    """
    from knowledgebot.knowledge.service import get_knowledge_base

    kb = get_knowledge_base(knowledge_base_id)
    if kb is None:
        raise ValueError(f"Knowledge base '{knowledge_base_id}' does not exist")

    doc = Document(
        name=name,
        knowledge_base_id=knowledge_base_id,
        content=content,
        file_type=file_type,
        meta=meta or {},
    )

    store = get_doc_store()
    return store.add(doc)


def get_document(doc_id: str) -> Optional[Document]:
    """
    Get a document by its ID.

    Args:
        doc_id: The ID of the document

    Returns:
        The Document object if found, None otherwise
    """
    store = get_doc_store()
    return store.get_by_id(doc_id)


def list_documents(knowledge_base_id: str) -> List[Document]:
    """
    List all documents for a knowledge base.

    Args:
        knowledge_base_id: The ID of the knowledge base

    Returns:
        List of Document objects in the knowledge base
    """
    store = get_doc_store()
    return store.list_by_knowledge_base(knowledge_base_id)


def update_document(
    doc_id: str,
    name: Optional[str] = None,
    content: Optional[str] = None,
    status: Optional[DocumentStatus] = None,
    meta: Optional[Dict] = None
) -> Optional[Document]:
    """
    Update a document.

    Args:
        doc_id: The ID of the document to update
        name: New name (optional)
        content: New content (optional)
        status: New status (optional)
        meta: Metadata to merge (optional)

    Returns:
        The updated Document object, or None if not found
    """
    store = get_doc_store()
    doc = store.get_by_id(doc_id)

    if doc is None:
        return None

    if name is not None:
        doc.name = name

    if content is not None:
        doc.content = content

    if status is not None:
        doc.status = status

    if meta is not None:
        doc.meta.update(meta)

    return store.update(doc)


def delete_document(doc_id: str) -> bool:
    """
    Delete a document.

    Args:
        doc_id: The ID of the document to delete

    Returns:
        True if the document was deleted, False if not found
    """
    store = get_doc_store()
    return store.delete(doc_id)


def get_document_content(doc_id: str) -> Optional[str]:
    """
    Get the content of a document.

    Args:
        doc_id: The ID of the document

    Returns:
        The document content if found, None otherwise
    """
    doc = get_document(doc_id)
    return doc.content if doc else None
