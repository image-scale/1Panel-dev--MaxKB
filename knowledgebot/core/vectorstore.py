"""
Vector store for storing and querying embeddings.

Provides in-memory storage and similarity search for vector embeddings.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from knowledgebot.core.embeddings import cosine_similarity


@dataclass
class VectorEntry:
    """
    Represents a stored vector with metadata.

    Attributes:
        id: Unique identifier for this entry
        vector: The embedding vector
        text: Original text that was embedded
        metadata: Additional metadata (document_id, chunk_index, etc.)
        knowledge_base_id: ID of the knowledge base this belongs to
        created_at: When this entry was created
    """
    vector: List[float]
    text: str
    knowledge_base_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding vector for brevity)."""
        return {
            "id": self.id,
            "text": self.text,
            "knowledge_base_id": self.knowledge_base_id,
            "metadata": self.metadata.copy(),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchResult:
    """
    Result from a similarity search.

    Attributes:
        entry: The matched vector entry
        score: Similarity score (higher is more similar)
    """
    entry: VectorEntry
    score: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.entry.id,
            "text": self.entry.text,
            "score": self.score,
            "metadata": self.entry.metadata.copy(),
        }


class VectorStore:
    """
    In-memory vector store for similarity search.

    Stores vectors and supports nearest neighbor search using cosine similarity.
    """

    def __init__(self):
        self._entries: Dict[str, VectorEntry] = {}
        self._kb_index: Dict[str, set] = {}  # knowledge_base_id -> set of entry_ids

    def add(self, entry: VectorEntry) -> VectorEntry:
        """
        Add a vector entry to the store.

        Args:
            entry: The vector entry to add

        Returns:
            The added entry
        """
        self._entries[entry.id] = entry

        # Update knowledge base index
        if entry.knowledge_base_id not in self._kb_index:
            self._kb_index[entry.knowledge_base_id] = set()
        self._kb_index[entry.knowledge_base_id].add(entry.id)

        return entry

    def add_vector(
        self,
        vector: List[float],
        text: str,
        knowledge_base_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VectorEntry:
        """
        Add a vector to the store.

        Args:
            vector: The embedding vector
            text: Original text
            knowledge_base_id: Knowledge base ID
            metadata: Optional metadata

        Returns:
            The created VectorEntry
        """
        entry = VectorEntry(
            vector=vector,
            text=text,
            knowledge_base_id=knowledge_base_id,
            metadata=metadata or {},
        )
        return self.add(entry)

    def get(self, entry_id: str) -> Optional[VectorEntry]:
        """Get a vector entry by ID."""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """
        Delete a vector entry.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        entry = self._entries.get(entry_id)
        if entry:
            del self._entries[entry_id]
            if entry.knowledge_base_id in self._kb_index:
                self._kb_index[entry.knowledge_base_id].discard(entry_id)
            return True
        return False

    def delete_by_knowledge_base(self, knowledge_base_id: str) -> int:
        """
        Delete all vectors for a knowledge base.

        Args:
            knowledge_base_id: The knowledge base ID

        Returns:
            Number of entries deleted
        """
        entry_ids = list(self._kb_index.get(knowledge_base_id, set()))
        count = 0
        for entry_id in entry_ids:
            if self.delete(entry_id):
                count += 1
        return count

    def delete_by_metadata(
        self,
        knowledge_base_id: str,
        key: str,
        value: Any
    ) -> int:
        """
        Delete entries matching metadata.

        Args:
            knowledge_base_id: Knowledge base to search in
            key: Metadata key to match
            value: Metadata value to match

        Returns:
            Number of entries deleted
        """
        entry_ids = list(self._kb_index.get(knowledge_base_id, set()))
        count = 0
        for entry_id in entry_ids:
            entry = self._entries.get(entry_id)
            if entry and entry.metadata.get(key) == value:
                if self.delete(entry_id):
                    count += 1
        return count

    def search(
        self,
        query_vector: List[float],
        knowledge_base_ids: List[str],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query embedding vector
            knowledge_base_ids: Knowledge bases to search in
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult sorted by score (highest first)
        """
        results = []

        # Collect entries from specified knowledge bases
        for kb_id in knowledge_base_ids:
            entry_ids = self._kb_index.get(kb_id, set())
            for entry_id in entry_ids:
                entry = self._entries.get(entry_id)
                if entry and entry.is_active:
                    # Compute similarity
                    score = cosine_similarity(query_vector, entry.vector)
                    if score >= min_score:
                        results.append(SearchResult(entry=entry, score=score))

        # Sort by score (descending) and take top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def count(self, knowledge_base_id: Optional[str] = None) -> int:
        """
        Count entries in the store.

        Args:
            knowledge_base_id: If provided, count only for this KB

        Returns:
            Number of entries
        """
        if knowledge_base_id:
            return len(self._kb_index.get(knowledge_base_id, set()))
        return len(self._entries)

    def list_by_knowledge_base(self, knowledge_base_id: str) -> List[VectorEntry]:
        """List all entries for a knowledge base."""
        entry_ids = self._kb_index.get(knowledge_base_id, set())
        return [
            self._entries[eid]
            for eid in entry_ids
            if eid in self._entries
        ]

    def clear(self) -> None:
        """Clear all entries from the store."""
        self._entries.clear()
        self._kb_index.clear()


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def reset_vector_store() -> None:
    """Reset the global vector store."""
    global _vector_store
    _vector_store = None
