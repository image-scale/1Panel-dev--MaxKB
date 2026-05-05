"""
Knowledge base models for the KnowledgeBot platform.
"""
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def default_kb_settings() -> Dict[str, Any]:
    """Return default settings for a knowledge base."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "similarity_threshold": 0.7,
        "top_k": 5,
        "search_mode": "embedding",  # embedding, keyword, or hybrid
    }


@dataclass
class KnowledgeBase:
    """
    Knowledge base model representing a collection of documents.

    Attributes:
        id: Unique identifier for the knowledge base
        name: Name of the knowledge base
        description: Description of the knowledge base
        user_id: ID of the user who owns this knowledge base
        settings: Configuration settings for the knowledge base
        created_at: Timestamp when the knowledge base was created
        updated_at: Timestamp when the knowledge base was last updated
    """
    name: str
    user_id: str
    description: str = ""
    settings: Dict[str, Any] = field(default_factory=default_kb_settings)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert knowledge base to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "settings": self.settings.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update knowledge base settings."""
        self.settings.update(new_settings)
        self.updated_at = datetime.utcnow()


class KnowledgeBaseStore:
    """
    In-memory storage for knowledge bases.
    In production, this would be backed by a database.
    """

    def __init__(self):
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}
        self._user_index: Dict[str, set] = {}  # user_id -> set of kb_ids
        self._name_index: Dict[str, str] = {}  # (user_id, name) key -> kb_id

    def _name_key(self, user_id: str, name: str) -> str:
        """Create a unique key for user_id + name combination."""
        return f"{user_id}:{name}"

    def add(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Add a knowledge base to the store."""
        self._knowledge_bases[kb.id] = kb

        # Update user index
        if kb.user_id not in self._user_index:
            self._user_index[kb.user_id] = set()
        self._user_index[kb.user_id].add(kb.id)

        # Update name index
        name_key = self._name_key(kb.user_id, kb.name)
        self._name_index[name_key] = kb.id

        return kb

    def get_by_id(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by its ID."""
        return self._knowledge_bases.get(kb_id)

    def get_by_name(self, user_id: str, name: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by user ID and name."""
        name_key = self._name_key(user_id, name)
        kb_id = self._name_index.get(name_key)
        return self._knowledge_bases.get(kb_id) if kb_id else None

    def exists_name(self, user_id: str, name: str) -> bool:
        """Check if a name already exists for this user."""
        name_key = self._name_key(user_id, name)
        return name_key in self._name_index

    def list_by_user(self, user_id: str) -> list[KnowledgeBase]:
        """List all knowledge bases for a user."""
        kb_ids = self._user_index.get(user_id, set())
        return [self._knowledge_bases[kb_id] for kb_id in kb_ids if kb_id in self._knowledge_bases]

    def update(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Update a knowledge base in the store."""
        old_kb = self._knowledge_bases.get(kb.id)
        if old_kb and old_kb.name != kb.name:
            # Remove old name index
            old_name_key = self._name_key(old_kb.user_id, old_kb.name)
            if old_name_key in self._name_index:
                del self._name_index[old_name_key]
            # Add new name index
            new_name_key = self._name_key(kb.user_id, kb.name)
            self._name_index[new_name_key] = kb.id

        kb.updated_at = datetime.utcnow()
        self._knowledge_bases[kb.id] = kb
        return kb

    def delete(self, kb_id: str) -> bool:
        """Delete a knowledge base from the store."""
        kb = self._knowledge_bases.get(kb_id)
        if kb:
            # Remove from main store
            del self._knowledge_bases[kb_id]

            # Remove from user index
            if kb.user_id in self._user_index:
                self._user_index[kb.user_id].discard(kb_id)

            # Remove from name index
            name_key = self._name_key(kb.user_id, kb.name)
            if name_key in self._name_index:
                del self._name_index[name_key]

            return True
        return False

    def clear(self) -> None:
        """Clear all knowledge bases from the store."""
        self._knowledge_bases.clear()
        self._user_index.clear()
        self._name_index.clear()


# Global knowledge base store instance
_kb_store = KnowledgeBaseStore()


def get_kb_store() -> KnowledgeBaseStore:
    """Get the global knowledge base store instance."""
    return _kb_store
