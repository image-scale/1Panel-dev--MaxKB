"""
Knowledge base service for the KnowledgeBot platform.

Provides high-level operations for managing knowledge bases.
"""
from typing import Optional, Dict, Any, List

from knowledgebot.knowledge.models import (
    KnowledgeBase,
    KnowledgeBaseStore,
    get_kb_store,
)


def create_knowledge_base(
    name: str,
    user_id: str,
    description: str = "",
    settings: Optional[Dict[str, Any]] = None
) -> KnowledgeBase:
    """
    Create a new knowledge base.

    Args:
        name: Name of the knowledge base
        user_id: ID of the user creating the knowledge base
        description: Optional description
        settings: Optional custom settings (merged with defaults)

    Returns:
        The newly created KnowledgeBase object

    Raises:
        ValueError: If a knowledge base with the same name already exists for this user
    """
    store = get_kb_store()

    if store.exists_name(user_id, name):
        raise ValueError(f"Knowledge base '{name}' already exists for this user")

    kb = KnowledgeBase(
        name=name,
        user_id=user_id,
        description=description,
    )

    if settings:
        kb.update_settings(settings)

    return store.add(kb)


def get_knowledge_base(kb_id: str) -> Optional[KnowledgeBase]:
    """
    Get a knowledge base by its ID.

    Args:
        kb_id: The ID of the knowledge base

    Returns:
        The KnowledgeBase object if found, None otherwise
    """
    store = get_kb_store()
    return store.get_by_id(kb_id)


def list_knowledge_bases(user_id: str) -> List[KnowledgeBase]:
    """
    List all knowledge bases for a user.

    Args:
        user_id: The ID of the user

    Returns:
        List of KnowledgeBase objects belonging to the user
    """
    store = get_kb_store()
    return store.list_by_user(user_id)


def update_knowledge_base(
    kb_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None
) -> Optional[KnowledgeBase]:
    """
    Update a knowledge base.

    Args:
        kb_id: The ID of the knowledge base to update
        name: New name (optional)
        description: New description (optional)
        settings: Settings to update (merged with existing)

    Returns:
        The updated KnowledgeBase object, or None if not found

    Raises:
        ValueError: If the new name already exists for this user
    """
    store = get_kb_store()
    kb = store.get_by_id(kb_id)

    if kb is None:
        return None

    if name is not None and name != kb.name:
        if store.exists_name(kb.user_id, name):
            raise ValueError(f"Knowledge base '{name}' already exists for this user")
        kb.name = name

    if description is not None:
        kb.description = description

    if settings is not None:
        kb.update_settings(settings)

    return store.update(kb)


def delete_knowledge_base(kb_id: str) -> bool:
    """
    Delete a knowledge base.

    Args:
        kb_id: The ID of the knowledge base to delete

    Returns:
        True if the knowledge base was deleted, False if not found
    """
    store = get_kb_store()
    return store.delete(kb_id)


def get_knowledge_base_by_name(user_id: str, name: str) -> Optional[KnowledgeBase]:
    """
    Get a knowledge base by user ID and name.

    Args:
        user_id: The ID of the user
        name: The name of the knowledge base

    Returns:
        The KnowledgeBase object if found, None otherwise
    """
    store = get_kb_store()
    return store.get_by_name(user_id, name)
