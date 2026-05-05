"""
Tests for the knowledge base management system.
"""
import pytest

from knowledgebot.knowledge.models import (
    KnowledgeBase,
    KnowledgeBaseStore,
    get_kb_store,
    default_kb_settings,
)
from knowledgebot.knowledge.service import (
    create_knowledge_base,
    get_knowledge_base,
    list_knowledge_bases,
    update_knowledge_base,
    delete_knowledge_base,
    get_knowledge_base_by_name,
)


class TestDefaultSettings:
    """Tests for default knowledge base settings."""

    def test_default_settings_has_chunk_size(self):
        """Default settings should include chunk_size."""
        settings = default_kb_settings()
        assert "chunk_size" in settings
        assert settings["chunk_size"] == 512

    def test_default_settings_has_similarity_threshold(self):
        """Default settings should include similarity_threshold."""
        settings = default_kb_settings()
        assert "similarity_threshold" in settings
        assert settings["similarity_threshold"] == 0.7

    def test_default_settings_has_top_k(self):
        """Default settings should include top_k."""
        settings = default_kb_settings()
        assert "top_k" in settings
        assert settings["top_k"] == 5


class TestKnowledgeBaseModel:
    """Tests for the KnowledgeBase model."""

    def test_knowledge_base_creation(self):
        """KnowledgeBase should be created with required fields."""
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        assert kb.name == "Test KB"
        assert kb.user_id == "user123"
        assert kb.description == ""
        assert kb.id is not None
        assert kb.created_at is not None
        assert kb.updated_at is not None

    def test_knowledge_base_with_description(self):
        """KnowledgeBase should accept description."""
        kb = KnowledgeBase(
            name="Test KB",
            user_id="user123",
            description="A test knowledge base"
        )
        assert kb.description == "A test knowledge base"

    def test_knowledge_base_default_settings(self):
        """KnowledgeBase should have default settings."""
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        assert "chunk_size" in kb.settings
        assert "similarity_threshold" in kb.settings

    def test_knowledge_base_to_dict(self):
        """to_dict should return dictionary representation."""
        kb = KnowledgeBase(
            name="Test KB",
            user_id="user123",
            description="Test description"
        )
        data = kb.to_dict()
        assert data["id"] == kb.id
        assert data["name"] == "Test KB"
        assert data["description"] == "Test description"
        assert data["user_id"] == "user123"
        assert "settings" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_knowledge_base_update_settings(self):
        """update_settings should merge with existing settings."""
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        original_chunk_size = kb.settings["chunk_size"]

        kb.update_settings({"similarity_threshold": 0.9})

        assert kb.settings["similarity_threshold"] == 0.9
        assert kb.settings["chunk_size"] == original_chunk_size


class TestKnowledgeBaseStore:
    """Tests for KnowledgeBaseStore."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_add_and_get_by_id(self):
        """Should add and retrieve knowledge base by ID."""
        store = KnowledgeBaseStore()
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        store.add(kb)

        retrieved = store.get_by_id(kb.id)
        assert retrieved is not None
        assert retrieved.name == "Test KB"

    def test_get_by_name(self):
        """Should retrieve knowledge base by user ID and name."""
        store = KnowledgeBaseStore()
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        store.add(kb)

        retrieved = store.get_by_name("user123", "Test KB")
        assert retrieved is not None
        assert retrieved.id == kb.id

    def test_get_by_name_not_found(self):
        """Should return None if knowledge base not found by name."""
        store = KnowledgeBaseStore()
        retrieved = store.get_by_name("user123", "Nonexistent")
        assert retrieved is None

    def test_exists_name(self):
        """Should check if name exists for user."""
        store = KnowledgeBaseStore()
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        store.add(kb)

        assert store.exists_name("user123", "Test KB") is True
        assert store.exists_name("user123", "Other KB") is False
        assert store.exists_name("user456", "Test KB") is False

    def test_list_by_user(self):
        """Should list all knowledge bases for a user."""
        store = KnowledgeBaseStore()
        kb1 = KnowledgeBase(name="KB 1", user_id="user123")
        kb2 = KnowledgeBase(name="KB 2", user_id="user123")
        kb3 = KnowledgeBase(name="KB 3", user_id="user456")
        store.add(kb1)
        store.add(kb2)
        store.add(kb3)

        user123_kbs = store.list_by_user("user123")
        assert len(user123_kbs) == 2
        names = {kb.name for kb in user123_kbs}
        assert names == {"KB 1", "KB 2"}

    def test_delete(self):
        """Should delete knowledge base and return True."""
        store = KnowledgeBaseStore()
        kb = KnowledgeBase(name="Test KB", user_id="user123")
        store.add(kb)

        result = store.delete(kb.id)
        assert result is True
        assert store.get_by_id(kb.id) is None

    def test_delete_nonexistent(self):
        """Should return False when deleting nonexistent knowledge base."""
        store = KnowledgeBaseStore()
        result = store.delete("nonexistent-id")
        assert result is False


class TestCreateKnowledgeBase:
    """Tests for create_knowledge_base function."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_create_knowledge_base_success(self):
        """create_knowledge_base should create and return new knowledge base."""
        kb = create_knowledge_base(
            name="My KB",
            user_id="user123",
            description="Test description"
        )
        assert kb is not None
        assert kb.name == "My KB"
        assert kb.user_id == "user123"
        assert kb.description == "Test description"

    def test_create_knowledge_base_with_settings(self):
        """create_knowledge_base should accept custom settings."""
        kb = create_knowledge_base(
            name="My KB",
            user_id="user123",
            settings={"chunk_size": 1024}
        )
        assert kb.settings["chunk_size"] == 1024

    def test_create_knowledge_base_duplicate_name(self):
        """create_knowledge_base should raise error for duplicate name."""
        create_knowledge_base(name="My KB", user_id="user123")
        with pytest.raises(ValueError, match="already exists"):
            create_knowledge_base(name="My KB", user_id="user123")

    def test_create_knowledge_base_same_name_different_user(self):
        """create_knowledge_base should allow same name for different users."""
        kb1 = create_knowledge_base(name="My KB", user_id="user123")
        kb2 = create_knowledge_base(name="My KB", user_id="user456")
        assert kb1.id != kb2.id


class TestGetKnowledgeBase:
    """Tests for get_knowledge_base function."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_get_knowledge_base_exists(self):
        """get_knowledge_base should return knowledge base if exists."""
        created = create_knowledge_base(name="My KB", user_id="user123")
        retrieved = get_knowledge_base(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_knowledge_base_not_found(self):
        """get_knowledge_base should return None if not found."""
        result = get_knowledge_base("nonexistent-id")
        assert result is None


class TestListKnowledgeBases:
    """Tests for list_knowledge_bases function."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_list_knowledge_bases_returns_user_kbs(self):
        """list_knowledge_bases should return all KBs for a user."""
        create_knowledge_base(name="KB 1", user_id="user123")
        create_knowledge_base(name="KB 2", user_id="user123")
        create_knowledge_base(name="KB 3", user_id="user456")

        kbs = list_knowledge_bases("user123")
        assert len(kbs) == 2
        names = {kb.name for kb in kbs}
        assert names == {"KB 1", "KB 2"}

    def test_list_knowledge_bases_empty(self):
        """list_knowledge_bases should return empty list if no KBs."""
        kbs = list_knowledge_bases("user123")
        assert kbs == []


class TestUpdateKnowledgeBase:
    """Tests for update_knowledge_base function."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_update_knowledge_base_name(self):
        """update_knowledge_base should update name."""
        kb = create_knowledge_base(name="Old Name", user_id="user123")
        updated = update_knowledge_base(kb.id, name="New Name")
        assert updated is not None
        assert updated.name == "New Name"

    def test_update_knowledge_base_description(self):
        """update_knowledge_base should update description."""
        kb = create_knowledge_base(name="My KB", user_id="user123")
        updated = update_knowledge_base(kb.id, description="New description")
        assert updated is not None
        assert updated.description == "New description"

    def test_update_knowledge_base_settings(self):
        """update_knowledge_base should update settings."""
        kb = create_knowledge_base(name="My KB", user_id="user123")
        updated = update_knowledge_base(kb.id, settings={"chunk_size": 1024})
        assert updated is not None
        assert updated.settings["chunk_size"] == 1024

    def test_update_knowledge_base_not_found(self):
        """update_knowledge_base should return None if not found."""
        result = update_knowledge_base("nonexistent-id", name="New Name")
        assert result is None

    def test_update_knowledge_base_duplicate_name(self):
        """update_knowledge_base should raise error for duplicate name."""
        create_knowledge_base(name="KB 1", user_id="user123")
        kb2 = create_knowledge_base(name="KB 2", user_id="user123")
        with pytest.raises(ValueError, match="already exists"):
            update_knowledge_base(kb2.id, name="KB 1")


class TestDeleteKnowledgeBase:
    """Tests for delete_knowledge_base function."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_delete_knowledge_base_success(self):
        """delete_knowledge_base should delete and return True."""
        kb = create_knowledge_base(name="My KB", user_id="user123")
        result = delete_knowledge_base(kb.id)
        assert result is True
        assert get_knowledge_base(kb.id) is None

    def test_delete_knowledge_base_not_found(self):
        """delete_knowledge_base should return False if not found."""
        result = delete_knowledge_base("nonexistent-id")
        assert result is False


class TestGetKnowledgeBaseByName:
    """Tests for get_knowledge_base_by_name function."""

    def setup_method(self):
        """Reset store before each test."""
        get_kb_store().clear()

    def test_get_knowledge_base_by_name_found(self):
        """get_knowledge_base_by_name should return KB if found."""
        created = create_knowledge_base(name="My KB", user_id="user123")
        retrieved = get_knowledge_base_by_name("user123", "My KB")
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_knowledge_base_by_name_not_found(self):
        """get_knowledge_base_by_name should return None if not found."""
        result = get_knowledge_base_by_name("user123", "Nonexistent")
        assert result is None
