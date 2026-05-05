"""
Tests for the application management system.
"""
import pytest

from knowledgebot.applications.models import (
    ApplicationType,
    PrologueType,
    RAGSettings,
    ModelSettings,
    PromptTemplate,
    Application,
    ApplicationStore,
    get_app_store,
    reset_app_store,
)
from knowledgebot.applications.service import (
    create_application,
    get_application,
    get_application_by_name,
    list_applications,
    list_public_applications,
    update_application,
    delete_application,
    add_knowledge_base_to_app,
    remove_knowledge_base_from_app,
    duplicate_application,
)


class TestApplicationType:
    """Tests for ApplicationType enum."""

    def test_simple_type(self):
        """ApplicationType should have SIMPLE value."""
        assert ApplicationType.SIMPLE.value == "simple"

    def test_rag_type(self):
        """ApplicationType should have RAG value."""
        assert ApplicationType.RAG.value == "rag"

    def test_agent_type(self):
        """ApplicationType should have AGENT value."""
        assert ApplicationType.AGENT.value == "agent"


class TestPrologueType:
    """Tests for PrologueType enum."""

    def test_none_type(self):
        """PrologueType should have NONE value."""
        assert PrologueType.NONE.value == "none"

    def test_default_type(self):
        """PrologueType should have DEFAULT value."""
        assert PrologueType.DEFAULT.value == "default"

    def test_custom_type(self):
        """PrologueType should have CUSTOM value."""
        assert PrologueType.CUSTOM.value == "custom"


class TestRAGSettings:
    """Tests for RAGSettings class."""

    def test_default_settings(self):
        """Should create with default values."""
        settings = RAGSettings()
        assert settings.enabled is True
        assert settings.top_k == 5
        assert settings.similarity_threshold == 0.5
        assert settings.max_context_length == 2000
        assert settings.show_source is True

    def test_custom_settings(self):
        """Should accept custom values."""
        settings = RAGSettings(
            enabled=False,
            top_k=10,
            similarity_threshold=0.7,
            max_context_length=4000,
            show_source=False,
        )
        assert settings.enabled is False
        assert settings.top_k == 10
        assert settings.similarity_threshold == 0.7

    def test_invalid_top_k(self):
        """Should reject top_k less than 1."""
        with pytest.raises(ValueError, match="top_k"):
            RAGSettings(top_k=0)

    def test_invalid_threshold_low(self):
        """Should reject threshold below 0."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            RAGSettings(similarity_threshold=-0.1)

    def test_invalid_threshold_high(self):
        """Should reject threshold above 1."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            RAGSettings(similarity_threshold=1.5)

    def test_invalid_context_length(self):
        """Should reject context length less than 100."""
        with pytest.raises(ValueError, match="max_context_length"):
            RAGSettings(max_context_length=50)

    def test_to_dict(self):
        """Should convert to dictionary."""
        settings = RAGSettings()
        data = settings.to_dict()
        assert data["enabled"] is True
        assert data["top_k"] == 5


class TestModelSettings:
    """Tests for ModelSettings class."""

    def test_default_settings(self):
        """Should create with default values."""
        settings = ModelSettings()
        assert settings.model_name == "gpt-3.5-turbo"
        assert settings.temperature == 0.7
        assert settings.max_tokens is None
        assert settings.top_p == 1.0

    def test_custom_settings(self):
        """Should accept custom values."""
        settings = ModelSettings(
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
        )
        assert settings.model_name == "gpt-4"
        assert settings.temperature == 0.5

    def test_invalid_temperature(self):
        """Should reject invalid temperature."""
        with pytest.raises(ValueError, match="temperature"):
            ModelSettings(temperature=3.0)

    def test_invalid_top_p(self):
        """Should reject invalid top_p."""
        with pytest.raises(ValueError, match="top_p"):
            ModelSettings(top_p=1.5)

    def test_invalid_max_tokens(self):
        """Should reject non-positive max_tokens."""
        with pytest.raises(ValueError, match="max_tokens"):
            ModelSettings(max_tokens=0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        settings = ModelSettings()
        data = settings.to_dict()
        assert data["model_name"] == "gpt-3.5-turbo"


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_default_template(self):
        """Should create with default values."""
        template = PromptTemplate()
        assert "helpful AI assistant" in template.system_prompt
        assert "{context}" in template.context_template
        assert "{question}" in template.context_template

    def test_custom_template(self):
        """Should accept custom values."""
        template = PromptTemplate(
            system_prompt="You are a code expert.",
            context_template="Context: {context}\nQ: {question}",
            no_context_template="Q: {question}",
        )
        assert template.system_prompt == "You are a code expert."

    def test_format_with_context(self):
        """Should format prompt with context."""
        template = PromptTemplate(
            context_template="Context: {context}\n\nQuestion: {question}"
        )
        result = template.format_with_context("What is Python?", "Python is a language.")
        assert "Python is a language" in result
        assert "What is Python" in result

    def test_format_without_context(self):
        """Should format prompt without context."""
        template = PromptTemplate(
            no_context_template="Please answer: {question}"
        )
        result = template.format_without_context("What is Python?")
        assert "What is Python" in result

    def test_to_dict(self):
        """Should convert to dictionary."""
        template = PromptTemplate()
        data = template.to_dict()
        assert "system_prompt" in data
        assert "context_template" in data


class TestApplication:
    """Tests for Application model."""

    def test_application_creation(self):
        """Should create application with required fields."""
        app = Application(name="Test App", user_id="user123")
        assert app.name == "Test App"
        assert app.user_id == "user123"
        assert app.id is not None
        assert app.app_type == ApplicationType.RAG
        assert app.is_public is False

    def test_application_with_options(self):
        """Should create application with custom options."""
        app = Application(
            name="Public App",
            user_id="user123",
            description="A public app",
            app_type=ApplicationType.AGENT,
            is_public=True,
        )
        assert app.description == "A public app"
        assert app.app_type == ApplicationType.AGENT
        assert app.is_public is True

    def test_application_to_dict(self):
        """Should convert to dictionary."""
        app = Application(
            name="Test App",
            user_id="user123",
            knowledge_base_ids=["kb1", "kb2"],
        )
        data = app.to_dict()
        assert data["name"] == "Test App"
        assert data["user_id"] == "user123"
        assert data["knowledge_base_ids"] == ["kb1", "kb2"]
        assert "rag_settings" in data
        assert "model_settings" in data

    def test_add_knowledge_base(self):
        """Should add knowledge base."""
        app = Application(name="Test", user_id="user1")
        result = app.add_knowledge_base("kb1")
        assert result is True
        assert "kb1" in app.knowledge_base_ids

    def test_add_knowledge_base_duplicate(self):
        """Should not add duplicate knowledge base."""
        app = Application(name="Test", user_id="user1", knowledge_base_ids=["kb1"])
        result = app.add_knowledge_base("kb1")
        assert result is False
        assert app.knowledge_base_ids.count("kb1") == 1

    def test_remove_knowledge_base(self):
        """Should remove knowledge base."""
        app = Application(name="Test", user_id="user1", knowledge_base_ids=["kb1", "kb2"])
        result = app.remove_knowledge_base("kb1")
        assert result is True
        assert "kb1" not in app.knowledge_base_ids

    def test_remove_knowledge_base_not_present(self):
        """Should handle removing non-present knowledge base."""
        app = Application(name="Test", user_id="user1")
        result = app.remove_knowledge_base("kb1")
        assert result is False


class TestApplicationStore:
    """Tests for ApplicationStore."""

    def setup_method(self):
        """Reset store before each test."""
        reset_app_store()

    def test_add_and_get_by_id(self):
        """Should add and retrieve by ID."""
        store = ApplicationStore()
        app = Application(name="Test", user_id="user1")
        store.add(app)
        retrieved = store.get_by_id(app.id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_by_id_not_found(self):
        """Should return None for unknown ID."""
        store = ApplicationStore()
        assert store.get_by_id("unknown") is None

    def test_get_by_name(self):
        """Should retrieve by name and user."""
        store = ApplicationStore()
        app = Application(name="MyApp", user_id="user1")
        store.add(app)
        retrieved = store.get_by_name("MyApp", "user1")
        assert retrieved is not None
        assert retrieved.id == app.id

    def test_get_by_name_wrong_user(self):
        """Should not find app for different user."""
        store = ApplicationStore()
        app = Application(name="MyApp", user_id="user1")
        store.add(app)
        assert store.get_by_name("MyApp", "user2") is None

    def test_exists_name(self):
        """Should check name existence."""
        store = ApplicationStore()
        app = Application(name="MyApp", user_id="user1")
        store.add(app)
        assert store.exists_name("MyApp", "user1") is True
        assert store.exists_name("OtherApp", "user1") is False

    def test_list_by_user(self):
        """Should list apps by user."""
        store = ApplicationStore()
        app1 = Application(name="App1", user_id="user1")
        app2 = Application(name="App2", user_id="user1")
        app3 = Application(name="App3", user_id="user2")
        store.add(app1)
        store.add(app2)
        store.add(app3)

        user1_apps = store.list_by_user("user1")
        assert len(user1_apps) == 2

    def test_list_public(self):
        """Should list public apps."""
        store = ApplicationStore()
        app1 = Application(name="Public1", user_id="user1", is_public=True)
        app2 = Application(name="Private", user_id="user1", is_public=False)
        app3 = Application(name="Public2", user_id="user2", is_public=True)
        store.add(app1)
        store.add(app2)
        store.add(app3)

        public_apps = store.list_public()
        assert len(public_apps) == 2

    def test_delete(self):
        """Should delete application."""
        store = ApplicationStore()
        app = Application(name="Test", user_id="user1")
        store.add(app)
        result = store.delete(app.id)
        assert result is True
        assert store.get_by_id(app.id) is None

    def test_delete_not_found(self):
        """Should return False for unknown ID."""
        store = ApplicationStore()
        assert store.delete("unknown") is False


class TestCreateApplication:
    """Tests for create_application function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_create_basic(self):
        """Should create application."""
        app = create_application(name="Test App", user_id="user123")
        assert app.name == "Test App"
        assert app.user_id == "user123"
        assert app.id is not None

    def test_create_with_options(self):
        """Should create with custom options."""
        rag = RAGSettings(top_k=10)
        model = ModelSettings(model_name="gpt-4")
        app = create_application(
            name="Advanced App",
            user_id="user123",
            description="An advanced app",
            app_type=ApplicationType.AGENT,
            is_public=True,
            rag_settings=rag,
            model_settings=model,
        )
        assert app.app_type == ApplicationType.AGENT
        assert app.is_public is True
        assert app.rag_settings.top_k == 10
        assert app.model_settings.model_name == "gpt-4"

    def test_create_duplicate_name(self):
        """Should reject duplicate name for same user."""
        create_application(name="MyApp", user_id="user1")
        with pytest.raises(ValueError, match="already exists"):
            create_application(name="MyApp", user_id="user1")

    def test_create_same_name_different_user(self):
        """Should allow same name for different users."""
        app1 = create_application(name="MyApp", user_id="user1")
        app2 = create_application(name="MyApp", user_id="user2")
        assert app1.id != app2.id


class TestGetApplication:
    """Tests for get_application function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_get_exists(self):
        """Should return application if exists."""
        created = create_application(name="Test", user_id="user1")
        retrieved = get_application(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_not_found(self):
        """Should return None if not found."""
        assert get_application("unknown") is None


class TestGetApplicationByName:
    """Tests for get_application_by_name function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_get_by_name_exists(self):
        """Should return application by name."""
        created = create_application(name="TestApp", user_id="user1")
        retrieved = get_application_by_name("TestApp", "user1")
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_by_name_not_found(self):
        """Should return None if not found."""
        assert get_application_by_name("Unknown", "user1") is None


class TestListApplications:
    """Tests for list_applications function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_list_returns_user_apps(self):
        """Should return all apps for user."""
        create_application(name="App1", user_id="user1")
        create_application(name="App2", user_id="user1")
        create_application(name="App3", user_id="user2")

        apps = list_applications("user1")
        assert len(apps) == 2

    def test_list_empty(self):
        """Should return empty list if no apps."""
        apps = list_applications("user1")
        assert apps == []


class TestListPublicApplications:
    """Tests for list_public_applications function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_list_public(self):
        """Should return public applications."""
        create_application(name="Public1", user_id="user1", is_public=True)
        create_application(name="Private", user_id="user1", is_public=False)
        create_application(name="Public2", user_id="user2", is_public=True)

        apps = list_public_applications()
        assert len(apps) == 2
        assert all(app.is_public for app in apps)


class TestUpdateApplication:
    """Tests for update_application function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_update_name(self):
        """Should update name."""
        app = create_application(name="OldName", user_id="user1")
        updated = update_application(app.id, name="NewName")
        assert updated is not None
        assert updated.name == "NewName"

    def test_update_description(self):
        """Should update description."""
        app = create_application(name="Test", user_id="user1")
        updated = update_application(app.id, description="New description")
        assert updated is not None
        assert updated.description == "New description"

    def test_update_settings(self):
        """Should update settings."""
        app = create_application(name="Test", user_id="user1")
        new_rag = RAGSettings(top_k=20)
        updated = update_application(app.id, rag_settings=new_rag)
        assert updated is not None
        assert updated.rag_settings.top_k == 20

    def test_update_not_found(self):
        """Should return None if not found."""
        result = update_application("unknown", name="New")
        assert result is None

    def test_update_duplicate_name(self):
        """Should reject duplicate name."""
        create_application(name="App1", user_id="user1")
        app2 = create_application(name="App2", user_id="user1")
        with pytest.raises(ValueError, match="already exists"):
            update_application(app2.id, name="App1")


class TestDeleteApplication:
    """Tests for delete_application function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_delete_success(self):
        """Should delete application."""
        app = create_application(name="Test", user_id="user1")
        result = delete_application(app.id)
        assert result is True
        assert get_application(app.id) is None

    def test_delete_not_found(self):
        """Should return False if not found."""
        result = delete_application("unknown")
        assert result is False


class TestAddRemoveKnowledgeBase:
    """Tests for knowledge base association functions."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_add_knowledge_base(self):
        """Should add knowledge base to app."""
        app = create_application(name="Test", user_id="user1")
        updated = add_knowledge_base_to_app(app.id, "kb123")
        assert updated is not None
        assert "kb123" in updated.knowledge_base_ids

    def test_add_knowledge_base_not_found(self):
        """Should return None if app not found."""
        result = add_knowledge_base_to_app("unknown", "kb123")
        assert result is None

    def test_remove_knowledge_base(self):
        """Should remove knowledge base from app."""
        app = create_application(
            name="Test",
            user_id="user1",
            knowledge_base_ids=["kb1", "kb2"],
        )
        updated = remove_knowledge_base_from_app(app.id, "kb1")
        assert updated is not None
        assert "kb1" not in updated.knowledge_base_ids
        assert "kb2" in updated.knowledge_base_ids

    def test_remove_knowledge_base_not_found(self):
        """Should return None if app not found."""
        result = remove_knowledge_base_from_app("unknown", "kb123")
        assert result is None


class TestDuplicateApplication:
    """Tests for duplicate_application function."""

    def setup_method(self):
        """Reset store before each test."""
        get_app_store().clear()

    def test_duplicate_basic(self):
        """Should duplicate application."""
        original = create_application(
            name="Original",
            user_id="user1",
            description="Original description",
            knowledge_base_ids=["kb1"],
        )
        copy = duplicate_application(original.id, "Copy", "user1")

        assert copy is not None
        assert copy.name == "Copy"
        assert copy.description == "Original description"
        assert copy.id != original.id
        assert "kb1" in copy.knowledge_base_ids

    def test_duplicate_to_different_user(self):
        """Should allow duplicating to different user."""
        original = create_application(name="Original", user_id="user1")
        copy = duplicate_application(original.id, "Copy", "user2")

        assert copy is not None
        assert copy.user_id == "user2"

    def test_duplicate_is_private(self):
        """Duplicated app should be private."""
        original = create_application(name="Original", user_id="user1", is_public=True)
        copy = duplicate_application(original.id, "Copy", "user1")

        assert copy is not None
        assert copy.is_public is False

    def test_duplicate_not_found(self):
        """Should return None if source not found."""
        result = duplicate_application("unknown", "Copy", "user1")
        assert result is None

    def test_duplicate_name_conflict(self):
        """Should reject duplicate name."""
        original = create_application(name="Original", user_id="user1")
        create_application(name="Existing", user_id="user1")

        with pytest.raises(ValueError, match="already exists"):
            duplicate_application(original.id, "Existing", "user1")
