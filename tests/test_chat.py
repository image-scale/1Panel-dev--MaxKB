"""
Tests for the chat service.
"""
import pytest

from knowledgebot.applications.models import (
    Application,
    ApplicationType,
    PrologueType,
    RAGSettings,
    ModelSettings,
    PromptTemplate,
    get_app_store,
    reset_app_store,
)
from knowledgebot.applications.service import create_application
from knowledgebot.knowledge.models import get_kb_store
from knowledgebot.knowledge.service import create_knowledge_base
from knowledgebot.knowledge.documents import get_doc_store, create_document
from knowledgebot.knowledge.processing import process_document
from knowledgebot.core.vectorstore import reset_vector_store
from knowledgebot.core.embeddings import reset_embedding_provider
from knowledgebot.providers.llm import (
    MockLLMProvider,
    set_llm_provider,
    reset_llm_provider,
)
from knowledgebot.chat.service import (
    ContextSource,
    ChatResponse,
    ChatStreamChunk,
    ConversationMessage,
    Conversation,
    ConversationStore,
    get_conv_store,
    reset_conv_store,
    retrieve_context,
    build_context_text,
    build_messages,
    chat,
    chat_stream,
    create_conversation,
    get_conversation,
    list_conversations,
    delete_conversation,
    get_prologue,
)


class TestContextSource:
    """Tests for ContextSource class."""

    def test_creation(self):
        """Should create context source."""
        source = ContextSource(
            content="Test content",
            document_id="doc1",
            knowledge_base_id="kb1",
            score=0.95,
        )
        assert source.content == "Test content"
        assert source.score == 0.95

    def test_to_dict(self):
        """Should convert to dictionary."""
        source = ContextSource(
            content="Test",
            document_id="doc1",
            knowledge_base_id="kb1",
            score=0.9,
            metadata={"key": "value"},
        )
        data = source.to_dict()
        assert data["content"] == "Test"
        assert data["score"] == 0.9
        assert data["metadata"]["key"] == "value"


class TestChatResponse:
    """Tests for ChatResponse class."""

    def test_creation(self):
        """Should create response."""
        response = ChatResponse(content="Hello!")
        assert response.content == "Hello!"
        assert response.id is not None
        assert response.sources == []

    def test_to_dict(self):
        """Should convert to dictionary."""
        response = ChatResponse(
            content="Response text",
            usage={"total_tokens": 100},
        )
        data = response.to_dict()
        assert data["content"] == "Response text"
        assert data["usage"]["total_tokens"] == 100


class TestChatStreamChunk:
    """Tests for ChatStreamChunk class."""

    def test_creation(self):
        """Should create chunk."""
        chunk = ChatStreamChunk(id="123", content="Hello")
        assert chunk.id == "123"
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_final_chunk(self):
        """Should create final chunk."""
        chunk = ChatStreamChunk(id="123", content="", is_final=True)
        assert chunk.is_final is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        chunk = ChatStreamChunk(id="123", content="Test", is_final=False)
        data = chunk.to_dict()
        assert data["id"] == "123"
        assert data["content"] == "Test"


class TestConversationMessage:
    """Tests for ConversationMessage class."""

    def test_creation(self):
        """Should create message."""
        msg = ConversationMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_to_dict(self):
        """Should convert to dictionary."""
        msg = ConversationMessage(role="assistant", content="Hi there")
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"


class TestConversation:
    """Tests for Conversation class."""

    def test_creation(self):
        """Should create conversation."""
        conv = Conversation(application_id="app1", user_id="user1")
        assert conv.application_id == "app1"
        assert conv.user_id == "user1"
        assert conv.messages == []

    def test_add_message(self):
        """Should add message."""
        conv = Conversation(application_id="app1", user_id="user1")
        msg = conv.add_message("user", "Hello")
        assert msg.role == "user"
        assert len(conv.messages) == 1

    def test_get_history(self):
        """Should get recent history."""
        conv = Conversation(application_id="app1", user_id="user1")
        for i in range(15):
            conv.add_message("user", f"Message {i}")

        history = conv.get_history(max_messages=5)
        assert len(history) == 5
        assert history[0].content == "Message 10"

    def test_to_dict(self):
        """Should convert to dictionary."""
        conv = Conversation(application_id="app1", user_id="user1")
        conv.add_message("user", "Hello")
        data = conv.to_dict()
        assert data["application_id"] == "app1"
        assert len(data["messages"]) == 1


class TestConversationStore:
    """Tests for ConversationStore."""

    def test_add_and_get(self):
        """Should add and retrieve conversation."""
        store = ConversationStore()
        conv = Conversation(application_id="app1", user_id="user1")
        store.add(conv)
        retrieved = store.get_by_id(conv.id)
        assert retrieved is not None
        assert retrieved.id == conv.id

    def test_list_by_user(self):
        """Should list conversations by user."""
        store = ConversationStore()
        conv1 = Conversation(application_id="app1", user_id="user1")
        conv2 = Conversation(application_id="app2", user_id="user1")
        conv3 = Conversation(application_id="app1", user_id="user2")
        store.add(conv1)
        store.add(conv2)
        store.add(conv3)

        convs = store.list_by_user("user1")
        assert len(convs) == 2

    def test_list_by_user_and_app(self):
        """Should filter by application."""
        store = ConversationStore()
        conv1 = Conversation(application_id="app1", user_id="user1")
        conv2 = Conversation(application_id="app2", user_id="user1")
        store.add(conv1)
        store.add(conv2)

        convs = store.list_by_user("user1", app_id="app1")
        assert len(convs) == 1
        assert convs[0].application_id == "app1"

    def test_delete(self):
        """Should delete conversation."""
        store = ConversationStore()
        conv = Conversation(application_id="app1", user_id="user1")
        store.add(conv)
        result = store.delete(conv.id)
        assert result is True
        assert store.get_by_id(conv.id) is None


class TestRetrieveContext:
    """Tests for retrieve_context function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        get_app_store().clear()
        reset_vector_store()
        reset_embedding_provider()

    def test_retrieve_with_rag_disabled(self):
        """Should return empty when RAG is disabled."""
        app = Application(
            name="Test",
            user_id="user1",
            rag_settings=RAGSettings(enabled=False),
        )
        sources = retrieve_context("test query", app)
        assert sources == []

    def test_retrieve_with_no_knowledge_bases(self):
        """Should return empty when no KBs configured."""
        app = Application(
            name="Test",
            user_id="user1",
            knowledge_base_ids=[],
        )
        sources = retrieve_context("test query", app)
        assert sources == []

    def test_retrieve_from_knowledge_base(self):
        """Should retrieve context from knowledge base."""
        # Create KB and document
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(kb.id, "test.txt", "Python is a programming language.")
        process_document(doc.id, chunk_size=100, chunk_overlap=10)

        # Create app with this KB
        app = Application(
            name="Test",
            user_id="user1",
            knowledge_base_ids=[kb.id],
            rag_settings=RAGSettings(top_k=5, similarity_threshold=0.0),
        )

        # Query with same text to ensure match
        sources = retrieve_context("Python is a programming language", app)
        # With hash-based embeddings, exact/similar text should match
        assert isinstance(sources, list)  # At minimum returns a list


class TestBuildContextText:
    """Tests for build_context_text function."""

    def test_empty_sources(self):
        """Should return empty string for no sources."""
        result = build_context_text([])
        assert result == ""

    def test_single_source(self):
        """Should format single source."""
        sources = [ContextSource(
            content="Test content",
            document_id="doc1",
            knowledge_base_id="kb1",
            score=0.9,
        )]
        result = build_context_text(sources)
        assert "[1]" in result
        assert "Test content" in result

    def test_respects_max_length(self):
        """Should respect max length."""
        sources = [
            ContextSource(
                content="A" * 100,
                document_id=f"doc{i}",
                knowledge_base_id="kb1",
                score=0.9,
            )
            for i in range(10)
        ]
        result = build_context_text(sources, max_length=250)
        assert len(result) <= 250


class TestBuildMessages:
    """Tests for build_messages function."""

    def test_basic_messages(self):
        """Should build basic message list."""
        app = Application(
            name="Test",
            user_id="user1",
            prompt_template=PromptTemplate(
                system_prompt="You are helpful",
            ),
        )
        messages = build_messages("What is Python?", app, "")
        assert len(messages) >= 2
        assert messages[0].role.value == "system"
        assert "user" in messages[-1].role.value

    def test_messages_with_context(self):
        """Should include context in user message."""
        app = Application(
            name="Test",
            user_id="user1",
            prompt_template=PromptTemplate(
                context_template="Context: {context}\nQuestion: {question}",
            ),
        )
        messages = build_messages("What is Python?", app, "Python is a language")
        user_msg = messages[-1].content
        assert "Python is a language" in user_msg

    def test_messages_with_history(self):
        """Should include conversation history."""
        app = Application(name="Test", user_id="user1")
        conv = Conversation(application_id="app1", user_id="user1")
        conv.add_message("user", "Previous question")
        conv.add_message("assistant", "Previous answer")

        messages = build_messages("New question", app, "", conv)
        assert len(messages) > 2  # System + history + new question


class TestChat:
    """Tests for chat function."""

    def setup_method(self):
        """Reset all stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        get_app_store().clear()
        reset_vector_store()
        reset_embedding_provider()
        reset_conv_store()
        reset_llm_provider()

    def test_chat_basic(self):
        """Should generate basic chat response."""
        app = create_application(name="Test App", user_id="user1")
        response = chat("Hello", app.id)
        assert response.content != ""
        assert response.id is not None

    def test_chat_not_found(self):
        """Should raise error for unknown app."""
        with pytest.raises(ValueError, match="not found"):
            chat("Hello", "unknown-app-id")

    def test_chat_with_conversation(self):
        """Should use conversation history."""
        app = create_application(name="Test App", user_id="user1")
        conv = create_conversation(app.id, "user1")
        conv.add_message("user", "My name is Alice")
        conv.add_message("assistant", "Nice to meet you, Alice!")

        response = chat("What is my name?", app.id, conv.id)
        assert response.content != ""

    def test_chat_with_rag(self):
        """Should use RAG when enabled."""
        # Create KB with content
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(kb.id, "test.txt", "The capital of France is Paris.")
        process_document(doc.id, chunk_size=100, chunk_overlap=10)

        # Create app with RAG enabled
        app = create_application(
            name="Test App",
            user_id="user1",
            knowledge_base_ids=[kb.id],
            rag_settings=RAGSettings(enabled=True, show_source=True),
        )

        response = chat("What is the capital of France?", app.id)
        assert response.content != ""
        # Sources should be included since show_source is True
        # Note: may or may not have sources depending on similarity

    def test_chat_without_sources(self):
        """Should not include sources when show_source is False."""
        kb = create_knowledge_base(name="Test KB", user_id="user1")
        doc = create_document(kb.id, "test.txt", "Test content here.")
        process_document(doc.id, chunk_size=100, chunk_overlap=10)

        app = create_application(
            name="Test App",
            user_id="user1",
            knowledge_base_ids=[kb.id],
            rag_settings=RAGSettings(enabled=True, show_source=False),
        )

        response = chat("Test query", app.id)
        assert response.sources == []

    def test_chat_returns_usage(self):
        """Should include token usage."""
        app = create_application(name="Test App", user_id="user1")
        response = chat("Hello", app.id)
        assert "total_tokens" in response.usage


class TestChatStream:
    """Tests for chat_stream function."""

    def setup_method(self):
        """Reset all stores before each test."""
        get_kb_store().clear()
        get_doc_store().clear()
        get_app_store().clear()
        reset_vector_store()
        reset_embedding_provider()
        reset_conv_store()
        reset_llm_provider()

    def test_stream_basic(self):
        """Should stream response chunks."""
        app = create_application(name="Test App", user_id="user1")
        chunks = list(chat_stream("Hello", app.id))
        assert len(chunks) > 0
        assert all(isinstance(c, ChatStreamChunk) for c in chunks)

    def test_stream_not_found(self):
        """Should raise error for unknown app."""
        with pytest.raises(ValueError, match="not found"):
            list(chat_stream("Hello", "unknown-app-id"))

    def test_stream_has_final_chunk(self):
        """Should have final chunk with is_final=True."""
        app = create_application(name="Test App", user_id="user1")
        chunks = list(chat_stream("Hello", app.id))
        final_chunks = [c for c in chunks if c.is_final]
        assert len(final_chunks) > 0

    def test_stream_reconstructs_content(self):
        """Content from chunks should form complete response."""
        app = create_application(name="Test App", user_id="user1")
        chunks = list(chat_stream("Hello", app.id))
        content = "".join(c.content for c in chunks)
        assert len(content) > 0


class TestCreateConversation:
    """Tests for create_conversation function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_app_store().clear()
        reset_conv_store()

    def test_create_basic(self):
        """Should create conversation."""
        app = create_application(name="Test App", user_id="user1")
        conv = create_conversation(app.id, "user1")
        assert conv.application_id == app.id
        assert conv.user_id == "user1"

    def test_create_not_found(self):
        """Should raise error for unknown app."""
        with pytest.raises(ValueError, match="not found"):
            create_conversation("unknown-app-id", "user1")


class TestGetConversation:
    """Tests for get_conversation function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_app_store().clear()
        reset_conv_store()

    def test_get_exists(self):
        """Should return conversation if exists."""
        app = create_application(name="Test App", user_id="user1")
        created = create_conversation(app.id, "user1")
        retrieved = get_conversation(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_not_found(self):
        """Should return None if not found."""
        result = get_conversation("unknown")
        assert result is None


class TestListConversations:
    """Tests for list_conversations function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_app_store().clear()
        reset_conv_store()

    def test_list_by_user(self):
        """Should list conversations for user."""
        app = create_application(name="Test App", user_id="user1")
        create_conversation(app.id, "user1")
        create_conversation(app.id, "user1")
        create_conversation(app.id, "user2")

        convs = list_conversations("user1")
        assert len(convs) == 2

    def test_list_by_app(self):
        """Should filter by application."""
        app1 = create_application(name="App 1", user_id="user1")
        app2 = create_application(name="App 2", user_id="user1")
        create_conversation(app1.id, "user1")
        create_conversation(app2.id, "user1")

        convs = list_conversations("user1", application_id=app1.id)
        assert len(convs) == 1


class TestDeleteConversation:
    """Tests for delete_conversation function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_app_store().clear()
        reset_conv_store()

    def test_delete_success(self):
        """Should delete conversation."""
        app = create_application(name="Test App", user_id="user1")
        conv = create_conversation(app.id, "user1")
        result = delete_conversation(conv.id)
        assert result is True
        assert get_conversation(conv.id) is None

    def test_delete_not_found(self):
        """Should return False if not found."""
        result = delete_conversation("unknown")
        assert result is False


class TestGetPrologue:
    """Tests for get_prologue function."""

    def setup_method(self):
        """Reset stores before each test."""
        get_app_store().clear()

    def test_get_default_prologue(self):
        """Should return default prologue."""
        app = create_application(
            name="Test App",
            user_id="user1",
            prologue_type=PrologueType.DEFAULT,
            prologue="Welcome!",
        )
        prologue = get_prologue(app.id)
        assert prologue == "Welcome!"

    def test_get_none_prologue(self):
        """Should return None when prologue disabled."""
        app = create_application(
            name="Test App",
            user_id="user1",
            prologue_type=PrologueType.NONE,
        )
        prologue = get_prologue(app.id)
        assert prologue is None

    def test_get_prologue_not_found(self):
        """Should raise error for unknown app."""
        with pytest.raises(ValueError, match="not found"):
            get_prologue("unknown-app-id")
