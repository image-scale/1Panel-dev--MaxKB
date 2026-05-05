"""
Tests for the LLM provider system.
"""
import pytest

from knowledgebot.providers.llm import (
    MessageRole,
    ChatMessage,
    LLMConfig,
    TokenUsage,
    ChatCompletion,
    ChatCompletionChunk,
    LLMProvider,
    MockLLMProvider,
    OpenAIProvider,
    get_llm_provider,
    set_llm_provider,
    reset_llm_provider,
    create_openai_provider,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_system_role(self):
        """MessageRole should have SYSTEM value."""
        assert MessageRole.SYSTEM.value == "system"

    def test_user_role(self):
        """MessageRole should have USER value."""
        assert MessageRole.USER.value == "user"

    def test_assistant_role(self):
        """MessageRole should have ASSISTANT value."""
        assert MessageRole.ASSISTANT.value == "assistant"


class TestChatMessage:
    """Tests for ChatMessage class."""

    def test_message_creation(self):
        """Should create message with role and content."""
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_to_dict(self):
        """Should convert to dictionary."""
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Hi there")
        data = msg.to_dict()
        assert data == {"role": "assistant", "content": "Hi there"}

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {"role": "user", "content": "Test message"}
        msg = ChatMessage.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"

    def test_system_helper(self):
        """Should create system message."""
        msg = ChatMessage.system("You are helpful")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful"

    def test_user_helper(self):
        """Should create user message."""
        msg = ChatMessage.user("What is Python?")
        assert msg.role == MessageRole.USER
        assert msg.content == "What is Python?"

    def test_assistant_helper(self):
        """Should create assistant message."""
        msg = ChatMessage.assistant("Python is a language")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Python is a language"


class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_default_config(self):
        """Should create with default values."""
        config = LLMConfig()
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.top_p == 1.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 100
        assert config.top_p == 0.9

    def test_invalid_temperature_low(self):
        """Should reject temperature below 0."""
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(temperature=-0.1)

    def test_invalid_temperature_high(self):
        """Should reject temperature above 2."""
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(temperature=2.5)

    def test_invalid_top_p_low(self):
        """Should reject top_p below 0."""
        with pytest.raises(ValueError, match="top_p"):
            LLMConfig(top_p=-0.1)

    def test_invalid_top_p_high(self):
        """Should reject top_p above 1."""
        with pytest.raises(ValueError, match="top_p"):
            LLMConfig(top_p=1.5)

    def test_invalid_max_tokens(self):
        """Should reject non-positive max_tokens."""
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(max_tokens=0)

    def test_stop_sequences(self):
        """Should accept stop sequences."""
        config = LLMConfig(stop=["END", "STOP"])
        assert config.stop == ["END", "STOP"]


class TestTokenUsage:
    """Tests for TokenUsage class."""

    def test_default_values(self):
        """Should have default zero values."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        """Should accept custom values."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestChatCompletion:
    """Tests for ChatCompletion class."""

    def test_default_creation(self):
        """Should create with default values."""
        completion = ChatCompletion(content="Hello")
        assert completion.content == "Hello"
        assert completion.role == MessageRole.ASSISTANT
        assert completion.finish_reason == "stop"
        assert completion.id.startswith("chatcmpl-")

    def test_to_dict(self):
        """Should convert to OpenAI-compatible format."""
        completion = ChatCompletion(
            id="chatcmpl-test123",
            content="Test response",
            model="gpt-3.5-turbo",
            usage=TokenUsage(10, 5, 15),
        )
        data = completion.to_dict()
        assert data["id"] == "chatcmpl-test123"
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-3.5-turbo"
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["usage"]["total_tokens"] == 15


class TestChatCompletionChunk:
    """Tests for ChatCompletionChunk class."""

    def test_chunk_creation(self):
        """Should create chunk with content."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            content="Hello",
            model="gpt-3.5-turbo",
        )
        assert chunk.id == "chatcmpl-123"
        assert chunk.content == "Hello"

    def test_chunk_with_role(self):
        """Should include role when present."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            content="",
            role=MessageRole.ASSISTANT,
            model="gpt-3.5-turbo",
        )
        data = chunk.to_dict()
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_chunk_with_finish_reason(self):
        """Should include finish_reason when present."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            content="",
            model="gpt-3.5-turbo",
            finish_reason="stop",
        )
        data = chunk.to_dict()
        assert data["choices"][0]["finish_reason"] == "stop"


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    def test_provider_name(self):
        """Should return 'mock' as provider name."""
        provider = MockLLMProvider()
        assert provider.provider_name == "mock"

    def test_chat_basic(self):
        """Should generate response for basic chat."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Hello")]
        result = provider.chat(messages)
        assert isinstance(result, ChatCompletion)
        assert result.content != ""
        assert result.role == MessageRole.ASSISTANT

    def test_chat_with_config(self):
        """Should use provided config."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Test")]
        config = LLMConfig(model="gpt-4", temperature=0.5)
        result = provider.chat(messages, config)
        assert result.model == "gpt-4"

    def test_chat_default_response(self):
        """Should use default response if set."""
        provider = MockLLMProvider(default_response="Default answer")
        messages = [ChatMessage.user("Any question")]
        result = provider.chat(messages)
        assert result.content == "Default answer"

    def test_chat_records_history(self):
        """Should record call history."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Test")]
        provider.chat(messages)
        assert len(provider.call_history) == 1
        assert provider.call_history[0]["method"] == "chat"

    def test_chat_clear_history(self):
        """Should clear call history."""
        provider = MockLLMProvider()
        provider.chat([ChatMessage.user("Test")])
        provider.clear_history()
        assert len(provider.call_history) == 0

    def test_chat_stream_basic(self):
        """Should stream response chunks."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Hello")]
        chunks = list(provider.chat_stream(messages))
        assert len(chunks) > 0
        assert all(isinstance(c, ChatCompletionChunk) for c in chunks)

    def test_chat_stream_first_chunk_has_role(self):
        """First chunk should have role."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Test")]
        chunks = list(provider.chat_stream(messages))
        assert chunks[0].role == MessageRole.ASSISTANT

    def test_chat_stream_last_chunk_has_finish_reason(self):
        """Last chunk should have finish_reason."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Test")]
        chunks = list(provider.chat_stream(messages))
        assert chunks[-1].finish_reason == "stop"

    def test_chat_stream_content_reconstructs(self):
        """Content from chunks should form complete response."""
        provider = MockLLMProvider(default_response="Hello world!")
        messages = [ChatMessage.user("Test")]
        chunks = list(provider.chat_stream(messages))
        content = "".join(c.content for c in chunks)
        assert content.strip() == "Hello world!"

    def test_chat_question_response(self):
        """Should respond appropriately to questions."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("What is Python?")]
        result = provider.chat(messages)
        assert "question" in result.content.lower() or "analysis" in result.content.lower()

    def test_chat_greeting_response(self):
        """Should respond appropriately to greetings."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Hello!")]
        result = provider.chat(messages)
        assert "hello" in result.content.lower() or "assist" in result.content.lower()

    def test_chat_token_usage(self):
        """Should estimate token usage."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("This is a test message")]
        result = provider.chat(messages)
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens

    def test_count_tokens(self):
        """Should estimate token count."""
        provider = MockLLMProvider()
        count = provider.count_tokens("This is a test")
        assert count > 0


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_provider_name(self):
        """Should return 'openai' as provider name."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider_name == "openai"

    def test_base_url_default(self):
        """Should use default OpenAI base URL."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.base_url == "https://api.openai.com/v1"

    def test_base_url_custom(self):
        """Should accept custom base URL."""
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://custom.api.com/v1/",
        )
        assert provider.base_url == "https://custom.api.com/v1"

    def test_is_llm_provider(self):
        """Should be an instance of LLMProvider."""
        provider = OpenAIProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)


class TestGlobalProvider:
    """Tests for global provider functions."""

    def setup_method(self):
        """Reset provider before each test."""
        reset_llm_provider()

    def test_get_provider_default(self):
        """Should return MockLLMProvider by default."""
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)

    def test_get_provider_singleton(self):
        """Should return same instance."""
        provider1 = get_llm_provider()
        provider2 = get_llm_provider()
        assert provider1 is provider2

    def test_set_provider(self):
        """Should set custom provider."""
        custom = MockLLMProvider(default_response="Custom")
        set_llm_provider(custom)
        provider = get_llm_provider()
        assert provider is custom

    def test_reset_provider(self):
        """Should reset to new default."""
        custom = MockLLMProvider()
        set_llm_provider(custom)
        reset_llm_provider()
        provider = get_llm_provider()
        assert provider is not custom


class TestCreateOpenAIProvider:
    """Tests for create_openai_provider helper."""

    def test_create_with_api_key(self):
        """Should create provider with API key."""
        provider = create_openai_provider(api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openai"

    def test_create_with_custom_url(self):
        """Should create provider with custom URL."""
        provider = create_openai_provider(
            api_key="test-key",
            base_url="https://custom.api.com",
        )
        assert provider.base_url == "https://custom.api.com"


class TestLLMProviderInterface:
    """Tests for the LLMProvider abstract interface."""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_mock_is_provider(self):
        """MockLLMProvider should be an LLMProvider."""
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_openai_is_provider(self):
        """OpenAIProvider should be an LLMProvider."""
        provider = OpenAIProvider(api_key="test")
        assert isinstance(provider, LLMProvider)


class TestConversationFlow:
    """Tests for multi-turn conversation handling."""

    def test_system_message_context(self):
        """Should handle system message in conversation."""
        provider = MockLLMProvider()
        messages = [
            ChatMessage.system("You are a helpful assistant"),
            ChatMessage.user("Hello"),
        ]
        result = provider.chat(messages)
        assert result.content != ""

    def test_multi_turn_conversation(self):
        """Should handle multi-turn conversation."""
        provider = MockLLMProvider()
        messages = [
            ChatMessage.system("You are helpful"),
            ChatMessage.user("Hello"),
            ChatMessage.assistant("Hi there!"),
            ChatMessage.user("How are you?"),
        ]
        result = provider.chat(messages)
        assert result.content != ""
        # Should respond based on last user message
        assert len(provider.call_history) == 1

    def test_deterministic_responses(self):
        """Same input should produce same output."""
        provider = MockLLMProvider()
        messages = [ChatMessage.user("Test message")]
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.5)

        result1 = provider.chat(messages, config)
        result2 = provider.chat(messages, config)

        assert result1.content == result2.content
