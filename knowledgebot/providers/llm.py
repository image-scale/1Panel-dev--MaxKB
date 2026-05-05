"""
LLM (Large Language Model) provider interface and implementations.

This module provides an abstract interface for LLM providers and includes:
- OpenAI-compatible provider for API-based models
- Mock provider for testing without external dependencies
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Iterator, Any
import hashlib
import json
import uuid


class MessageRole(str, Enum):
    """Role of a chat message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
        )

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[list[str]] = None

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive if specified")


@dataclass
class TokenUsage:
    """Token usage statistics for a completion."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletion:
    """Response from a chat completion request."""
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    content: str = ""
    role: MessageRole = MessageRole.ASSISTANT
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": int(self.created_at.timestamp()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": self.role.value,
                    "content": self.content,
                },
                "finish_reason": self.finish_reason,
            }],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        }


@dataclass
class ChatCompletionChunk:
    """A chunk of a streaming chat completion."""
    id: str
    content: str
    role: Optional[MessageRole] = None
    model: str = ""
    finish_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        delta = {}
        if self.role:
            delta["role"] = self.role.value
        if self.content:
            delta["content"] = self.content

        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": self.finish_reason,
            }],
            "model": self.model,
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> ChatCompletion:
        """
        Generate a chat completion.

        Args:
            messages: List of chat messages.
            config: Optional LLM configuration.

        Returns:
            ChatCompletion with the generated response.
        """
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Generate a streaming chat completion.

        Args:
            messages: List of chat messages.
            config: Optional LLM configuration.

        Yields:
            ChatCompletionChunk objects as they are generated.
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        This is a simple approximation using word count.
        Override for more accurate provider-specific counting.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Simple approximation: ~4 characters per token on average
        return len(text) // 4 + 1


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Generates deterministic responses based on input for testing purposes.
    """

    def __init__(self, default_response: Optional[str] = None):
        """
        Initialize mock provider.

        Args:
            default_response: Optional default response to return.
        """
        self._default_response = default_response
        self._call_history: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def call_history(self) -> list[dict]:
        """Return history of calls made to this provider."""
        return self._call_history.copy()

    def clear_history(self):
        """Clear the call history."""
        self._call_history.clear()

    def _generate_response(self, messages: list[ChatMessage], config: LLMConfig) -> str:
        """Generate a deterministic response based on input."""
        if self._default_response:
            return self._default_response

        # Generate response based on last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return "I understand. How can I help you?"

        # Generate deterministic response using hash
        hash_input = f"{last_user_msg}:{config.model}:{config.temperature}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()

        # Create varied response based on content
        if "?" in last_user_msg:
            return f"Based on my analysis, here's what I can tell you about your question: {last_user_msg[:50]}..."
        elif any(word in last_user_msg.lower() for word in ["hello", "hi", "hey"]):
            return "Hello! How can I assist you today?"
        elif any(word in last_user_msg.lower() for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help with?"
        else:
            return f"I've processed your request. The key points I identified are related to: {last_user_msg[:30]}..."

    def chat(
        self,
        messages: list[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> ChatCompletion:
        """Generate a mock chat completion."""
        config = config or LLMConfig()

        # Record the call
        self._call_history.append({
            "method": "chat",
            "messages": [m.to_dict() for m in messages],
            "config": {
                "model": config.model,
                "temperature": config.temperature,
            }
        })

        response_content = self._generate_response(messages, config)

        # Estimate tokens
        prompt_text = " ".join(m.content for m in messages)
        prompt_tokens = self.count_tokens(prompt_text)
        completion_tokens = self.count_tokens(response_content)

        return ChatCompletion(
            content=response_content,
            model=config.model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a mock streaming chat completion."""
        config = config or LLMConfig()

        # Record the call
        self._call_history.append({
            "method": "chat_stream",
            "messages": [m.to_dict() for m in messages],
            "config": {
                "model": config.model,
                "temperature": config.temperature,
            }
        })

        response_content = self._generate_response(messages, config)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # First chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            content="",
            role=MessageRole.ASSISTANT,
            model=config.model,
        )

        # Stream content word by word
        words = response_content.split()
        for i, word in enumerate(words):
            content = word if i == 0 else " " + word
            yield ChatCompletionChunk(
                id=completion_id,
                content=content,
                model=config.model,
            )

        # Final chunk with finish reason
        yield ChatCompletionChunk(
            id=completion_id,
            content="",
            model=config.model,
            finish_reason="stop",
        )


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible LLM provider.

    Works with OpenAI API and compatible APIs (Azure, local servers, etc.).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            organization: Optional organization ID.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._organization = organization
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        return headers

    def _build_request_body(
        self,
        messages: list[ChatMessage],
        config: LLMConfig,
        stream: bool = False,
    ) -> dict:
        """Build the request body for chat completion."""
        body = {
            "model": config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": stream,
        }

        if config.max_tokens:
            body["max_tokens"] = config.max_tokens
        if config.stop:
            body["stop"] = config.stop

        return body

    def chat(
        self,
        messages: list[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> ChatCompletion:
        """
        Generate a chat completion via OpenAI API.

        Note: This implementation requires the 'requests' library.
        In production, consider using the official openai library.
        """
        config = config or LLMConfig()

        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for OpenAI provider")

        url = f"{self._base_url}/chat/completions"
        body = self._build_request_body(messages, config, stream=False)

        response = requests.post(
            url,
            headers=self._get_headers(),
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage_data = data.get("usage", {})

        return ChatCompletion(
            id=data["id"],
            content=choice["message"]["content"],
            role=MessageRole(choice["message"]["role"]),
            model=data["model"],
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Generate a streaming chat completion via OpenAI API.

        Note: This implementation requires the 'requests' library.
        """
        config = config or LLMConfig()

        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for OpenAI provider")

        url = f"{self._base_url}/chat/completions"
        body = self._build_request_body(messages, config, stream=True)

        response = requests.post(
            url,
            headers=self._get_headers(),
            json=body,
            timeout=self._timeout,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode("utf-8")
            if not line_text.startswith("data: "):
                continue

            data_str = line_text[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                choice = data["choices"][0]
                delta = choice.get("delta", {})

                yield ChatCompletionChunk(
                    id=data["id"],
                    content=delta.get("content", ""),
                    role=MessageRole(delta["role"]) if "role" in delta else None,
                    model=data.get("model", ""),
                    finish_reason=choice.get("finish_reason"),
                )
            except (json.JSONDecodeError, KeyError):
                continue


# Global provider instance
_llm_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """
    Get the global LLM provider instance.

    Returns a MockLLMProvider by default.
    """
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = MockLLMProvider()
    return _llm_provider


def set_llm_provider(provider: LLMProvider) -> None:
    """Set the global LLM provider instance."""
    global _llm_provider
    _llm_provider = provider


def reset_llm_provider() -> None:
    """Reset the global LLM provider to default."""
    global _llm_provider
    _llm_provider = None


def create_openai_provider(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    **kwargs: Any,
) -> OpenAIProvider:
    """
    Create an OpenAI provider with the given configuration.

    Args:
        api_key: API key for authentication.
        base_url: Base URL for the API.
        **kwargs: Additional configuration options.

    Returns:
        Configured OpenAIProvider instance.
    """
    return OpenAIProvider(api_key=api_key, base_url=base_url, **kwargs)
