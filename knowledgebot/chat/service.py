"""
Chat service with RAG pipeline.

This module provides the chat functionality that combines:
- Context retrieval from knowledge bases
- Prompt construction using application templates
- LLM response generation
- Conversation history management
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Iterator
import uuid

from knowledgebot.applications.models import Application, get_app_store
from knowledgebot.knowledge.processing import search_knowledge_base
from knowledgebot.providers.llm import (
    ChatMessage,
    ChatCompletion,
    ChatCompletionChunk,
    LLMConfig,
    get_llm_provider,
)


@dataclass
class ContextSource:
    """A source of context retrieved for the chat."""
    content: str
    document_id: str
    knowledge_base_id: str
    score: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "document_id": self.document_id,
            "knowledge_base_id": self.knowledge_base_id,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class ChatResponse:
    """Response from the chat service."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    sources: list[ContextSource] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "usage": self.usage,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChatStreamChunk:
    """A chunk of a streaming chat response."""
    id: str
    content: str
    is_final: bool = False
    sources: Optional[list[ContextSource]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "content": self.content,
            "is_final": self.is_final,
        }
        if self.sources is not None:
            result["sources"] = [s.to_dict() for s in self.sources]
        return result


@dataclass
class ConversationMessage:
    """A message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Conversation:
    """A chat conversation with history."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    application_id: str = ""
    user_id: str = ""
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, role: str, content: str) -> ConversationMessage:
        """Add a message to the conversation."""
        msg = ConversationMessage(role=role, content=content)
        self.messages.append(msg)
        self.updated_at = datetime.utcnow()
        return msg

    def get_history(self, max_messages: int = 10) -> list[ConversationMessage]:
        """Get recent conversation history."""
        return self.messages[-max_messages:] if self.messages else []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "application_id": self.application_id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ConversationStore:
    """In-memory storage for conversations."""

    def __init__(self):
        """Initialize empty store."""
        self._conversations: dict[str, Conversation] = {}

    def add(self, conv: Conversation) -> None:
        """Add a conversation."""
        self._conversations[conv.id] = conv

    def get_by_id(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conv_id)

    def list_by_user(self, user_id: str, app_id: Optional[str] = None) -> list[Conversation]:
        """List conversations for a user, optionally filtered by application."""
        convs = [c for c in self._conversations.values() if c.user_id == user_id]
        if app_id:
            convs = [c for c in convs if c.application_id == app_id]
        return sorted(convs, key=lambda c: c.updated_at, reverse=True)

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all conversations."""
        self._conversations.clear()


# Global conversation store
_conv_store: Optional[ConversationStore] = None


def get_conv_store() -> ConversationStore:
    """Get the global conversation store."""
    global _conv_store
    if _conv_store is None:
        _conv_store = ConversationStore()
    return _conv_store


def reset_conv_store() -> None:
    """Reset the global conversation store."""
    global _conv_store
    _conv_store = None


def retrieve_context(
    query: str,
    app: Application,
) -> list[ContextSource]:
    """
    Retrieve relevant context from the application's knowledge bases.

    Args:
        query: The user's question.
        app: The application configuration.

    Returns:
        List of context sources sorted by relevance.
    """
    if not app.rag_settings.enabled or not app.knowledge_base_ids:
        return []

    # Search knowledge bases
    results = search_knowledge_base(
        query=query,
        knowledge_base_ids=app.knowledge_base_ids,
        top_k=app.rag_settings.top_k,
        min_score=app.rag_settings.similarity_threshold,
    )

    # Convert to ContextSource objects
    sources = []
    for r in results:
        sources.append(ContextSource(
            content=r["text"],
            document_id=r.get("metadata", {}).get("document_id", ""),
            knowledge_base_id=r.get("metadata", {}).get("knowledge_base_id", ""),
            score=r["score"],
            metadata=r.get("metadata", {}),
        ))

    return sources


def build_context_text(
    sources: list[ContextSource],
    max_length: int = 2000,
) -> str:
    """
    Build context text from sources.

    Args:
        sources: List of context sources.
        max_length: Maximum character length for context.

    Returns:
        Formatted context text.
    """
    if not sources:
        return ""

    context_parts = []
    current_length = 0

    for i, source in enumerate(sources, 1):
        part = f"[{i}] {source.content}"
        part_length = len(part) + 2  # Account for newlines

        if current_length + part_length > max_length:
            break

        context_parts.append(part)
        current_length += part_length

    return "\n\n".join(context_parts)


def build_messages(
    query: str,
    app: Application,
    context: str,
    conversation: Optional[Conversation] = None,
    max_history: int = 5,
) -> list[ChatMessage]:
    """
    Build the message list for the LLM.

    Args:
        query: The user's question.
        app: The application configuration.
        context: The retrieved context text.
        conversation: Optional conversation for history.
        max_history: Maximum history messages to include.

    Returns:
        List of ChatMessage objects.
    """
    messages = []

    # System message
    messages.append(ChatMessage.system(app.prompt_template.system_prompt))

    # Add conversation history if available
    if conversation:
        history = conversation.get_history(max_history)
        for msg in history:
            if msg.role == "user":
                messages.append(ChatMessage.user(msg.content))
            elif msg.role == "assistant":
                messages.append(ChatMessage.assistant(msg.content))

    # Format user message with context
    if context:
        user_content = app.prompt_template.format_with_context(query, context)
    else:
        user_content = app.prompt_template.format_without_context(query)

    messages.append(ChatMessage.user(user_content))

    return messages


def chat(
    query: str,
    application_id: str,
    conversation_id: Optional[str] = None,
    user_id: str = "",
) -> ChatResponse:
    """
    Process a chat message and generate a response.

    Args:
        query: The user's question.
        application_id: The application ID to use.
        conversation_id: Optional conversation ID for history.
        user_id: The user's ID.

    Returns:
        ChatResponse with the generated answer and sources.

    Raises:
        ValueError: If application is not found.
    """
    # Get application
    app = get_app_store().get_by_id(application_id)
    if not app:
        raise ValueError(f"Application '{application_id}' not found")

    # Get or create conversation
    conv_store = get_conv_store()
    conversation = None
    if conversation_id:
        conversation = conv_store.get_by_id(conversation_id)

    # Retrieve context
    sources = retrieve_context(query, app)
    context_text = build_context_text(
        sources,
        max_length=app.rag_settings.max_context_length,
    )

    # Build messages
    messages = build_messages(query, app, context_text, conversation)

    # Get LLM config from app settings
    llm_config = LLMConfig(
        model=app.model_settings.model_name,
        temperature=app.model_settings.temperature,
        max_tokens=app.model_settings.max_tokens,
        top_p=app.model_settings.top_p,
    )

    # Generate response
    provider = get_llm_provider()
    completion = provider.chat(messages, llm_config)

    # Update conversation if exists
    if conversation:
        conversation.add_message("user", query)
        conversation.add_message("assistant", completion.content)

    # Build response
    response = ChatResponse(
        content=completion.content,
        sources=sources if app.rag_settings.show_source else [],
        usage={
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        },
    )

    return response


def chat_stream(
    query: str,
    application_id: str,
    conversation_id: Optional[str] = None,
    user_id: str = "",
) -> Iterator[ChatStreamChunk]:
    """
    Process a chat message and stream the response.

    Args:
        query: The user's question.
        application_id: The application ID to use.
        conversation_id: Optional conversation ID for history.
        user_id: The user's ID.

    Yields:
        ChatStreamChunk objects as the response is generated.

    Raises:
        ValueError: If application is not found.
    """
    # Get application
    app = get_app_store().get_by_id(application_id)
    if not app:
        raise ValueError(f"Application '{application_id}' not found")

    # Get or create conversation
    conv_store = get_conv_store()
    conversation = None
    if conversation_id:
        conversation = conv_store.get_by_id(conversation_id)

    # Retrieve context
    sources = retrieve_context(query, app)
    context_text = build_context_text(
        sources,
        max_length=app.rag_settings.max_context_length,
    )

    # Build messages
    messages = build_messages(query, app, context_text, conversation)

    # Get LLM config from app settings
    llm_config = LLMConfig(
        model=app.model_settings.model_name,
        temperature=app.model_settings.temperature,
        max_tokens=app.model_settings.max_tokens,
        top_p=app.model_settings.top_p,
    )

    # Generate streaming response
    provider = get_llm_provider()
    response_id = str(uuid.uuid4())
    full_content = []

    for chunk in provider.chat_stream(messages, llm_config):
        full_content.append(chunk.content)
        yield ChatStreamChunk(
            id=response_id,
            content=chunk.content,
            is_final=chunk.finish_reason is not None,
        )

    # Final chunk with sources
    content = "".join(full_content)
    yield ChatStreamChunk(
        id=response_id,
        content="",
        is_final=True,
        sources=sources if app.rag_settings.show_source else [],
    )

    # Update conversation if exists
    if conversation:
        conversation.add_message("user", query)
        conversation.add_message("assistant", content)


def create_conversation(
    application_id: str,
    user_id: str,
) -> Conversation:
    """
    Create a new conversation.

    Args:
        application_id: The application ID.
        user_id: The user's ID.

    Returns:
        The created Conversation.

    Raises:
        ValueError: If application is not found.
    """
    app = get_app_store().get_by_id(application_id)
    if not app:
        raise ValueError(f"Application '{application_id}' not found")

    conv = Conversation(
        application_id=application_id,
        user_id=user_id,
    )

    get_conv_store().add(conv)
    return conv


def get_conversation(conversation_id: str) -> Optional[Conversation]:
    """
    Get a conversation by ID.

    Args:
        conversation_id: The conversation ID.

    Returns:
        The Conversation if found, None otherwise.
    """
    return get_conv_store().get_by_id(conversation_id)


def list_conversations(
    user_id: str,
    application_id: Optional[str] = None,
) -> list[Conversation]:
    """
    List conversations for a user.

    Args:
        user_id: The user's ID.
        application_id: Optional filter by application.

    Returns:
        List of conversations sorted by most recent.
    """
    return get_conv_store().list_by_user(user_id, application_id)


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation.

    Args:
        conversation_id: The conversation ID.

    Returns:
        True if deleted, False if not found.
    """
    return get_conv_store().delete(conversation_id)


def get_prologue(application_id: str) -> Optional[str]:
    """
    Get the prologue/opening message for an application.

    Args:
        application_id: The application ID.

    Returns:
        The prologue text if configured, None otherwise.

    Raises:
        ValueError: If application is not found.
    """
    app = get_app_store().get_by_id(application_id)
    if not app:
        raise ValueError(f"Application '{application_id}' not found")

    from knowledgebot.applications.models import PrologueType

    if app.prologue_type == PrologueType.NONE:
        return None
    return app.prologue
