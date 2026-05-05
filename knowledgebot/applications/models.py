"""
Application and agent configuration models.

This module provides data models for AI applications/agents including:
- Application settings for RAG, prompt templates, and model configuration
- Application model with ownership and knowledge base associations
- In-memory storage for application data
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class ApplicationType(str, Enum):
    """Type of application."""
    SIMPLE = "simple"  # Basic Q&A without RAG
    RAG = "rag"  # RAG-enabled with knowledge base
    AGENT = "agent"  # Full agent capabilities


class PrologueType(str, Enum):
    """Type of opening message."""
    NONE = "none"
    DEFAULT = "default"
    CUSTOM = "custom"


@dataclass
class RAGSettings:
    """Settings for RAG (Retrieval-Augmented Generation)."""
    enabled: bool = True
    top_k: int = 5
    similarity_threshold: float = 0.5
    max_context_length: int = 2000
    show_source: bool = True

    def __post_init__(self):
        """Validate settings."""
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.max_context_length < 100:
            raise ValueError("max_context_length must be at least 100")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_context_length": self.max_context_length,
            "show_source": self.show_source,
        }


@dataclass
class ModelSettings:
    """Settings for the LLM model."""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0

    def __post_init__(self):
        """Validate settings."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive if specified")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


@dataclass
class PromptTemplate:
    """Template for system and user prompts."""
    system_prompt: str = "You are a helpful AI assistant."
    context_template: str = "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    no_context_template: str = "Answer the following question:\n\n{question}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "system_prompt": self.system_prompt,
            "context_template": self.context_template,
            "no_context_template": self.no_context_template,
        }

    def format_with_context(self, question: str, context: str) -> str:
        """Format prompt with context."""
        return self.context_template.format(context=context, question=question)

    def format_without_context(self, question: str) -> str:
        """Format prompt without context."""
        return self.no_context_template.format(question=question)


@dataclass
class Application:
    """An AI application/agent configuration."""
    name: str
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    app_type: ApplicationType = ApplicationType.RAG
    is_public: bool = False
    prologue_type: PrologueType = PrologueType.DEFAULT
    prologue: str = "Hello! How can I help you today?"
    knowledge_base_ids: list[str] = field(default_factory=list)
    rag_settings: RAGSettings = field(default_factory=RAGSettings)
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    prompt_template: PromptTemplate = field(default_factory=PromptTemplate)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "app_type": self.app_type.value,
            "is_public": self.is_public,
            "prologue_type": self.prologue_type.value,
            "prologue": self.prologue,
            "knowledge_base_ids": self.knowledge_base_ids.copy(),
            "rag_settings": self.rag_settings.to_dict(),
            "model_settings": self.model_settings.to_dict(),
            "prompt_template": self.prompt_template.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def add_knowledge_base(self, kb_id: str) -> bool:
        """
        Add a knowledge base to this application.

        Returns True if added, False if already present.
        """
        if kb_id not in self.knowledge_base_ids:
            self.knowledge_base_ids.append(kb_id)
            self.updated_at = datetime.utcnow()
            return True
        return False

    def remove_knowledge_base(self, kb_id: str) -> bool:
        """
        Remove a knowledge base from this application.

        Returns True if removed, False if not present.
        """
        if kb_id in self.knowledge_base_ids:
            self.knowledge_base_ids.remove(kb_id)
            self.updated_at = datetime.utcnow()
            return True
        return False


class ApplicationStore:
    """In-memory storage for applications."""

    def __init__(self):
        """Initialize empty store."""
        self._apps: dict[str, Application] = {}
        self._user_index: dict[str, list[str]] = {}

    def add(self, app: Application) -> None:
        """Add an application to the store."""
        self._apps[app.id] = app
        if app.user_id not in self._user_index:
            self._user_index[app.user_id] = []
        if app.id not in self._user_index[app.user_id]:
            self._user_index[app.user_id].append(app.id)

    def get_by_id(self, app_id: str) -> Optional[Application]:
        """Get an application by ID."""
        return self._apps.get(app_id)

    def get_by_name(self, name: str, user_id: str) -> Optional[Application]:
        """Get an application by name within a user's apps."""
        for app_id in self._user_index.get(user_id, []):
            app = self._apps.get(app_id)
            if app and app.name == name:
                return app
        return None

    def exists_name(self, name: str, user_id: str) -> bool:
        """Check if an application name exists for a user."""
        return self.get_by_name(name, user_id) is not None

    def list_by_user(self, user_id: str) -> list[Application]:
        """List all applications for a user."""
        app_ids = self._user_index.get(user_id, [])
        return [self._apps[aid] for aid in app_ids if aid in self._apps]

    def list_public(self) -> list[Application]:
        """List all public applications."""
        return [app for app in self._apps.values() if app.is_public]

    def update(self, app: Application) -> bool:
        """Update an existing application."""
        if app.id in self._apps:
            app.updated_at = datetime.utcnow()
            self._apps[app.id] = app
            return True
        return False

    def delete(self, app_id: str) -> bool:
        """Delete an application by ID."""
        app = self._apps.get(app_id)
        if app:
            del self._apps[app_id]
            if app.user_id in self._user_index:
                self._user_index[app.user_id] = [
                    aid for aid in self._user_index[app.user_id] if aid != app_id
                ]
            return True
        return False

    def clear(self) -> None:
        """Clear all applications."""
        self._apps.clear()
        self._user_index.clear()


# Global store instance
_app_store: Optional[ApplicationStore] = None


def get_app_store() -> ApplicationStore:
    """Get the global application store instance."""
    global _app_store
    if _app_store is None:
        _app_store = ApplicationStore()
    return _app_store


def reset_app_store() -> None:
    """Reset the global application store."""
    global _app_store
    _app_store = None
