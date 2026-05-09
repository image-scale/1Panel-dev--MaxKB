"""
Base model provider abstractions and interfaces.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field


class ModelType(Enum):
    """Types of AI models supported."""
    LLM = 'LLM'
    EMBEDDING = 'EMBEDDING'
    STT = 'STT'
    TTS = 'TTS'
    IMAGE = 'IMAGE'
    RERANKER = 'RERANKER'

    @classmethod
    def choices(cls):
        return [(member.value, member.value) for member in cls]


class ModelStatus(Enum):
    """Status of a model configuration."""
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'
    VALIDATING = 'VALIDATING'

    @classmethod
    def choices(cls):
        return [(member.value, member.value) for member in cls]


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    description: str
    model_type: ModelType
    provider: str
    default_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type.value,
            'provider': self.provider,
            'default_params': self.default_params
        }


@dataclass
class ProviderInfo:
    """Information about a model provider."""
    provider_id: str
    name: str
    description: str = ""
    icon: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider_id': self.provider_id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon
        }


class BaseModelCredential(ABC):
    """Base class for model credentials."""

    @abstractmethod
    def validate(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any]
    ) -> bool:
        """Validate credentials for a model."""
        pass

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required credential fields."""
        pass

    def encrypt_sensitive_fields(
        self,
        credentials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encrypt sensitive credential fields for storage/display."""
        encrypted = credentials.copy()
        sensitive_fields = ['api_key', 'secret_key', 'password', 'token']
        for key in sensitive_fields:
            if key in encrypted and encrypted[key]:
                value = str(encrypted[key])
                if len(value) > 6:
                    encrypted[key] = value[:3] + '***' + value[-3:]
                else:
                    encrypted[key] = '***'
        return encrypted


class BaseModel(ABC):
    """Base class for AI model implementations."""

    def __init__(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.credentials = credentials
        self.params = kwargs

    @abstractmethod
    def invoke(self, **kwargs) -> Any:
        """Invoke the model."""
        pass

    @classmethod
    @abstractmethod
    def new_instance(
        cls,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ) -> 'BaseModel':
        """Create a new model instance."""
        pass


class BaseLLMModel(BaseModel):
    """Base class for LLM models."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send chat messages and get response."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Stream chat responses."""
        pass

    def invoke(self, **kwargs) -> str:
        messages = kwargs.pop('messages', [])
        return self.chat(messages, **kwargs)


class BaseEmbeddingModel(BaseModel):
    """Base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    def invoke(self, **kwargs) -> List[float]:
        text = kwargs.get('text', '')
        return self.embed_text(text)


class IModelProvider(ABC):
    """Interface for model providers."""

    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Get information about the provider."""
        pass

    @abstractmethod
    def get_model_types(self) -> List[ModelType]:
        """Get supported model types."""
        pass

    @abstractmethod
    def get_models(self, model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        """Get available models, optionally filtered by type."""
        pass

    @abstractmethod
    def get_model_credential(self, model_type: ModelType) -> BaseModelCredential:
        """Get credential handler for a model type."""
        pass

    @abstractmethod
    def get_model(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ) -> BaseModel:
        """Get a model instance."""
        pass

    def validate_credentials(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any]
    ) -> bool:
        """Validate credentials for a model."""
        credential = self.get_model_credential(model_type)
        return credential.validate(model_type, model_name, credentials)


class ProviderRegistry:
    """Registry for model providers."""

    _providers: Dict[str, IModelProvider] = {}

    @classmethod
    def register(cls, provider_id: str, provider: IModelProvider) -> None:
        """Register a provider."""
        cls._providers[provider_id] = provider

    @classmethod
    def unregister(cls, provider_id: str) -> None:
        """Unregister a provider."""
        if provider_id in cls._providers:
            del cls._providers[provider_id]

    @classmethod
    def get(cls, provider_id: str) -> Optional[IModelProvider]:
        """Get a provider by ID."""
        return cls._providers.get(provider_id)

    @classmethod
    def list_providers(cls) -> List[ProviderInfo]:
        """List all registered providers."""
        return [p.get_provider_info() for p in cls._providers.values()]

    @classmethod
    def get_provider_ids(cls) -> List[str]:
        """Get all provider IDs."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        cls._providers.clear()
