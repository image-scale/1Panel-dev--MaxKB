"""
OpenAI model provider implementation.
"""
from typing import Dict, List, Any, Optional

from apps.models_provider.base import (
    IModelProvider, BaseModel, BaseLLMModel, BaseEmbeddingModel,
    BaseModelCredential, ModelType, ModelInfo, ProviderInfo
)


class OpenAICredential(BaseModelCredential):
    """Credential handler for OpenAI models."""

    def validate(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any]
    ) -> bool:
        api_key = credentials.get('api_key', '')
        if not api_key:
            return False
        if not api_key.startswith('sk-') and not api_key.startswith('org-'):
            return False
        return True

    def get_required_fields(self) -> List[str]:
        return ['api_key']


class OpenAIChatModel(BaseLLMModel):
    """OpenAI chat model implementation."""

    def __init__(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ):
        super().__init__(model_type, model_name, credentials, **kwargs)
        self.api_key = credentials.get('api_key', '')
        self.api_base = credentials.get('api_base', 'https://api.openai.com/v1')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send chat messages to OpenAI.

        In production, this would call the OpenAI API. For now, returns
        a placeholder indicating the model would be called.
        """
        return f"[OpenAI {self.model_name}] Would process {len(messages)} messages"

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Stream chat responses from OpenAI."""
        response = self.chat(messages, temperature, max_tokens, **kwargs)
        for chunk in response.split():
            yield chunk + " "

    @classmethod
    def new_instance(
        cls,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ) -> 'OpenAIChatModel':
        return cls(model_type, model_name, credentials, **kwargs)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ):
        super().__init__(model_type, model_name, credentials, **kwargs)
        self.api_key = credentials.get('api_key', '')
        self.api_base = credentials.get('api_base', 'https://api.openai.com/v1')
        self.dimensions = kwargs.get('dimensions', 1536)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        In production, this would call the OpenAI API. For now, returns
        a placeholder vector.
        """
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val >> i & 0xFF) / 255.0 for i in range(self.dimensions)]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]

    @classmethod
    def new_instance(
        cls,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ) -> 'OpenAIEmbeddingModel':
        return cls(model_type, model_name, credentials, **kwargs)


class OpenAIModelProvider(IModelProvider):
    """OpenAI model provider."""

    PROVIDER_ID = 'openai'

    LLM_MODELS = [
        ModelInfo(
            name='gpt-4o',
            description='Latest GPT-4o model, fast and capable',
            model_type=ModelType.LLM,
            provider=PROVIDER_ID,
            default_params={'temperature': 0.7, 'max_tokens': 4096}
        ),
        ModelInfo(
            name='gpt-4o-mini',
            description='Smaller, faster GPT-4o variant',
            model_type=ModelType.LLM,
            provider=PROVIDER_ID,
            default_params={'temperature': 0.7, 'max_tokens': 4096}
        ),
        ModelInfo(
            name='gpt-4-turbo',
            description='GPT-4 Turbo with 128k context',
            model_type=ModelType.LLM,
            provider=PROVIDER_ID,
            default_params={'temperature': 0.7, 'max_tokens': 4096}
        ),
        ModelInfo(
            name='gpt-3.5-turbo',
            description='Fast and cost-effective model',
            model_type=ModelType.LLM,
            provider=PROVIDER_ID,
            default_params={'temperature': 0.7, 'max_tokens': 4096}
        ),
    ]

    EMBEDDING_MODELS = [
        ModelInfo(
            name='text-embedding-3-large',
            description='Best quality embedding model',
            model_type=ModelType.EMBEDDING,
            provider=PROVIDER_ID,
            default_params={'dimensions': 3072}
        ),
        ModelInfo(
            name='text-embedding-3-small',
            description='Efficient embedding model',
            model_type=ModelType.EMBEDDING,
            provider=PROVIDER_ID,
            default_params={'dimensions': 1536}
        ),
        ModelInfo(
            name='text-embedding-ada-002',
            description='Legacy embedding model',
            model_type=ModelType.EMBEDDING,
            provider=PROVIDER_ID,
            default_params={'dimensions': 1536}
        ),
    ]

    def __init__(self):
        self._credential = OpenAICredential()
        self._models = {
            ModelType.LLM: self.LLM_MODELS,
            ModelType.EMBEDDING: self.EMBEDDING_MODELS,
        }
        self._model_classes = {
            ModelType.LLM: OpenAIChatModel,
            ModelType.EMBEDDING: OpenAIEmbeddingModel,
        }

    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_id=self.PROVIDER_ID,
            name='OpenAI',
            description='OpenAI models including GPT-4 and embeddings',
            icon='openai'
        )

    def get_model_types(self) -> List[ModelType]:
        return [ModelType.LLM, ModelType.EMBEDDING]

    def get_models(self, model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        if model_type is None:
            result = []
            for models in self._models.values():
                result.extend(models)
            return result
        return self._models.get(model_type, [])

    def get_model_credential(self, model_type: ModelType) -> BaseModelCredential:
        return self._credential

    def get_model(
        self,
        model_type: ModelType,
        model_name: str,
        credentials: Dict[str, Any],
        **kwargs
    ) -> BaseModel:
        model_class = self._model_classes.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_class.new_instance(model_type, model_name, credentials, **kwargs)
