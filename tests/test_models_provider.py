"""
Tests for Model Provider models and API endpoints.
"""
import pytest
from django.test import TestCase
from rest_framework.test import APIClient

from apps.users.models import User
from apps.models_provider.models import ModelConfig
from apps.models_provider.base import (
    ModelType, ModelStatus, ModelInfo, ProviderInfo,
    BaseModelCredential, BaseLLMModel, BaseEmbeddingModel,
    IModelProvider, ProviderRegistry
)
from apps.models_provider.providers.openai_provider import (
    OpenAIModelProvider, OpenAICredential, OpenAIChatModel, OpenAIEmbeddingModel
)


class TestModelType(TestCase):
    """Tests for ModelType enum."""

    def test_model_types(self):
        """Test model type values."""
        assert ModelType.LLM.value == 'LLM'
        assert ModelType.EMBEDDING.value == 'EMBEDDING'
        assert ModelType.STT.value == 'STT'
        assert ModelType.TTS.value == 'TTS'
        assert ModelType.IMAGE.value == 'IMAGE'
        assert ModelType.RERANKER.value == 'RERANKER'

    def test_choices(self):
        """Test choices method."""
        choices = ModelType.choices()
        assert len(choices) == 6
        assert ('LLM', 'LLM') in choices


class TestModelStatus(TestCase):
    """Tests for ModelStatus enum."""

    def test_model_status(self):
        """Test model status values."""
        assert ModelStatus.SUCCESS.value == 'SUCCESS'
        assert ModelStatus.ERROR.value == 'ERROR'
        assert ModelStatus.VALIDATING.value == 'VALIDATING'


class TestModelInfo(TestCase):
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        info = ModelInfo(
            name='gpt-4',
            description='GPT-4 model',
            model_type=ModelType.LLM,
            provider='openai',
            default_params={'temperature': 0.7}
        )
        assert info.name == 'gpt-4'
        assert info.model_type == ModelType.LLM
        assert info.provider == 'openai'

    def test_model_info_to_dict(self):
        """Test ModelInfo to_dict."""
        info = ModelInfo(
            name='gpt-4',
            description='GPT-4 model',
            model_type=ModelType.LLM,
            provider='openai'
        )
        d = info.to_dict()
        assert d['name'] == 'gpt-4'
        assert d['model_type'] == 'LLM'


class TestProviderInfo(TestCase):
    """Tests for ProviderInfo dataclass."""

    def test_provider_info_creation(self):
        """Test creating ProviderInfo."""
        info = ProviderInfo(
            provider_id='openai',
            name='OpenAI',
            description='OpenAI models'
        )
        assert info.provider_id == 'openai'
        assert info.name == 'OpenAI'

    def test_provider_info_to_dict(self):
        """Test ProviderInfo to_dict."""
        info = ProviderInfo(
            provider_id='openai',
            name='OpenAI',
            description='OpenAI models'
        )
        d = info.to_dict()
        assert d['provider_id'] == 'openai'
        assert d['name'] == 'OpenAI'


class TestOpenAICredential(TestCase):
    """Tests for OpenAI credential validation."""

    def setUp(self):
        self.credential = OpenAICredential()

    def test_valid_api_key(self):
        """Test validating a valid API key."""
        assert self.credential.validate(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-1234567890abcdef'}
        )

    def test_invalid_api_key_format(self):
        """Test rejecting invalid API key format."""
        assert not self.credential.validate(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'invalid-key'}
        )

    def test_missing_api_key(self):
        """Test rejecting missing API key."""
        assert not self.credential.validate(
            ModelType.LLM, 'gpt-4', {}
        )

    def test_empty_api_key(self):
        """Test rejecting empty API key."""
        assert not self.credential.validate(
            ModelType.LLM, 'gpt-4', {'api_key': ''}
        )

    def test_required_fields(self):
        """Test getting required fields."""
        fields = self.credential.get_required_fields()
        assert 'api_key' in fields

    def test_encrypt_sensitive_fields(self):
        """Test encrypting sensitive fields."""
        creds = {'api_key': 'sk-1234567890abcdef'}
        encrypted = self.credential.encrypt_sensitive_fields(creds)
        assert encrypted['api_key'] == 'sk-***def'


class TestOpenAIChatModel(TestCase):
    """Tests for OpenAI chat model."""

    def test_create_instance(self):
        """Test creating a chat model instance."""
        model = OpenAIChatModel.new_instance(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-test123'}
        )
        assert model.model_name == 'gpt-4'
        assert model.model_type == ModelType.LLM

    def test_chat(self):
        """Test chat method."""
        model = OpenAIChatModel.new_instance(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-test123'}
        )
        messages = [{'role': 'user', 'content': 'Hello'}]
        response = model.chat(messages)
        assert 'gpt-4' in response
        assert '1 messages' in response

    def test_invoke(self):
        """Test invoke method."""
        model = OpenAIChatModel.new_instance(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-test123'}
        )
        response = model.invoke(messages=[{'role': 'user', 'content': 'Hi'}])
        assert response is not None

    def test_stream_chat(self):
        """Test stream chat method."""
        model = OpenAIChatModel.new_instance(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-test123'}
        )
        messages = [{'role': 'user', 'content': 'Hello'}]
        chunks = list(model.stream_chat(messages))
        assert len(chunks) > 0


class TestOpenAIEmbeddingModel(TestCase):
    """Tests for OpenAI embedding model."""

    def test_create_instance(self):
        """Test creating an embedding model instance."""
        model = OpenAIEmbeddingModel.new_instance(
            ModelType.EMBEDDING, 'text-embedding-3-small',
            {'api_key': 'sk-test123'}
        )
        assert model.model_name == 'text-embedding-3-small'

    def test_embed_text(self):
        """Test embedding single text."""
        model = OpenAIEmbeddingModel.new_instance(
            ModelType.EMBEDDING, 'text-embedding-3-small',
            {'api_key': 'sk-test123'},
            dimensions=384
        )
        embedding = model.embed_text('Hello world')
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_texts(self):
        """Test embedding multiple texts."""
        model = OpenAIEmbeddingModel.new_instance(
            ModelType.EMBEDDING, 'text-embedding-3-small',
            {'api_key': 'sk-test123'},
            dimensions=128
        )
        embeddings = model.embed_texts(['Hello', 'World'])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 128


class TestOpenAIModelProvider(TestCase):
    """Tests for OpenAI model provider."""

    def setUp(self):
        self.provider = OpenAIModelProvider()

    def test_get_provider_info(self):
        """Test getting provider info."""
        info = self.provider.get_provider_info()
        assert info.provider_id == 'openai'
        assert info.name == 'OpenAI'

    def test_get_model_types(self):
        """Test getting supported model types."""
        types = self.provider.get_model_types()
        assert ModelType.LLM in types
        assert ModelType.EMBEDDING in types

    def test_get_all_models(self):
        """Test getting all models."""
        models = self.provider.get_models()
        assert len(models) > 0
        llm_models = [m for m in models if m.model_type == ModelType.LLM]
        embedding_models = [m for m in models if m.model_type == ModelType.EMBEDDING]
        assert len(llm_models) > 0
        assert len(embedding_models) > 0

    def test_get_models_by_type(self):
        """Test getting models filtered by type."""
        llm_models = self.provider.get_models(ModelType.LLM)
        assert all(m.model_type == ModelType.LLM for m in llm_models)

    def test_get_model_credential(self):
        """Test getting model credential handler."""
        cred = self.provider.get_model_credential(ModelType.LLM)
        assert isinstance(cred, OpenAICredential)

    def test_get_model(self):
        """Test getting a model instance."""
        model = self.provider.get_model(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-test123'}
        )
        assert isinstance(model, OpenAIChatModel)

    def test_validate_credentials(self):
        """Test validating credentials."""
        assert self.provider.validate_credentials(
            ModelType.LLM, 'gpt-4',
            {'api_key': 'sk-valid123'}
        )


class TestProviderRegistry(TestCase):
    """Tests for ProviderRegistry."""

    def setUp(self):
        ProviderRegistry.clear()

    def tearDown(self):
        ProviderRegistry.clear()

    def test_register_provider(self):
        """Test registering a provider."""
        provider = OpenAIModelProvider()
        ProviderRegistry.register('openai', provider)
        assert ProviderRegistry.get('openai') is provider

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        provider = OpenAIModelProvider()
        ProviderRegistry.register('openai', provider)
        ProviderRegistry.unregister('openai')
        assert ProviderRegistry.get('openai') is None

    def test_list_providers(self):
        """Test listing providers."""
        provider = OpenAIModelProvider()
        ProviderRegistry.register('openai', provider)
        providers = ProviderRegistry.list_providers()
        assert len(providers) == 1
        assert providers[0].provider_id == 'openai'

    def test_get_provider_ids(self):
        """Test getting provider IDs."""
        provider = OpenAIModelProvider()
        ProviderRegistry.register('openai', provider)
        ids = ProviderRegistry.get_provider_ids()
        assert 'openai' in ids


class TestModelConfigModel(TestCase):
    """Tests for ModelConfig model."""

    def setUp(self):
        self.user = User.objects.create(
            username='modeluser',
            password='hashedpassword',
            role='USER'
        )

    def test_create_config(self):
        """Test creating a model configuration."""
        config = ModelConfig.create_config(
            name='My GPT-4',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={'api_key': 'sk-test123'},
            workspace_id='ws-1',
            user=self.user
        )
        assert config.id is not None
        assert config.name == 'My GPT-4'
        assert config.provider == 'openai'
        assert config.model_type == 'LLM'
        assert config.status == 'SUCCESS'
        assert config.is_active is True

    def test_update_credentials(self):
        """Test updating credentials."""
        config = ModelConfig.create_config(
            name='Test Model',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={'api_key': 'old-key'}
        )
        config.update_credentials({'api_key': 'new-key'})
        config.refresh_from_db()
        assert config.credential['api_key'] == 'new-key'

    def test_update_status(self):
        """Test updating status."""
        config = ModelConfig.create_config(
            name='Test Model',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={}
        )
        config.update_status('ERROR', {'error': 'Invalid key'})
        config.refresh_from_db()
        assert config.status == 'ERROR'
        assert config.meta['error'] == 'Invalid key'

    def test_activate_deactivate(self):
        """Test activating and deactivating."""
        config = ModelConfig.create_config(
            name='Test Model',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={}
        )
        config.deactivate()
        assert config.is_active is False
        config.activate()
        assert config.is_active is True

    def test_unique_name_per_workspace(self):
        """Test unique name constraint within workspace."""
        ModelConfig.create_config(
            name='Same Name',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-1'
        )
        with pytest.raises(Exception):
            ModelConfig.create_config(
                name='Same Name',
                provider='openai',
                model_type='LLM',
                model_name='gpt-4',
                credential={},
                workspace_id='ws-1'
            )

    def test_same_name_different_workspaces(self):
        """Test same name allowed in different workspaces."""
        config1 = ModelConfig.create_config(
            name='Same Name',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-1'
        )
        config2 = ModelConfig.create_config(
            name='Same Name',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-2'
        )
        assert config1.id != config2.id


class TestModelConfigAPI(TestCase):
    """Tests for ModelConfig API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='apiuser',
            password='hashedpassword',
            role='USER'
        )
        self.base_url = '/api/workspace/ws-api/model'

    def test_create_model_config(self):
        """Test creating a model config via API."""
        response = self.client.post(
            self.base_url,
            {
                'name': 'API GPT-4',
                'provider': 'openai',
                'model_type': 'LLM',
                'model_name': 'gpt-4',
                'credential': {'api_key': 'sk-apitest123'}
            },
            format='json'
        )
        assert response.status_code == 201
        assert response.data['data']['name'] == 'API GPT-4'

    def test_list_model_configs(self):
        """Test listing model configs."""
        ModelConfig.create_config(
            name='Config 1',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        ModelConfig.create_config(
            name='Config 2',
            provider='openai',
            model_type='EMBEDDING',
            model_name='text-embedding-3-small',
            credential={},
            workspace_id='ws-api'
        )
        response = self.client.get(self.base_url)
        assert response.status_code == 200
        assert response.data['data']['total'] == 2

    def test_filter_by_model_type(self):
        """Test filtering by model type."""
        ModelConfig.create_config(
            name='LLM Config',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        ModelConfig.create_config(
            name='Embedding Config',
            provider='openai',
            model_type='EMBEDDING',
            model_name='text-embedding-3-small',
            credential={},
            workspace_id='ws-api'
        )
        response = self.client.get(f'{self.base_url}?model_type=LLM')
        assert response.data['data']['total'] == 1
        assert response.data['data']['items'][0]['model_type'] == 'LLM'

    def test_filter_by_provider(self):
        """Test filtering by provider."""
        ModelConfig.create_config(
            name='OpenAI Config',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        response = self.client.get(f'{self.base_url}?provider=openai')
        assert response.data['data']['total'] == 1

    def test_get_model_config(self):
        """Test getting a specific model config."""
        config = ModelConfig.create_config(
            name='Get Me',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={'api_key': 'sk-secret123'},
            workspace_id='ws-api'
        )
        response = self.client.get(f'{self.base_url}/{config.id}')
        assert response.status_code == 200
        assert response.data['data']['name'] == 'Get Me'
        assert '***' in response.data['data']['credential_masked']['api_key']

    def test_update_model_config(self):
        """Test updating a model config."""
        config = ModelConfig.create_config(
            name='Original Name',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        response = self.client.put(
            f'{self.base_url}/{config.id}',
            {'name': 'Updated Name'},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['name'] == 'Updated Name'

    def test_delete_model_config(self):
        """Test deleting a model config."""
        config = ModelConfig.create_config(
            name='Delete Me',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        config_id = config.id
        response = self.client.delete(f'{self.base_url}/{config.id}')
        assert response.status_code == 200
        assert not ModelConfig.objects.filter(id=config_id).exists()

    def test_activate_model_config(self):
        """Test activating a model config."""
        config = ModelConfig.create_config(
            name='Activate Me',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        config.deactivate()
        response = self.client.post(f'{self.base_url}/{config.id}/activate')
        assert response.status_code == 200
        assert response.data['data']['is_active'] is True

    def test_deactivate_model_config(self):
        """Test deactivating a model config."""
        config = ModelConfig.create_config(
            name='Deactivate Me',
            provider='openai',
            model_type='LLM',
            model_name='gpt-4',
            credential={},
            workspace_id='ws-api'
        )
        response = self.client.delete(f'{self.base_url}/{config.id}/activate')
        assert response.status_code == 200
        assert response.data['data']['is_active'] is False


class TestProviderAPI(TestCase):
    """Tests for Provider API endpoints."""

    def setUp(self):
        self.client = APIClient()
        ProviderRegistry.clear()
        ProviderRegistry.register('openai', OpenAIModelProvider())

    def tearDown(self):
        ProviderRegistry.clear()

    def test_list_providers(self):
        """Test listing providers."""
        response = self.client.get('/api/providers')
        assert response.status_code == 200
        assert len(response.data['data']) == 1
        assert response.data['data'][0]['provider_id'] == 'openai'

    def test_get_provider_detail(self):
        """Test getting provider detail."""
        response = self.client.get('/api/providers/openai')
        assert response.status_code == 200
        assert response.data['data']['provider']['name'] == 'OpenAI'
        assert 'models' in response.data['data']

    def test_get_provider_not_found(self):
        """Test getting non-existent provider."""
        response = self.client.get('/api/providers/notfound')
        assert response.status_code == 404

    def test_get_provider_models(self):
        """Test getting provider models."""
        response = self.client.get('/api/providers/openai/models')
        assert response.status_code == 200
        assert len(response.data['data']) > 0

    def test_get_provider_models_by_type(self):
        """Test getting provider models filtered by type."""
        response = self.client.get('/api/providers/openai/models?model_type=LLM')
        assert response.status_code == 200
        assert all(m['model_type'] == 'LLM' for m in response.data['data'])

    def test_validate_credentials(self):
        """Test validating credentials."""
        response = self.client.post(
            '/api/providers/validate',
            {
                'provider': 'openai',
                'model_type': 'LLM',
                'model_name': 'gpt-4',
                'credential': {'api_key': 'sk-valid123'}
            },
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['valid'] is True

    def test_validate_invalid_credentials(self):
        """Test validating invalid credentials."""
        response = self.client.post(
            '/api/providers/validate',
            {
                'provider': 'openai',
                'model_type': 'LLM',
                'model_name': 'gpt-4',
                'credential': {'api_key': 'invalid'}
            },
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['valid'] is False
