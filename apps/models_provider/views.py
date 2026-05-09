"""
Views for Model Provider API endpoints.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import ModelConfig
from .base import ProviderRegistry, ModelType
from .serializers import (
    ModelConfigSerializer, ModelConfigCreateSerializer,
    ModelConfigUpdateSerializer, ProviderInfoSerializer,
    ModelInfoSerializer, ValidateCredentialSerializer
)


class ProviderListView(APIView):
    """List available model providers."""

    def get(self, request):
        """Get list of registered providers."""
        providers = ProviderRegistry.list_providers()
        serializer = ProviderInfoSerializer(
            [p.to_dict() for p in providers], many=True
        )
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })


class ProviderDetailView(APIView):
    """Get details of a specific provider."""

    def get(self, request, provider_id):
        """Get provider info and available models."""
        provider = ProviderRegistry.get(provider_id)
        if not provider:
            return Response({
                'code': 404,
                'message': f'Provider not found: {provider_id}'
            }, status=status.HTTP_404_NOT_FOUND)

        info = provider.get_provider_info()
        model_types = provider.get_model_types()
        models = provider.get_models()

        return Response({
            'code': 200,
            'message': 'success',
            'data': {
                'provider': info.to_dict(),
                'model_types': [mt.value for mt in model_types],
                'models': [m.to_dict() for m in models]
            }
        })


class ProviderModelsView(APIView):
    """Get models for a provider, optionally filtered by type."""

    def get(self, request, provider_id):
        """Get models for a provider."""
        provider = ProviderRegistry.get(provider_id)
        if not provider:
            return Response({
                'code': 404,
                'message': f'Provider not found: {provider_id}'
            }, status=status.HTTP_404_NOT_FOUND)

        model_type_str = request.query_params.get('model_type')
        model_type = None
        if model_type_str:
            try:
                model_type = ModelType(model_type_str)
            except ValueError:
                return Response({
                    'code': 400,
                    'message': f'Invalid model type: {model_type_str}'
                }, status=status.HTTP_400_BAD_REQUEST)

        models = provider.get_models(model_type)
        return Response({
            'code': 200,
            'message': 'success',
            'data': [m.to_dict() for m in models]
        })


class ValidateCredentialView(APIView):
    """Validate model credentials."""

    def post(self, request):
        """Validate credentials for a model."""
        serializer = ValidateCredentialSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'code': 400,
                'message': 'Validation error',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        provider = ProviderRegistry.get(data['provider'])
        if not provider:
            return Response({
                'code': 404,
                'message': f"Provider not found: {data['provider']}"
            }, status=status.HTTP_404_NOT_FOUND)

        try:
            model_type = ModelType(data['model_type'])
        except ValueError:
            return Response({
                'code': 400,
                'message': f"Invalid model type: {data['model_type']}"
            }, status=status.HTTP_400_BAD_REQUEST)

        is_valid = provider.validate_credentials(
            model_type, data['model_name'], data['credential']
        )

        return Response({
            'code': 200,
            'message': 'success',
            'data': {'valid': is_valid}
        })


class ModelConfigListView(APIView):
    """List model configurations or create a new one."""

    def get(self, request, workspace_id):
        """Get list of model configurations."""
        provider = request.query_params.get('provider')
        model_type = request.query_params.get('model_type')
        is_active = request.query_params.get('is_active')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))

        queryset = ModelConfig.objects.filter(workspace_id=workspace_id)

        if provider:
            queryset = queryset.filter(provider=provider)
        if model_type:
            queryset = queryset.filter(model_type=model_type)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')

        total = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        configs = queryset[start:end]

        serializer = ModelConfigSerializer(configs, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': {
                'items': serializer.data,
                'total': total,
                'page': page,
                'page_size': page_size
            }
        })

    def post(self, request, workspace_id):
        """Create a new model configuration."""
        data = request.data.copy()
        serializer = ModelConfigCreateSerializer(data=data)
        if serializer.is_valid():
            config = ModelConfig.create_config(
                workspace_id=workspace_id,
                **serializer.validated_data
            )
            return Response({
                'code': 201,
                'message': 'Model configuration created successfully',
                'data': ModelConfigSerializer(config).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class ModelConfigDetailView(APIView):
    """Retrieve, update, or delete a model configuration."""

    def get(self, request, workspace_id, config_id):
        """Get model configuration by ID."""
        config = get_object_or_404(
            ModelConfig, id=config_id, workspace_id=workspace_id
        )
        serializer = ModelConfigSerializer(config)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, workspace_id, config_id):
        """Update model configuration."""
        config = get_object_or_404(
            ModelConfig, id=config_id, workspace_id=workspace_id
        )
        serializer = ModelConfigUpdateSerializer(
            config, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response({
                'code': 200,
                'message': 'Model configuration updated successfully',
                'data': ModelConfigSerializer(config).data
            })
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, workspace_id, config_id):
        """Delete model configuration."""
        config = get_object_or_404(
            ModelConfig, id=config_id, workspace_id=workspace_id
        )
        config.delete()
        return Response({
            'code': 200,
            'message': 'Model configuration deleted successfully'
        })


class ModelConfigActivateView(APIView):
    """Activate or deactivate a model configuration."""

    def post(self, request, workspace_id, config_id):
        """Activate a model configuration."""
        config = get_object_or_404(
            ModelConfig, id=config_id, workspace_id=workspace_id
        )
        config.activate()
        return Response({
            'code': 200,
            'message': 'Model configuration activated',
            'data': ModelConfigSerializer(config).data
        })

    def delete(self, request, workspace_id, config_id):
        """Deactivate a model configuration."""
        config = get_object_or_404(
            ModelConfig, id=config_id, workspace_id=workspace_id
        )
        config.deactivate()
        return Response({
            'code': 200,
            'message': 'Model configuration deactivated',
            'data': ModelConfigSerializer(config).data
        })
