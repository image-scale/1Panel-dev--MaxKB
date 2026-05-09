"""
Serializers for Model Provider models.
"""
from rest_framework import serializers
from .models import ModelConfig
from .base import ModelType, ModelStatus


class ModelConfigSerializer(serializers.ModelSerializer):
    """Serializer for reading model configuration."""
    user_name = serializers.SerializerMethodField()
    credential_masked = serializers.SerializerMethodField()

    class Meta:
        model = ModelConfig
        fields = [
            'id', 'name', 'workspace_id', 'user', 'user_name',
            'provider', 'model_type', 'model_name', 'status',
            'credential_masked', 'model_params', 'meta',
            'is_active', 'create_time', 'update_time'
        ]
        read_only_fields = ['id', 'create_time', 'update_time']

    def get_user_name(self, obj):
        return obj.user.nick_name if obj.user else None

    def get_credential_masked(self, obj):
        from .base import BaseModelCredential
        cred = BaseModelCredential.__subclasses__()[0]() if BaseModelCredential.__subclasses__() else None
        if cred:
            return cred.encrypt_sensitive_fields(obj.credential)
        masked = obj.credential.copy()
        for key in ['api_key', 'secret_key', 'password', 'token']:
            if key in masked and masked[key]:
                value = str(masked[key])
                if len(value) > 6:
                    masked[key] = value[:3] + '***' + value[-3:]
                else:
                    masked[key] = '***'
        return masked


class ModelConfigCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a model configuration."""

    class Meta:
        model = ModelConfig
        fields = [
            'name', 'provider', 'model_type', 'model_name',
            'credential', 'model_params', 'meta', 'user'
        ]

    def validate_name(self, value):
        if not value or not value.strip():
            raise serializers.ValidationError("Name cannot be empty")
        return value

    def validate_model_type(self, value):
        valid_types = [mt.value for mt in ModelType]
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Model type must be one of {valid_types}"
            )
        return value


class ModelConfigUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating a model configuration."""

    class Meta:
        model = ModelConfig
        fields = [
            'name', 'credential', 'model_params', 'meta', 'is_active'
        ]

    def validate_name(self, value):
        if value is not None and not value.strip():
            raise serializers.ValidationError("Name cannot be empty")
        return value


class ProviderInfoSerializer(serializers.Serializer):
    """Serializer for provider info."""
    provider_id = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField()
    icon = serializers.CharField()


class ModelInfoSerializer(serializers.Serializer):
    """Serializer for model info."""
    name = serializers.CharField()
    description = serializers.CharField()
    model_type = serializers.CharField()
    provider = serializers.CharField()
    default_params = serializers.DictField()


class ValidateCredentialSerializer(serializers.Serializer):
    """Serializer for validating credentials."""
    provider = serializers.CharField()
    model_type = serializers.CharField()
    model_name = serializers.CharField()
    credential = serializers.DictField()
