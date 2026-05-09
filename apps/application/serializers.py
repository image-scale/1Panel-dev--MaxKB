"""
Serializers for Application models.
"""
from rest_framework import serializers
from .models import Application, ApplicationKnowledgeMapping, ApplicationType


class ApplicationSerializer(serializers.ModelSerializer):
    """Serializer for reading application data."""
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    user_name = serializers.SerializerMethodField()
    knowledge_count = serializers.SerializerMethodField()

    class Meta:
        model = Application
        fields = [
            'id', 'name', 'workspace_id', 'description', 'user', 'user_name',
            'type', 'type_display', 'prologue', 'system_prompt',
            'is_published', 'is_active', 'icon', 'dialogue_count',
            'knowledge_setting', 'model_setting', 'meta',
            'knowledge_count', 'create_time', 'update_time'
        ]
        read_only_fields = ['id', 'create_time', 'update_time']

    def get_user_name(self, obj):
        return obj.user.nick_name if obj.user else None

    def get_knowledge_count(self, obj):
        return obj.knowledge_mappings.count()


class ApplicationCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating an application."""

    class Meta:
        model = Application
        fields = [
            'name', 'workspace_id', 'description', 'user', 'type',
            'prologue', 'system_prompt', 'icon', 'knowledge_setting',
            'model_setting', 'meta'
        ]

    def validate_name(self, value):
        if not value or not value.strip():
            raise serializers.ValidationError("Name cannot be empty")
        return value


class ApplicationUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating an application."""

    class Meta:
        model = Application
        fields = [
            'name', 'description', 'type', 'prologue', 'system_prompt',
            'is_published', 'is_active', 'icon', 'knowledge_setting',
            'model_setting', 'meta'
        ]

    def validate_name(self, value):
        if value is not None and not value.strip():
            raise serializers.ValidationError("Name cannot be empty")
        return value

    def validate_type(self, value):
        valid_types = [choice[0] for choice in ApplicationType.CHOICES]
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Type must be one of {valid_types}"
            )
        return value


class ApplicationKnowledgeMappingSerializer(serializers.ModelSerializer):
    """Serializer for reading application-knowledge mapping."""
    application_name = serializers.SerializerMethodField()
    knowledge_name = serializers.SerializerMethodField()

    class Meta:
        model = ApplicationKnowledgeMapping
        fields = [
            'id', 'application', 'application_name',
            'knowledge', 'knowledge_name', 'create_time'
        ]
        read_only_fields = ['id', 'create_time']

    def get_application_name(self, obj):
        return obj.application.name if obj.application else None

    def get_knowledge_name(self, obj):
        return obj.knowledge.name if obj.knowledge else None


class ApplicationKnowledgeMappingCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating an application-knowledge mapping."""

    class Meta:
        model = ApplicationKnowledgeMapping
        fields = ['application', 'knowledge']
