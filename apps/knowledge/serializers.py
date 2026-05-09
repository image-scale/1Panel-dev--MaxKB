"""
Serializers for Knowledge Base models.
"""
from rest_framework import serializers
from .models import Knowledge, KnowledgeFolder, KnowledgeType, KnowledgeScope


class KnowledgeFolderSerializer(serializers.ModelSerializer):
    """Serializer for reading folder data."""
    children_count = serializers.SerializerMethodField()

    class Meta:
        model = KnowledgeFolder
        fields = [
            'id', 'name', 'description', 'workspace_id',
            'parent', 'children_count', 'create_time', 'update_time'
        ]
        read_only_fields = ['id', 'create_time', 'update_time']

    def get_children_count(self, obj):
        return obj.children.count()


class KnowledgeFolderCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a folder."""

    class Meta:
        model = KnowledgeFolder
        fields = ['name', 'description', 'workspace_id', 'parent']


class KnowledgeSerializer(serializers.ModelSerializer):
    """Serializer for reading knowledge base data."""
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    scope_display = serializers.CharField(source='get_scope_display', read_only=True)
    folder_name = serializers.SerializerMethodField()
    user_name = serializers.SerializerMethodField()

    class Meta:
        model = Knowledge
        fields = [
            'id', 'name', 'workspace_id', 'description', 'user', 'user_name',
            'type', 'type_display', 'scope', 'scope_display',
            'folder', 'folder_name', 'file_size_limit', 'file_count_limit',
            'meta', 'create_time', 'update_time'
        ]
        read_only_fields = ['id', 'create_time', 'update_time']

    def get_folder_name(self, obj):
        return obj.folder.name if obj.folder else None

    def get_user_name(self, obj):
        return obj.user.username if obj.user else None


class KnowledgeCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a knowledge base."""

    class Meta:
        model = Knowledge
        fields = [
            'name', 'workspace_id', 'description', 'user',
            'type', 'scope', 'folder', 'file_size_limit',
            'file_count_limit', 'meta'
        ]

    def validate_type(self, value):
        valid_types = [choice[0] for choice in KnowledgeType.CHOICES]
        if value not in valid_types:
            raise serializers.ValidationError(f"Type must be one of {valid_types}")
        return value

    def validate_scope(self, value):
        valid_scopes = [choice[0] for choice in KnowledgeScope.CHOICES]
        if value not in valid_scopes:
            raise serializers.ValidationError(f"Scope must be one of {valid_scopes}")
        return value


class KnowledgeUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating a knowledge base."""

    class Meta:
        model = Knowledge
        fields = [
            'name', 'description', 'scope', 'folder',
            'file_size_limit', 'file_count_limit', 'meta'
        ]
