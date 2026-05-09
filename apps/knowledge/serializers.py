"""
Serializers for Knowledge Base models.
"""
from rest_framework import serializers
from .models import Knowledge, KnowledgeFolder, KnowledgeType, KnowledgeScope, Document, DocumentStatus, HitHandlingMethod


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


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for reading document data."""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    hit_handling_display = serializers.SerializerMethodField()
    knowledge_name = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            'id', 'knowledge', 'knowledge_name', 'name', 'char_length',
            'status', 'status_display', 'status_meta', 'is_active',
            'type', 'type_display', 'hit_handling_method', 'hit_handling_display',
            'directly_return_similarity', 'meta', 'create_time', 'update_time'
        ]
        read_only_fields = ['id', 'create_time', 'update_time']

    def get_hit_handling_display(self, obj):
        return dict(HitHandlingMethod.CHOICES).get(obj.hit_handling_method)

    def get_knowledge_name(self, obj):
        return obj.knowledge.name if obj.knowledge else None


class DocumentCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a document."""
    content = serializers.CharField(write_only=True, required=False, default="")

    class Meta:
        model = Document
        fields = [
            'knowledge', 'name', 'content', 'type',
            'hit_handling_method', 'directly_return_similarity', 'meta'
        ]

    def create(self, validated_data):
        content = validated_data.pop('content', "")
        validated_data['char_length'] = len(content)
        return super().create(validated_data)

    def validate_hit_handling_method(self, value):
        valid_methods = [choice[0] for choice in HitHandlingMethod.CHOICES]
        if value not in valid_methods:
            raise serializers.ValidationError(
                f"Hit handling method must be one of {valid_methods}"
            )
        return value


class DocumentUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating a document."""

    class Meta:
        model = Document
        fields = [
            'name', 'status', 'status_meta', 'is_active',
            'hit_handling_method', 'directly_return_similarity', 'meta'
        ]

    def validate_status(self, value):
        valid_statuses = [choice[0] for choice in DocumentStatus.CHOICES]
        if value not in valid_statuses:
            raise serializers.ValidationError(
                f"Status must be one of {valid_statuses}"
            )
        return value
