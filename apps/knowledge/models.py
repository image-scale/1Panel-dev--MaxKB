"""
Knowledge base models for MaxKB platform.
"""
import uuid
from django.db import models
from apps.users.models import User


class KnowledgeType:
    """Types of knowledge bases."""
    BASE = 0
    WEB = 1
    WORKFLOW = 2

    CHOICES = [
        (BASE, 'General'),
        (WEB, 'Web Site'),
        (WORKFLOW, 'Workflow'),
    ]


class KnowledgeScope:
    """Scope of knowledge base access."""
    SHARED = 'SHARED'
    WORKSPACE = 'WORKSPACE'

    CHOICES = [
        (SHARED, 'Shared'),
        (WORKSPACE, 'Workspace'),
    ]


class KnowledgeFolder(models.Model):
    """Folder for organizing knowledge bases in a hierarchy."""

    id = models.CharField(
        primary_key=True,
        max_length=64,
        editable=False,
        verbose_name="Folder ID"
    )
    name = models.CharField(
        max_length=64,
        verbose_name="Folder Name",
        db_index=True
    )
    description = models.CharField(
        max_length=200,
        null=True,
        blank=True,
        verbose_name="Description"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        verbose_name="Owner"
    )
    workspace_id = models.CharField(
        max_length=64,
        verbose_name="Workspace ID",
        default="default",
        db_index=True
    )
    parent = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='children',
        verbose_name="Parent Folder"
    )
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )
    update_time = models.DateTimeField(
        auto_now=True,
        verbose_name="Updated At"
    )

    class Meta:
        db_table = "knowledge_folder"
        ordering = ['name']

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.id:
            self.id = str(uuid.uuid4())
        super().save(*args, **kwargs)

    def get_ancestors(self):
        """Get all ancestor folders."""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self):
        """Get all descendant folders."""
        descendants = []
        for child in self.children.all():
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants


class Knowledge(models.Model):
    """Knowledge base for storing documents and content."""

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Knowledge Base ID"
    )
    name = models.CharField(
        max_length=150,
        verbose_name="Name",
        db_index=True
    )
    workspace_id = models.CharField(
        max_length=64,
        verbose_name="Workspace ID",
        default="default",
        db_index=True
    )
    description = models.CharField(
        max_length=256,
        default="",
        blank=True,
        verbose_name="Description"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        verbose_name="Owner"
    )
    type = models.IntegerField(
        verbose_name='Type',
        choices=KnowledgeType.CHOICES,
        default=KnowledgeType.BASE,
        db_index=True
    )
    scope = models.CharField(
        max_length=20,
        verbose_name='Scope',
        choices=KnowledgeScope.CHOICES,
        default=KnowledgeScope.WORKSPACE,
        db_index=True
    )
    folder = models.ForeignKey(
        KnowledgeFolder,
        on_delete=models.SET_NULL,
        verbose_name="Folder",
        null=True,
        blank=True
    )
    file_size_limit = models.IntegerField(
        verbose_name="File Size Limit (MB)",
        default=100
    )
    file_count_limit = models.IntegerField(
        verbose_name="File Count Limit",
        default=50
    )
    meta = models.JSONField(
        verbose_name="Metadata",
        default=dict,
        blank=True
    )
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )
    update_time = models.DateTimeField(
        auto_now=True,
        verbose_name="Updated At"
    )

    class Meta:
        db_table = "knowledge"
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"

    @classmethod
    def create_knowledge(
        cls,
        name: str,
        workspace_id: str = "default",
        description: str = "",
        user: User = None,
        kb_type: int = KnowledgeType.BASE,
        scope: str = KnowledgeScope.WORKSPACE,
        folder: KnowledgeFolder = None,
        file_size_limit: int = 100,
        file_count_limit: int = 50,
        meta: dict = None
    ) -> 'Knowledge':
        """Create a new knowledge base."""
        knowledge = cls(
            name=name,
            workspace_id=workspace_id,
            description=description,
            user=user,
            type=kb_type,
            scope=scope,
            folder=folder,
            file_size_limit=file_size_limit,
            file_count_limit=file_count_limit,
            meta=meta or {}
        )
        knowledge.save()
        return knowledge
