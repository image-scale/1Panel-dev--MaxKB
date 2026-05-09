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


class DocumentStatus:
    """Document processing status states."""
    PENDING = '0'
    STARTED = '1'
    SUCCESS = '2'
    FAILURE = '3'
    REVOKE = '4'
    REVOKED = '5'
    IGNORED = 'n'

    CHOICES = [
        (PENDING, 'Pending'),
        (STARTED, 'Started'),
        (SUCCESS, 'Success'),
        (FAILURE, 'Failure'),
        (REVOKE, 'Revoke'),
        (REVOKED, 'Revoked'),
        (IGNORED, 'Ignored'),
    ]


class HitHandlingMethod:
    """How to handle document hits during retrieval."""
    OPTIMIZATION = 'optimization'
    DIRECTLY_RETURN = 'directly_return'

    CHOICES = [
        (OPTIMIZATION, 'Model Optimization'),
        (DIRECTLY_RETURN, 'Direct Return'),
    ]


class Document(models.Model):
    """Document within a knowledge base."""

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Document ID"
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        related_name='documents',
        verbose_name="Knowledge Base"
    )
    name = models.CharField(
        max_length=150,
        verbose_name="Document Name",
        db_index=True
    )
    char_length = models.IntegerField(
        verbose_name="Character Length",
        default=0
    )
    status = models.CharField(
        verbose_name='Status',
        max_length=20,
        choices=DocumentStatus.CHOICES,
        default=DocumentStatus.PENDING,
        db_index=True
    )
    status_meta = models.JSONField(
        verbose_name="Status Metadata",
        default=dict,
        blank=True
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        verbose_name="Active"
    )
    type = models.IntegerField(
        verbose_name='Type',
        choices=KnowledgeType.CHOICES,
        default=KnowledgeType.BASE,
        db_index=True
    )
    hit_handling_method = models.CharField(
        verbose_name='Hit Handling Method',
        max_length=20,
        choices=HitHandlingMethod.CHOICES,
        default=HitHandlingMethod.OPTIMIZATION
    )
    directly_return_similarity = models.FloatField(
        verbose_name='Direct Return Similarity Threshold',
        default=0.9
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
        db_table = "document"
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.name} ({self.get_status_display()})"

    @classmethod
    def create_document(
        cls,
        knowledge: Knowledge,
        name: str,
        content: str = "",
        doc_type: int = KnowledgeType.BASE,
        hit_handling_method: str = HitHandlingMethod.OPTIMIZATION,
        directly_return_similarity: float = 0.9,
        meta: dict = None
    ) -> 'Document':
        """Create a new document in a knowledge base."""
        document = cls(
            knowledge=knowledge,
            name=name,
            char_length=len(content),
            type=doc_type,
            hit_handling_method=hit_handling_method,
            directly_return_similarity=directly_return_similarity,
            meta=meta or {}
        )
        document.save()
        return document

    def update_status(self, new_status: str, meta: dict = None) -> None:
        """Update document status with optional metadata."""
        self.status = new_status
        if meta:
            self.status_meta.update(meta)
        self.save()

    def activate(self) -> None:
        """Activate the document."""
        self.is_active = True
        self.save()

    def deactivate(self) -> None:
        """Deactivate the document."""
        self.is_active = False
        self.save()


class Paragraph(models.Model):
    """Paragraph within a document for RAG retrieval."""

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Paragraph ID"
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='paragraphs',
        verbose_name="Document"
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        related_name='paragraphs',
        verbose_name="Knowledge Base"
    )
    content = models.TextField(
        verbose_name="Content",
        max_length=102400
    )
    title = models.CharField(
        max_length=256,
        default="",
        blank=True,
        verbose_name="Title",
        db_index=True
    )
    status = models.CharField(
        verbose_name='Status',
        max_length=20,
        choices=DocumentStatus.CHOICES,
        default=DocumentStatus.PENDING,
        db_index=True
    )
    status_meta = models.JSONField(
        verbose_name="Status Metadata",
        default=dict,
        blank=True
    )
    hit_num = models.IntegerField(
        verbose_name="Hit Count",
        default=0
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        verbose_name="Active"
    )
    position = models.IntegerField(
        verbose_name="Position",
        default=0,
        db_index=True
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
        db_table = "paragraph"
        ordering = ['position', '-create_time']

    def __str__(self):
        title_preview = self.title[:30] if self.title else self.content[:30]
        return f"{title_preview}..."

    @classmethod
    def create_paragraph(
        cls,
        document: Document,
        content: str,
        title: str = "",
        position: int = 0
    ) -> 'Paragraph':
        """Create a new paragraph in a document."""
        paragraph = cls(
            document=document,
            knowledge=document.knowledge,
            content=content,
            title=title,
            position=position
        )
        paragraph.save()
        return paragraph

    def update_status(self, new_status: str, meta: dict = None) -> None:
        """Update paragraph status with optional metadata."""
        self.status = new_status
        if meta:
            self.status_meta.update(meta)
        self.save()

    def record_hit(self) -> None:
        """Record a hit on this paragraph."""
        self.hit_num += 1
        self.save()

    def activate(self) -> None:
        """Activate the paragraph."""
        self.is_active = True
        self.save()

    def deactivate(self) -> None:
        """Deactivate the paragraph."""
        self.is_active = False
        self.save()
