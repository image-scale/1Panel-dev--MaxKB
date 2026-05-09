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


class SourceType:
    """Source types for embeddings."""
    PARAGRAPH = 'p'
    PROBLEM = 'q'
    TITLE = 't'

    CHOICES = [
        (PARAGRAPH, 'Paragraph'),
        (PROBLEM, 'Problem'),
        (TITLE, 'Title'),
    ]


class Embedding(models.Model):
    """
    Vector embedding for semantic search.

    Stores vector representations of text chunks for RAG retrieval.
    """

    id = models.CharField(
        primary_key=True,
        max_length=128,
        editable=False,
        verbose_name="Embedding ID"
    )
    source_id = models.CharField(
        max_length=128,
        verbose_name="Source ID",
        db_index=True
    )
    source_type = models.CharField(
        verbose_name='Source Type',
        max_length=5,
        choices=SourceType.CHOICES,
        default=SourceType.PARAGRAPH,
        db_index=True
    )
    is_active = models.BooleanField(
        verbose_name="Active",
        default=True,
        db_index=True
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        verbose_name="Knowledge Base",
        related_name='embeddings'
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        verbose_name="Document",
        related_name='embeddings'
    )
    paragraph = models.ForeignKey(
        Paragraph,
        on_delete=models.CASCADE,
        verbose_name="Paragraph",
        related_name='embeddings'
    )
    embedding = models.JSONField(
        verbose_name="Vector",
        default=list
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
        db_table = "embedding"
        ordering = ['-create_time']

    def __str__(self):
        return f"Embedding({self.source_type}:{self.source_id})"

    def save(self, *args, **kwargs):
        if not self.id:
            self.id = str(uuid.uuid4())
        super().save(*args, **kwargs)

    @classmethod
    def create_embedding(
        cls,
        paragraph: Paragraph,
        vector: list,
        source_type: str = SourceType.PARAGRAPH,
        source_id: str = None,
        is_active: bool = True,
        meta: dict = None
    ) -> 'Embedding':
        """Create a new embedding for a paragraph."""
        embedding = cls(
            source_id=source_id or str(paragraph.id),
            source_type=source_type,
            is_active=is_active,
            knowledge=paragraph.knowledge,
            document=paragraph.document,
            paragraph=paragraph,
            embedding=vector,
            meta=meta or {}
        )
        embedding.save()
        return embedding

    @classmethod
    def batch_create(
        cls,
        embeddings_data: list
    ) -> list:
        """
        Batch create embeddings for efficiency.

        Args:
            embeddings_data: List of dicts with keys:
                - paragraph: Paragraph instance
                - vector: List of floats
                - source_type: Source type (optional)
                - source_id: Source ID (optional)
                - meta: Metadata dict (optional)

        Returns:
            List of created Embedding instances.
        """
        embedding_objects = []
        for data in embeddings_data:
            paragraph = data['paragraph']
            embedding = cls(
                id=str(uuid.uuid4()),
                source_id=data.get('source_id', str(paragraph.id)),
                source_type=data.get('source_type', SourceType.PARAGRAPH),
                is_active=data.get('is_active', True),
                knowledge=paragraph.knowledge,
                document=paragraph.document,
                paragraph=paragraph,
                embedding=data['vector'],
                meta=data.get('meta', {})
            )
            embedding_objects.append(embedding)

        cls.objects.bulk_create(embedding_objects)
        return embedding_objects

    @classmethod
    def delete_by_knowledge(cls, knowledge_id: str) -> int:
        """Delete all embeddings for a knowledge base."""
        count, _ = cls.objects.filter(knowledge_id=knowledge_id).delete()
        return count

    @classmethod
    def delete_by_document(cls, document_id: str) -> int:
        """Delete all embeddings for a document."""
        count, _ = cls.objects.filter(document_id=document_id).delete()
        return count

    @classmethod
    def delete_by_paragraph(cls, paragraph_id: str) -> int:
        """Delete all embeddings for a paragraph."""
        count, _ = cls.objects.filter(paragraph_id=paragraph_id).delete()
        return count

    @classmethod
    def delete_by_source_id(cls, source_id: str, source_type: str = None) -> int:
        """Delete embeddings by source ID."""
        queryset = cls.objects.filter(source_id=source_id)
        if source_type:
            queryset = queryset.filter(source_type=source_type)
        count, _ = queryset.delete()
        return count

    def activate(self) -> None:
        """Activate the embedding."""
        self.is_active = True
        self.save()

    def deactivate(self) -> None:
        """Deactivate the embedding."""
        self.is_active = False
        self.save()


class Problem(models.Model):
    """
    Question/problem linked to paragraphs for Q&A retrieval.

    Problems serve as alternate entry points for finding relevant paragraphs.
    When a user asks a question similar to a stored problem, the linked
    paragraphs can be retrieved.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Problem ID"
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        related_name='problems',
        verbose_name="Knowledge Base"
    )
    content = models.CharField(
        max_length=256,
        verbose_name="Question Content",
        db_index=True
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
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )
    update_time = models.DateTimeField(
        auto_now=True,
        verbose_name="Updated At"
    )

    class Meta:
        db_table = "problem"
        ordering = ['-create_time']

    def __str__(self):
        return self.content[:50]

    @classmethod
    def create_problem(
        cls,
        knowledge: Knowledge,
        content: str
    ) -> 'Problem':
        """Create a new problem/question."""
        problem = cls(
            knowledge=knowledge,
            content=content
        )
        problem.save()
        return problem

    def record_hit(self) -> None:
        """Record a hit on this problem."""
        self.hit_num += 1
        self.save()

    def activate(self) -> None:
        """Activate the problem."""
        self.is_active = True
        self.save()

    def deactivate(self) -> None:
        """Deactivate the problem."""
        self.is_active = False
        self.save()


class ProblemParagraphMapping(models.Model):
    """
    Maps problems to paragraphs for Q&A retrieval.

    A problem can be associated with multiple paragraphs, and a paragraph
    can be linked to multiple problems.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Mapping ID"
    )
    problem = models.ForeignKey(
        Problem,
        on_delete=models.CASCADE,
        related_name='paragraph_mappings',
        verbose_name="Problem"
    )
    paragraph = models.ForeignKey(
        Paragraph,
        on_delete=models.CASCADE,
        related_name='problem_mappings',
        verbose_name="Paragraph"
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='problem_mappings',
        verbose_name="Document"
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        related_name='problem_mappings',
        verbose_name="Knowledge Base"
    )
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )

    class Meta:
        db_table = "problem_paragraph_mapping"
        unique_together = ['problem', 'paragraph']
        ordering = ['-create_time']

    def __str__(self):
        return f"Problem({self.problem_id}) -> Paragraph({self.paragraph_id})"

    @classmethod
    def create_mapping(
        cls,
        problem: 'Problem',
        paragraph: Paragraph
    ) -> 'ProblemParagraphMapping':
        """Create a mapping between a problem and paragraph."""
        mapping = cls(
            problem=problem,
            paragraph=paragraph,
            document=paragraph.document,
            knowledge=paragraph.knowledge
        )
        mapping.save()
        return mapping

    @classmethod
    def get_paragraphs_for_problem(cls, problem_id: str) -> list:
        """Get all paragraphs linked to a problem."""
        mappings = cls.objects.filter(problem_id=problem_id).select_related('paragraph')
        return [m.paragraph for m in mappings]

    @classmethod
    def get_problems_for_paragraph(cls, paragraph_id: str) -> list:
        """Get all problems linked to a paragraph."""
        mappings = cls.objects.filter(paragraph_id=paragraph_id).select_related('problem')
        return [m.problem for m in mappings]

    @classmethod
    def delete_by_problem(cls, problem_id: str) -> int:
        """Delete all mappings for a problem."""
        count, _ = cls.objects.filter(problem_id=problem_id).delete()
        return count

    @classmethod
    def delete_by_paragraph(cls, paragraph_id: str) -> int:
        """Delete all mappings for a paragraph."""
        count, _ = cls.objects.filter(paragraph_id=paragraph_id).delete()
        return count


class Tag(models.Model):
    """
    Tag with key-value pairs for categorizing documents.

    Tags are scoped to knowledge bases. The key-value combination must be
    unique within a knowledge base.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Tag ID"
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        related_name='tags',
        verbose_name="Knowledge Base"
    )
    key = models.CharField(
        max_length=64,
        verbose_name="Tag Key",
        db_index=True
    )
    value = models.CharField(
        max_length=128,
        verbose_name="Tag Value",
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
        db_table = "tag"
        unique_together = ['knowledge', 'key', 'value']
        ordering = ['key', 'value']
        indexes = [
            models.Index(fields=['knowledge', 'key']),
        ]

    def __str__(self):
        return f"{self.key}={self.value}"

    @classmethod
    def create_tag(
        cls,
        knowledge: Knowledge,
        key: str,
        value: str
    ) -> 'Tag':
        """Create a new tag."""
        tag = cls(
            knowledge=knowledge,
            key=key,
            value=value
        )
        tag.save()
        return tag

    @classmethod
    def get_or_create_tag(
        cls,
        knowledge: Knowledge,
        key: str,
        value: str
    ) -> tuple:
        """Get or create a tag, returns (tag, created)."""
        return cls.objects.get_or_create(
            knowledge=knowledge,
            key=key,
            value=value
        )

    @classmethod
    def get_tags_for_knowledge(cls, knowledge_id: str) -> list:
        """Get all tags for a knowledge base."""
        return list(cls.objects.filter(knowledge_id=knowledge_id))

    @classmethod
    def get_unique_keys(cls, knowledge_id: str) -> list:
        """Get unique tag keys for a knowledge base."""
        return list(
            cls.objects.filter(knowledge_id=knowledge_id)
            .values_list('key', flat=True)
            .order_by('key')
            .distinct()
        )

    @classmethod
    def get_values_for_key(cls, knowledge_id: str, key: str) -> list:
        """Get all values for a specific key in a knowledge base."""
        return list(
            cls.objects.filter(knowledge_id=knowledge_id, key=key)
            .values_list('value', flat=True)
        )


class DocumentTag(models.Model):
    """
    Maps tags to documents.

    A document can have multiple tags, and a tag can be applied to
    multiple documents.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Mapping ID"
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='tag_mappings',
        verbose_name="Document"
    )
    tag = models.ForeignKey(
        Tag,
        on_delete=models.CASCADE,
        related_name='document_mappings',
        verbose_name="Tag"
    )
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )

    class Meta:
        db_table = "document_tag"
        unique_together = ['document', 'tag']
        ordering = ['-create_time']

    def __str__(self):
        return f"Doc({self.document_id}) -> Tag({self.tag.key}={self.tag.value})"

    @classmethod
    def create_mapping(
        cls,
        document: Document,
        tag: Tag
    ) -> 'DocumentTag':
        """Create a mapping between a document and tag."""
        mapping = cls(
            document=document,
            tag=tag
        )
        mapping.save()
        return mapping

    @classmethod
    def add_tags_to_document(
        cls,
        document: Document,
        tags: list
    ) -> list:
        """Add multiple tags to a document."""
        mappings = []
        for tag in tags:
            mapping, created = cls.objects.get_or_create(
                document=document,
                tag=tag
            )
            if created:
                mappings.append(mapping)
        return mappings

    @classmethod
    def remove_tags_from_document(
        cls,
        document: Document,
        tags: list
    ) -> int:
        """Remove multiple tags from a document."""
        tag_ids = [t.id for t in tags]
        count, _ = cls.objects.filter(
            document=document,
            tag_id__in=tag_ids
        ).delete()
        return count

    @classmethod
    def get_tags_for_document(cls, document_id: str) -> list:
        """Get all tags for a document."""
        mappings = cls.objects.filter(document_id=document_id).select_related('tag')
        return [m.tag for m in mappings]

    @classmethod
    def get_documents_for_tag(cls, tag_id: str) -> list:
        """Get all documents with a specific tag."""
        mappings = cls.objects.filter(tag_id=tag_id).select_related('document')
        return [m.document for m in mappings]

    @classmethod
    def get_documents_by_tags(
        cls,
        knowledge_id: str,
        tag_filters: list
    ) -> list:
        """
        Get documents matching tag filters.

        Args:
            knowledge_id: Knowledge base ID
            tag_filters: List of dicts with 'key' and optional 'value'

        Returns:
            List of documents matching all tag filters
        """
        from django.db.models import Q, Count

        document_ids = None
        for filter_dict in tag_filters:
            key = filter_dict.get('key')
            value = filter_dict.get('value')

            query = Q(tag__knowledge_id=knowledge_id, tag__key=key)
            if value:
                query &= Q(tag__value=value)

            matching_docs = set(
                cls.objects.filter(query)
                .values_list('document_id', flat=True)
            )

            if document_ids is None:
                document_ids = matching_docs
            else:
                document_ids &= matching_docs

        if not document_ids:
            return []

        return list(Document.objects.filter(id__in=document_ids))
