"""
Application/Agent models for the MaxKB platform.
"""
import uuid
from django.db import models
from apps.users.models import User
from apps.knowledge.models import Knowledge


class ApplicationType:
    """Application types."""
    SIMPLE = 'SIMPLE'
    WORKFLOW = 'WORKFLOW'

    CHOICES = [
        (SIMPLE, 'Simple Chat'),
        (WORKFLOW, 'Workflow'),
    ]


class Application(models.Model):
    """
    AI application/agent configuration.

    Applications define how the AI interacts with users, which knowledge
    bases to use for RAG, prompt configurations, and various settings.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Application ID"
    )
    name = models.CharField(
        max_length=128,
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
        max_length=512,
        verbose_name="Description",
        default="",
        blank=True
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        verbose_name="Owner"
    )
    type = models.CharField(
        verbose_name='Type',
        max_length=20,
        choices=ApplicationType.CHOICES,
        default=ApplicationType.SIMPLE,
        db_index=True
    )
    prologue = models.TextField(
        verbose_name="Opening Message",
        default="",
        blank=True
    )
    system_prompt = models.TextField(
        verbose_name="System Prompt",
        default="",
        blank=True
    )
    is_published = models.BooleanField(
        verbose_name="Published",
        default=False,
        db_index=True
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        verbose_name="Active"
    )
    icon = models.CharField(
        max_length=256,
        verbose_name="Icon URL",
        default="",
        blank=True
    )
    dialogue_count = models.IntegerField(
        verbose_name="Dialogue Count",
        default=0
    )
    knowledge_setting = models.JSONField(
        verbose_name="Knowledge Settings",
        default=dict,
        blank=True
    )
    model_setting = models.JSONField(
        verbose_name="Model Settings",
        default=dict,
        blank=True
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
        db_table = "application"
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"

    @staticmethod
    def get_default_knowledge_setting():
        """Get default knowledge retrieval settings."""
        return {
            'top_n': 3,
            'similarity': 0.6,
            'max_paragraph_chars': 5000,
            'search_mode': 'embedding'
        }

    @staticmethod
    def get_default_system_prompt():
        """Get default system prompt."""
        return (
            "You are a helpful AI assistant. Use the following context to "
            "answer the user's question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "If you don't know the answer based on the context, say so."
        )

    @classmethod
    def create_application(
        cls,
        name: str,
        workspace_id: str = "default",
        description: str = "",
        user: User = None,
        app_type: str = ApplicationType.SIMPLE,
        prologue: str = "",
        system_prompt: str = None,
        icon: str = "",
        knowledge_setting: dict = None,
        model_setting: dict = None
    ) -> 'Application':
        """Create a new application."""
        application = cls(
            name=name,
            workspace_id=workspace_id,
            description=description,
            user=user,
            type=app_type,
            prologue=prologue,
            system_prompt=system_prompt or cls.get_default_system_prompt(),
            icon=icon,
            knowledge_setting=knowledge_setting or cls.get_default_knowledge_setting(),
            model_setting=model_setting or {}
        )
        application.save()
        return application

    def publish(self) -> None:
        """Publish the application."""
        self.is_published = True
        self.save()

    def unpublish(self) -> None:
        """Unpublish the application."""
        self.is_published = False
        self.save()

    def activate(self) -> None:
        """Activate the application."""
        self.is_active = True
        self.save()

    def deactivate(self) -> None:
        """Deactivate the application."""
        self.is_active = False
        self.save()

    def increment_dialogue_count(self) -> None:
        """Increment the dialogue counter."""
        self.dialogue_count += 1
        self.save()


class ApplicationKnowledgeMapping(models.Model):
    """
    Maps applications to knowledge bases for RAG retrieval.

    An application can use multiple knowledge bases for context retrieval.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Mapping ID"
    )
    application = models.ForeignKey(
        Application,
        on_delete=models.CASCADE,
        related_name='knowledge_mappings',
        verbose_name="Application"
    )
    knowledge = models.ForeignKey(
        Knowledge,
        on_delete=models.CASCADE,
        related_name='application_mappings',
        verbose_name="Knowledge Base"
    )
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )

    class Meta:
        db_table = "application_knowledge_mapping"
        unique_together = ['application', 'knowledge']
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.application.name} -> {self.knowledge.name}"

    @classmethod
    def create_mapping(
        cls,
        application: Application,
        knowledge: Knowledge
    ) -> 'ApplicationKnowledgeMapping':
        """Create a mapping between an application and knowledge base."""
        mapping = cls(
            application=application,
            knowledge=knowledge
        )
        mapping.save()
        return mapping

    @classmethod
    def get_knowledge_bases(cls, application_id: str) -> list:
        """Get all knowledge bases for an application."""
        mappings = cls.objects.filter(
            application_id=application_id
        ).select_related('knowledge')
        return [m.knowledge for m in mappings]

    @classmethod
    def get_applications(cls, knowledge_id: str) -> list:
        """Get all applications using a knowledge base."""
        mappings = cls.objects.filter(
            knowledge_id=knowledge_id
        ).select_related('application')
        return [m.application for m in mappings]

    @classmethod
    def delete_by_application(cls, application_id: str) -> int:
        """Delete all mappings for an application."""
        count, _ = cls.objects.filter(application_id=application_id).delete()
        return count

    @classmethod
    def delete_by_knowledge(cls, knowledge_id: str) -> int:
        """Delete all mappings for a knowledge base."""
        count, _ = cls.objects.filter(knowledge_id=knowledge_id).delete()
        return count
