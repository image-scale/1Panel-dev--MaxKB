"""
Models for model provider management.
"""
import uuid
from django.db import models
from apps.users.models import User
from apps.models_provider.base import ModelType, ModelStatus


class ModelConfig(models.Model):
    """
    Configuration for an AI model.

    Stores model credentials and settings for use with applications.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Model Config ID"
    )
    name = models.CharField(
        max_length=128,
        verbose_name="Display Name",
        db_index=True
    )
    workspace_id = models.CharField(
        max_length=64,
        verbose_name="Workspace ID",
        default="default",
        db_index=True
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='model_configs',
        verbose_name="Created By"
    )
    provider = models.CharField(
        max_length=64,
        verbose_name="Provider ID",
        db_index=True
    )
    model_type = models.CharField(
        max_length=20,
        verbose_name="Model Type",
        choices=ModelType.choices(),
        db_index=True
    )
    model_name = models.CharField(
        max_length=128,
        verbose_name="Model Name",
        db_index=True
    )
    status = models.CharField(
        max_length=20,
        verbose_name="Status",
        choices=ModelStatus.choices(),
        default=ModelStatus.SUCCESS.value
    )
    credential = models.JSONField(
        verbose_name="Credentials",
        default=dict
    )
    model_params = models.JSONField(
        verbose_name="Model Parameters",
        default=dict,
        blank=True
    )
    meta = models.JSONField(
        verbose_name="Metadata",
        default=dict,
        blank=True
    )
    is_active = models.BooleanField(
        verbose_name="Active",
        default=True,
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
        db_table = "model_config"
        unique_together = ['name', 'workspace_id']
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.name} ({self.provider}/{self.model_name})"

    @classmethod
    def create_config(
        cls,
        name: str,
        provider: str,
        model_type: str,
        model_name: str,
        credential: dict,
        workspace_id: str = "default",
        user: User = None,
        model_params: dict = None,
        meta: dict = None
    ) -> 'ModelConfig':
        """Create a new model configuration."""
        config = cls(
            name=name,
            provider=provider,
            model_type=model_type,
            model_name=model_name,
            credential=credential,
            workspace_id=workspace_id,
            user=user,
            model_params=model_params or {},
            meta=meta or {}
        )
        config.save()
        return config

    def update_credentials(self, credential: dict) -> None:
        """Update model credentials."""
        self.credential = credential
        self.save()

    def update_status(self, status: str, meta: dict = None) -> None:
        """Update model status."""
        self.status = status
        if meta:
            self.meta = meta
        self.save()

    def activate(self) -> None:
        """Activate the model configuration."""
        self.is_active = True
        self.save()

    def deactivate(self) -> None:
        """Deactivate the model configuration."""
        self.is_active = False
        self.save()

    def get_model_instance(self):
        """Get a model instance using this configuration."""
        from apps.models_provider.base import ProviderRegistry, ModelType as MT

        provider = ProviderRegistry.get(self.provider)
        if not provider:
            raise ValueError(f"Provider not found: {self.provider}")

        model_type = MT(self.model_type)
        return provider.get_model(
            model_type=model_type,
            model_name=self.model_name,
            credentials=self.credential,
            **self.model_params
        )
