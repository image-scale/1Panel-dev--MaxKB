"""
Application service for CRUD operations.

This module provides high-level functions for managing AI applications.
"""
from datetime import datetime
from typing import Optional

from knowledgebot.applications.models import (
    Application,
    ApplicationType,
    PrologueType,
    RAGSettings,
    ModelSettings,
    PromptTemplate,
    get_app_store,
)


def create_application(
    name: str,
    user_id: str,
    description: str = "",
    app_type: ApplicationType = ApplicationType.RAG,
    is_public: bool = False,
    prologue_type: PrologueType = PrologueType.DEFAULT,
    prologue: str = "Hello! How can I help you today?",
    knowledge_base_ids: Optional[list[str]] = None,
    rag_settings: Optional[RAGSettings] = None,
    model_settings: Optional[ModelSettings] = None,
    prompt_template: Optional[PromptTemplate] = None,
) -> Application:
    """
    Create a new application.

    Args:
        name: Application name (must be unique per user).
        user_id: ID of the user creating the application.
        description: Optional description.
        app_type: Type of application (simple, rag, agent).
        is_public: Whether the application is publicly accessible.
        prologue_type: Type of opening message.
        prologue: Custom opening message.
        knowledge_base_ids: List of knowledge base IDs to associate.
        rag_settings: RAG configuration settings.
        model_settings: LLM model configuration.
        prompt_template: Prompt template configuration.

    Returns:
        The created Application.

    Raises:
        ValueError: If name already exists for this user.
    """
    store = get_app_store()

    if store.exists_name(name, user_id):
        raise ValueError(f"Application with name '{name}' already exists")

    app = Application(
        name=name,
        user_id=user_id,
        description=description,
        app_type=app_type,
        is_public=is_public,
        prologue_type=prologue_type,
        prologue=prologue,
        knowledge_base_ids=knowledge_base_ids or [],
        rag_settings=rag_settings or RAGSettings(),
        model_settings=model_settings or ModelSettings(),
        prompt_template=prompt_template or PromptTemplate(),
    )

    store.add(app)
    return app


def get_application(app_id: str) -> Optional[Application]:
    """
    Get an application by ID.

    Args:
        app_id: The application ID.

    Returns:
        The Application if found, None otherwise.
    """
    return get_app_store().get_by_id(app_id)


def get_application_by_name(name: str, user_id: str) -> Optional[Application]:
    """
    Get an application by name for a specific user.

    Args:
        name: The application name.
        user_id: The user ID.

    Returns:
        The Application if found, None otherwise.
    """
    return get_app_store().get_by_name(name, user_id)


def list_applications(user_id: str) -> list[Application]:
    """
    List all applications for a user.

    Args:
        user_id: The user ID.

    Returns:
        List of applications owned by the user.
    """
    return get_app_store().list_by_user(user_id)


def list_public_applications() -> list[Application]:
    """
    List all public applications.

    Returns:
        List of public applications.
    """
    return get_app_store().list_public()


def update_application(
    app_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_public: Optional[bool] = None,
    prologue_type: Optional[PrologueType] = None,
    prologue: Optional[str] = None,
    rag_settings: Optional[RAGSettings] = None,
    model_settings: Optional[ModelSettings] = None,
    prompt_template: Optional[PromptTemplate] = None,
) -> Optional[Application]:
    """
    Update an application.

    Args:
        app_id: The application ID.
        name: New name (optional).
        description: New description (optional).
        is_public: New public status (optional).
        prologue_type: New prologue type (optional).
        prologue: New prologue (optional).
        rag_settings: New RAG settings (optional).
        model_settings: New model settings (optional).
        prompt_template: New prompt template (optional).

    Returns:
        The updated Application if found, None otherwise.

    Raises:
        ValueError: If new name already exists for this user.
    """
    store = get_app_store()
    app = store.get_by_id(app_id)

    if not app:
        return None

    if name is not None and name != app.name:
        if store.exists_name(name, app.user_id):
            raise ValueError(f"Application with name '{name}' already exists")
        app.name = name

    if description is not None:
        app.description = description

    if is_public is not None:
        app.is_public = is_public

    if prologue_type is not None:
        app.prologue_type = prologue_type

    if prologue is not None:
        app.prologue = prologue

    if rag_settings is not None:
        app.rag_settings = rag_settings

    if model_settings is not None:
        app.model_settings = model_settings

    if prompt_template is not None:
        app.prompt_template = prompt_template

    app.updated_at = datetime.utcnow()
    store.update(app)
    return app


def delete_application(app_id: str) -> bool:
    """
    Delete an application.

    Args:
        app_id: The application ID.

    Returns:
        True if deleted, False if not found.
    """
    return get_app_store().delete(app_id)


def add_knowledge_base_to_app(app_id: str, kb_id: str) -> Optional[Application]:
    """
    Add a knowledge base to an application.

    Args:
        app_id: The application ID.
        kb_id: The knowledge base ID to add.

    Returns:
        The updated Application if found, None otherwise.
    """
    store = get_app_store()
    app = store.get_by_id(app_id)

    if not app:
        return None

    app.add_knowledge_base(kb_id)
    store.update(app)
    return app


def remove_knowledge_base_from_app(app_id: str, kb_id: str) -> Optional[Application]:
    """
    Remove a knowledge base from an application.

    Args:
        app_id: The application ID.
        kb_id: The knowledge base ID to remove.

    Returns:
        The updated Application if found, None otherwise.
    """
    store = get_app_store()
    app = store.get_by_id(app_id)

    if not app:
        return None

    app.remove_knowledge_base(kb_id)
    store.update(app)
    return app


def duplicate_application(app_id: str, new_name: str, user_id: str) -> Optional[Application]:
    """
    Duplicate an application with a new name.

    Args:
        app_id: The source application ID.
        new_name: Name for the duplicated application.
        user_id: User ID for the new application (can be different).

    Returns:
        The new Application if source found, None otherwise.

    Raises:
        ValueError: If new name already exists for the user.
    """
    store = get_app_store()
    source = store.get_by_id(app_id)

    if not source:
        return None

    return create_application(
        name=new_name,
        user_id=user_id,
        description=source.description,
        app_type=source.app_type,
        is_public=False,  # Duplicates start as private
        prologue_type=source.prologue_type,
        prologue=source.prologue,
        knowledge_base_ids=source.knowledge_base_ids.copy(),
        rag_settings=RAGSettings(
            enabled=source.rag_settings.enabled,
            top_k=source.rag_settings.top_k,
            similarity_threshold=source.rag_settings.similarity_threshold,
            max_context_length=source.rag_settings.max_context_length,
            show_source=source.rag_settings.show_source,
        ),
        model_settings=ModelSettings(
            model_name=source.model_settings.model_name,
            temperature=source.model_settings.temperature,
            max_tokens=source.model_settings.max_tokens,
            top_p=source.model_settings.top_p,
        ),
        prompt_template=PromptTemplate(
            system_prompt=source.prompt_template.system_prompt,
            context_template=source.prompt_template.context_template,
            no_context_template=source.prompt_template.no_context_template,
        ),
    )
