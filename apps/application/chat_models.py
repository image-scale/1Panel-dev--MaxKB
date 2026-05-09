"""
Chat models for conversation management.
"""
import uuid
from django.db import models
from apps.application.models import Application
from apps.users.models import User


class ChatUserType:
    """Types of chat users."""
    ANONYMOUS = 'ANONYMOUS'
    REGISTERED = 'REGISTERED'
    API_KEY = 'API_KEY'

    CHOICES = [
        (ANONYMOUS, 'Anonymous User'),
        (REGISTERED, 'Registered User'),
        (API_KEY, 'API Key'),
    ]


class MessageRole:
    """Chat message roles."""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

    CHOICES = [
        (SYSTEM, 'System'),
        (USER, 'User'),
        (ASSISTANT, 'Assistant'),
    ]


class VoteStatus:
    """Vote status for chat records."""
    NONE = 'none'
    UP = 'up'
    DOWN = 'down'

    CHOICES = [
        (NONE, 'No Vote'),
        (UP, 'Upvote'),
        (DOWN, 'Downvote'),
    ]


class Chat(models.Model):
    """
    Chat session/conversation with an application.

    A chat represents a conversation thread between a user and an AI application.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Chat ID"
    )
    application = models.ForeignKey(
        Application,
        on_delete=models.CASCADE,
        related_name='chats',
        verbose_name="Application"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chats',
        verbose_name="User"
    )
    chat_user_id = models.CharField(
        max_length=128,
        verbose_name="Chat User ID",
        default="",
        blank=True,
        db_index=True
    )
    chat_user_type = models.CharField(
        max_length=20,
        verbose_name="User Type",
        choices=ChatUserType.CHOICES,
        default=ChatUserType.ANONYMOUS
    )
    title = models.CharField(
        max_length=256,
        verbose_name="Title",
        default="",
        blank=True
    )
    abstract = models.CharField(
        max_length=1024,
        verbose_name="Abstract",
        default="",
        blank=True
    )
    message_count = models.IntegerField(
        verbose_name="Message Count",
        default=0
    )
    is_deleted = models.BooleanField(
        verbose_name="Deleted",
        default=False,
        db_index=True
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
        db_table = "chat"
        ordering = ['-update_time']

    def __str__(self):
        return f"Chat({self.id}) - {self.title or 'Untitled'}"

    @classmethod
    def create_chat(
        cls,
        application: Application,
        user: User = None,
        chat_user_id: str = "",
        chat_user_type: str = ChatUserType.ANONYMOUS,
        title: str = ""
    ) -> 'Chat':
        """Create a new chat session."""
        chat = cls(
            application=application,
            user=user,
            chat_user_id=chat_user_id,
            chat_user_type=chat_user_type,
            title=title
        )
        chat.save()
        application.increment_dialogue_count()
        return chat

    def soft_delete(self) -> None:
        """Soft delete the chat."""
        self.is_deleted = True
        self.save()

    def restore(self) -> None:
        """Restore a soft-deleted chat."""
        self.is_deleted = False
        self.save()

    def update_abstract(self, text: str) -> None:
        """Update the chat abstract from first message."""
        self.abstract = text[:1024] if len(text) > 1024 else text
        if not self.title:
            self.title = text[:256] if len(text) > 256 else text
        self.save()

    def increment_message_count(self) -> None:
        """Increment message counter."""
        self.message_count += 1
        self.save()


class ChatRecord(models.Model):
    """
    Individual message/record within a chat session.

    Stores the question, answer, context, and metadata for each exchange.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Record ID"
    )
    chat = models.ForeignKey(
        Chat,
        on_delete=models.CASCADE,
        related_name='records',
        verbose_name="Chat"
    )
    index = models.IntegerField(
        verbose_name="Index",
        default=0,
        db_index=True
    )
    problem_text = models.TextField(
        verbose_name="Question",
        max_length=10240
    )
    answer_text = models.TextField(
        verbose_name="Answer",
        max_length=40960,
        default="",
        blank=True
    )
    vote_status = models.CharField(
        max_length=10,
        verbose_name="Vote Status",
        choices=VoteStatus.CHOICES,
        default=VoteStatus.NONE
    )
    message_tokens = models.IntegerField(
        verbose_name="Input Tokens",
        default=0
    )
    answer_tokens = models.IntegerField(
        verbose_name="Output Tokens",
        default=0
    )
    run_time = models.FloatField(
        verbose_name="Run Time (seconds)",
        default=0.0
    )
    details = models.JSONField(
        verbose_name="Details",
        default=dict,
        blank=True
    )
    paragraph_ids = models.JSONField(
        verbose_name="Retrieved Paragraph IDs",
        default=list,
        blank=True
    )
    create_time = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )

    class Meta:
        db_table = "chat_record"
        ordering = ['index', 'create_time']

    def __str__(self):
        return f"Record({self.index}): {self.problem_text[:50]}..."

    @classmethod
    def create_record(
        cls,
        chat: Chat,
        problem_text: str,
        answer_text: str = "",
        message_tokens: int = 0,
        answer_tokens: int = 0,
        run_time: float = 0.0,
        paragraph_ids: list = None,
        details: dict = None
    ) -> 'ChatRecord':
        """Create a new chat record."""
        index = chat.records.count()
        record = cls(
            chat=chat,
            index=index,
            problem_text=problem_text,
            answer_text=answer_text,
            message_tokens=message_tokens,
            answer_tokens=answer_tokens,
            run_time=run_time,
            paragraph_ids=paragraph_ids or [],
            details=details or {}
        )
        record.save()
        chat.increment_message_count()
        if index == 0:
            chat.update_abstract(problem_text)
        return record

    def set_answer(
        self,
        answer_text: str,
        answer_tokens: int = 0,
        run_time: float = 0.0,
        paragraph_ids: list = None
    ) -> None:
        """Set the answer for this record."""
        self.answer_text = answer_text
        self.answer_tokens = answer_tokens
        self.run_time = run_time
        if paragraph_ids:
            self.paragraph_ids = paragraph_ids
        self.save()

    def upvote(self) -> None:
        """Upvote this record."""
        self.vote_status = VoteStatus.UP
        self.save()

    def downvote(self) -> None:
        """Downvote this record."""
        self.vote_status = VoteStatus.DOWN
        self.save()

    def clear_vote(self) -> None:
        """Clear vote status."""
        self.vote_status = VoteStatus.NONE
        self.save()


class ChatMessage(models.Model):
    """
    Individual message in chat history.

    Used to track the full conversation including system messages.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Message ID"
    )
    chat = models.ForeignKey(
        Chat,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name="Chat"
    )
    role = models.CharField(
        max_length=20,
        verbose_name="Role",
        choices=MessageRole.CHOICES,
        default=MessageRole.USER
    )
    content = models.TextField(
        verbose_name="Content",
        max_length=40960
    )
    tokens = models.IntegerField(
        verbose_name="Token Count",
        default=0
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

    class Meta:
        db_table = "chat_message"
        ordering = ['create_time']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."

    @classmethod
    def create_message(
        cls,
        chat: Chat,
        role: str,
        content: str,
        tokens: int = 0,
        meta: dict = None
    ) -> 'ChatMessage':
        """Create a new chat message."""
        message = cls(
            chat=chat,
            role=role,
            content=content,
            tokens=tokens,
            meta=meta or {}
        )
        message.save()
        return message

    @classmethod
    def get_history(cls, chat_id: str, limit: int = None) -> list:
        """Get chat message history."""
        queryset = cls.objects.filter(chat_id=chat_id).order_by('create_time')
        if limit:
            queryset = queryset[:limit]
        return list(queryset)

    @classmethod
    def get_formatted_history(cls, chat_id: str, limit: int = None) -> list:
        """Get chat history as list of role/content dicts."""
        messages = cls.get_history(chat_id, limit)
        return [
            {'role': m.role, 'content': m.content}
            for m in messages
        ]
