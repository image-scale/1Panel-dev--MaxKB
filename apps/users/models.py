"""
User model for MaxKB platform.
"""
import hashlib
import uuid
from django.db import models


def generate_password_hash(raw_password: str) -> str:
    """Generate SHA-256 hash for a password."""
    return hashlib.sha256(raw_password.encode('utf-8')).hexdigest()


class UserSource:
    LOCAL = 'LOCAL'
    LDAP = 'LDAP'
    OAUTH = 'OAUTH'

    CHOICES = [
        (LOCAL, 'Local'),
        (LDAP, 'LDAP'),
        (OAUTH, 'OAuth'),
    ]


class UserRole:
    ADMIN = 'ADMIN'
    USER = 'USER'

    CHOICES = [
        (ADMIN, 'Administrator'),
        (USER, 'User'),
    ]


class User(models.Model):
    """User account for the MaxKB platform."""

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="User ID"
    )
    email = models.EmailField(
        unique=True,
        null=True,
        blank=True,
        verbose_name="Email"
    )
    phone = models.CharField(
        max_length=20,
        default="",
        blank=True,
        verbose_name="Phone"
    )
    nick_name = models.CharField(
        max_length=150,
        unique=True,
        verbose_name="Nickname"
    )
    username = models.CharField(
        max_length=150,
        unique=True,
        verbose_name="Username"
    )
    password = models.CharField(
        max_length=150,
        verbose_name="Password Hash"
    )
    role = models.CharField(
        max_length=50,
        choices=UserRole.CHOICES,
        default=UserRole.USER,
        verbose_name="Role"
    )
    source = models.CharField(
        max_length=20,
        choices=UserSource.CHOICES,
        default=UserSource.LOCAL,
        verbose_name="Source"
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name="Active"
    )
    language = models.CharField(
        max_length=10,
        null=True,
        blank=True,
        default=None,
        verbose_name="Language"
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
        db_table = "user"
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.username} ({self.nick_name})"

    def set_password(self, raw_password: str) -> None:
        """Set the user's password using secure hashing."""
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Verify a password against the stored hash."""
        return self.password == generate_password_hash(raw_password)

    @classmethod
    def create_user(
        cls,
        username: str,
        password: str,
        nick_name: str,
        role: str = UserRole.USER,
        email: str = None,
        phone: str = "",
        source: str = UserSource.LOCAL,
        language: str = None
    ) -> 'User':
        """Create a new user with hashed password."""
        user = cls(
            username=username,
            nick_name=nick_name,
            role=role,
            email=email,
            phone=phone,
            source=source,
            language=language
        )
        user.set_password(password)
        user.save()
        return user
