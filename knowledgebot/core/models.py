"""
User model and related data structures for the KnowledgeBot platform.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from knowledgebot.core.auth import password_encrypt


class UserRole(Enum):
    """Enumeration of user roles in the system."""
    ADMIN = "admin"
    USER = "user"


@dataclass
class User:
    """
    User model representing a user in the system.

    Attributes:
        id: Unique identifier for the user
        username: Unique username for login
        email: User's email address
        password_hash: Hashed password (never store plaintext)
        role: User's role (ADMIN or USER)
        is_active: Whether the user account is active
        created_at: Timestamp when user was created
        updated_at: Timestamp when user was last updated
    """
    username: str
    email: str
    password_hash: str = ""
    role: UserRole = UserRole.USER
    is_active: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def set_password(self, raw_password: str) -> None:
        """
        Hash and store the password.

        Args:
            raw_password: The plaintext password to hash and store
        """
        self.password_hash = password_encrypt(raw_password)
        self.updated_at = datetime.utcnow()

    def check_password(self, raw_password: str) -> bool:
        """
        Check if the provided password matches the stored hash.

        Args:
            raw_password: The plaintext password to check

        Returns:
            True if password matches, False otherwise
        """
        return self.password_hash == password_encrypt(raw_password)

    def to_dict(self) -> dict:
        """Convert user to dictionary representation (excluding password)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class UserStore:
    """
    In-memory user storage for managing users.
    In production, this would be backed by a database.
    """

    def __init__(self):
        self._users: dict[str, User] = {}
        self._username_index: dict[str, str] = {}  # username -> id
        self._email_index: dict[str, str] = {}  # email -> id

    def add(self, user: User) -> User:
        """Add a user to the store."""
        self._users[user.id] = user
        self._username_index[user.username] = user.id
        if user.email:
            self._email_index[user.email] = user.id
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by their ID."""
        return self._users.get(user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        """Get a user by their username."""
        user_id = self._username_index.get(username)
        return self._users.get(user_id) if user_id else None

    def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by their email."""
        user_id = self._email_index.get(email)
        return self._users.get(user_id) if user_id else None

    def exists_username(self, username: str) -> bool:
        """Check if a username already exists."""
        return username in self._username_index

    def exists_email(self, email: str) -> bool:
        """Check if an email already exists."""
        return email in self._email_index

    def update(self, user: User) -> User:
        """Update a user in the store."""
        user.updated_at = datetime.utcnow()
        self._users[user.id] = user
        return user

    def delete(self, user_id: str) -> bool:
        """Delete a user from the store."""
        user = self._users.get(user_id)
        if user:
            del self._users[user_id]
            del self._username_index[user.username]
            if user.email and user.email in self._email_index:
                del self._email_index[user.email]
            return True
        return False

    def list_all(self) -> list[User]:
        """List all users."""
        return list(self._users.values())

    def clear(self) -> None:
        """Clear all users from the store."""
        self._users.clear()
        self._username_index.clear()
        self._email_index.clear()


# Global user store instance
_user_store = UserStore()


def get_user_store() -> UserStore:
    """Get the global user store instance."""
    return _user_store
