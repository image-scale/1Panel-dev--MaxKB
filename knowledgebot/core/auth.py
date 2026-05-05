"""
Authentication utilities for the KnowledgeBot platform.

Provides password hashing, JWT token generation/verification, and user
authentication functions.
"""
import hashlib
import hmac
import json
import base64
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledgebot.core.models import User

# Secret key for JWT signing (in production, use environment variable)
JWT_SECRET = "knowledgebot-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_SECONDS = 86400  # 24 hours


def password_encrypt(raw_password: str) -> str:
    """
    Encrypt a password using MD5 hashing.

    Args:
        raw_password: The plaintext password to encrypt

    Returns:
        The MD5 hash of the password as a hexadecimal string
    """
    md5 = hashlib.md5()
    md5.update(raw_password.encode('utf-8'))
    return md5.hexdigest()


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _base64url_decode(data: str) -> bytes:
    """Decode base64url string to bytes."""
    # Add padding if needed
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data.encode('utf-8'))


def generate_token(user: "User") -> str:
    """
    Generate a JWT token for the given user.

    Args:
        user: The user to generate a token for

    Returns:
        A JWT token string
    """
    # Create header
    header = {
        "alg": JWT_ALGORITHM,
        "typ": "JWT"
    }

    # Create payload
    current_time = int(time.time())
    payload = {
        "user_id": user.id,
        "username": user.username,
        "role": user.role.value,
        "iat": current_time,
        "exp": current_time + JWT_EXPIRATION_SECONDS
    }

    # Encode header and payload
    header_encoded = _base64url_encode(json.dumps(header).encode('utf-8'))
    payload_encoded = _base64url_encode(json.dumps(payload).encode('utf-8'))

    # Create signature
    message = f"{header_encoded}.{payload_encoded}"
    signature = hmac.new(
        JWT_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    signature_encoded = _base64url_encode(signature)

    # Return complete token
    return f"{header_encoded}.{payload_encoded}.{signature_encoded}"


def verify_token(token: str) -> Optional[str]:
    """
    Verify a JWT token and return the user_id if valid.

    Args:
        token: The JWT token to verify

    Returns:
        The user_id from the token if valid, None otherwise
    """
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None

        header_encoded, payload_encoded, signature_encoded = parts

        # Verify signature
        message = f"{header_encoded}.{payload_encoded}"
        expected_signature = hmac.new(
            JWT_SECRET.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        expected_signature_encoded = _base64url_encode(expected_signature)

        if not hmac.compare_digest(signature_encoded, expected_signature_encoded):
            return None

        # Decode and validate payload
        payload = json.loads(_base64url_decode(payload_encoded))

        # Check expiration
        current_time = int(time.time())
        if payload.get('exp', 0) < current_time:
            return None

        return payload.get('user_id')

    except Exception:
        return None


def create_user(
    username: str,
    email: str,
    password: str,
    role: Optional["UserRole"] = None
) -> "User":
    """
    Create a new user in the system.

    Args:
        username: The username for the new user
        email: The email address for the new user
        password: The plaintext password for the new user
        role: The role for the new user (defaults to USER)

    Returns:
        The newly created User object

    Raises:
        ValueError: If the username or email already exists
    """
    from knowledgebot.core.models import User, UserRole, get_user_store

    store = get_user_store()

    if store.exists_username(username):
        raise ValueError(f"Username '{username}' already exists")

    if store.exists_email(email):
        raise ValueError(f"Email '{email}' already exists")

    user = User(
        username=username,
        email=email,
        role=role or UserRole.USER
    )
    user.set_password(password)

    return store.add(user)


def authenticate_user(username: str, password: str) -> Optional["User"]:
    """
    Authenticate a user with username and password.

    Args:
        username: The username to authenticate
        password: The plaintext password to check

    Returns:
        The User object if authentication succeeds, None otherwise
    """
    from knowledgebot.core.models import get_user_store

    store = get_user_store()
    user = store.get_by_username(username)

    if user is None:
        return None

    if not user.is_active:
        return None

    if not user.check_password(password):
        return None

    return user


def get_user_by_token(token: str) -> Optional["User"]:
    """
    Get a user from a JWT token.

    Args:
        token: The JWT token

    Returns:
        The User object if token is valid, None otherwise
    """
    from knowledgebot.core.models import get_user_store

    user_id = verify_token(token)
    if user_id is None:
        return None

    store = get_user_store()
    return store.get_by_id(user_id)
