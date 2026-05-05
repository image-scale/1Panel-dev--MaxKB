"""
Tests for the user authentication system.
"""
import pytest
import time
from unittest.mock import patch

from knowledgebot.core.auth import (
    password_encrypt,
    generate_token,
    verify_token,
    create_user,
    authenticate_user,
    get_user_by_token,
    JWT_EXPIRATION_SECONDS,
)
from knowledgebot.core.models import User, UserRole, UserStore, get_user_store


class TestPasswordEncrypt:
    """Tests for password encryption."""

    def test_password_encrypt_returns_hash(self):
        """Password encrypt should return a hash different from input."""
        password = "password123"
        hashed = password_encrypt(password)
        assert hashed != password
        assert len(hashed) == 32  # MD5 produces 32 hex characters

    def test_password_encrypt_is_consistent(self):
        """Same password should produce same hash."""
        password = "password123"
        hash1 = password_encrypt(password)
        hash2 = password_encrypt(password)
        assert hash1 == hash2

    def test_password_encrypt_different_passwords(self):
        """Different passwords should produce different hashes."""
        hash1 = password_encrypt("password123")
        hash2 = password_encrypt("password456")
        assert hash1 != hash2

    def test_password_encrypt_empty_string(self):
        """Empty string should produce valid hash."""
        hashed = password_encrypt("")
        assert len(hashed) == 32


class TestUserModel:
    """Tests for the User model."""

    def test_user_creation(self):
        """User should be created with default values."""
        user = User(username="testuser", email="test@test.com")
        assert user.username == "testuser"
        assert user.email == "test@test.com"
        assert user.role == UserRole.USER
        assert user.is_active is True
        assert user.id is not None
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_user_set_password(self):
        """set_password should hash the password."""
        user = User(username="testuser", email="test@test.com")
        user.set_password("password123")
        assert user.password_hash != "password123"
        assert user.password_hash == password_encrypt("password123")

    def test_user_check_password_correct(self):
        """check_password should return True for correct password."""
        user = User(username="testuser", email="test@test.com")
        user.set_password("password123")
        assert user.check_password("password123") is True

    def test_user_check_password_incorrect(self):
        """check_password should return False for incorrect password."""
        user = User(username="testuser", email="test@test.com")
        user.set_password("password123")
        assert user.check_password("wrongpassword") is False

    def test_user_to_dict(self):
        """to_dict should return user data without password."""
        user = User(username="testuser", email="test@test.com")
        user.set_password("password123")
        data = user.to_dict()
        assert "id" in data
        assert data["username"] == "testuser"
        assert data["email"] == "test@test.com"
        assert data["role"] == "user"
        assert data["is_active"] is True
        assert "password_hash" not in data
        assert "password" not in data


class TestUserRole:
    """Tests for UserRole enum."""

    def test_user_role_admin(self):
        """UserRole should have ADMIN value."""
        assert UserRole.ADMIN.value == "admin"

    def test_user_role_user(self):
        """UserRole should have USER value."""
        assert UserRole.USER.value == "user"


class TestUserStore:
    """Tests for UserStore."""

    def setup_method(self):
        """Reset user store before each test."""
        get_user_store().clear()

    def test_add_and_get_by_id(self):
        """Should add and retrieve user by ID."""
        store = UserStore()
        user = User(username="testuser", email="test@test.com")
        store.add(user)
        retrieved = store.get_by_id(user.id)
        assert retrieved is not None
        assert retrieved.username == "testuser"

    def test_get_by_username(self):
        """Should retrieve user by username."""
        store = UserStore()
        user = User(username="testuser", email="test@test.com")
        store.add(user)
        retrieved = store.get_by_username("testuser")
        assert retrieved is not None
        assert retrieved.id == user.id

    def test_get_by_email(self):
        """Should retrieve user by email."""
        store = UserStore()
        user = User(username="testuser", email="test@test.com")
        store.add(user)
        retrieved = store.get_by_email("test@test.com")
        assert retrieved is not None
        assert retrieved.id == user.id

    def test_exists_username(self):
        """Should check if username exists."""
        store = UserStore()
        user = User(username="testuser", email="test@test.com")
        store.add(user)
        assert store.exists_username("testuser") is True
        assert store.exists_username("nonexistent") is False

    def test_exists_email(self):
        """Should check if email exists."""
        store = UserStore()
        user = User(username="testuser", email="test@test.com")
        store.add(user)
        assert store.exists_email("test@test.com") is True
        assert store.exists_email("nonexistent@test.com") is False


class TestCreateUser:
    """Tests for create_user function."""

    def setup_method(self):
        """Reset user store before each test."""
        get_user_store().clear()

    def test_create_user_success(self):
        """create_user should create and return a new user."""
        user = create_user(
            username="testuser",
            email="test@test.com",
            password="pass123"
        )
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@test.com"
        assert user.check_password("pass123") is True
        assert user.role == UserRole.USER

    def test_create_user_with_role(self):
        """create_user should accept a role parameter."""
        user = create_user(
            username="admin",
            email="admin@test.com",
            password="pass123",
            role=UserRole.ADMIN
        )
        assert user.role == UserRole.ADMIN

    def test_create_user_duplicate_username(self):
        """create_user should raise error for duplicate username."""
        create_user(username="testuser", email="test1@test.com", password="pass123")
        with pytest.raises(ValueError, match="Username 'testuser' already exists"):
            create_user(username="testuser", email="test2@test.com", password="pass123")

    def test_create_user_duplicate_email(self):
        """create_user should raise error for duplicate email."""
        create_user(username="user1", email="test@test.com", password="pass123")
        with pytest.raises(ValueError, match="Email 'test@test.com' already exists"):
            create_user(username="user2", email="test@test.com", password="pass123")


class TestAuthenticateUser:
    """Tests for authenticate_user function."""

    def setup_method(self):
        """Reset user store before each test."""
        get_user_store().clear()

    def test_authenticate_user_success(self):
        """authenticate_user should return user for valid credentials."""
        create_user(username="testuser", email="test@test.com", password="pass123")
        user = authenticate_user("testuser", "pass123")
        assert user is not None
        assert user.username == "testuser"

    def test_authenticate_user_wrong_password(self):
        """authenticate_user should return None for wrong password."""
        create_user(username="testuser", email="test@test.com", password="pass123")
        user = authenticate_user("testuser", "wrongpass")
        assert user is None

    def test_authenticate_user_nonexistent(self):
        """authenticate_user should return None for nonexistent user."""
        user = authenticate_user("nonexistent", "pass123")
        assert user is None

    def test_authenticate_user_inactive(self):
        """authenticate_user should return None for inactive user."""
        user = create_user(username="testuser", email="test@test.com", password="pass123")
        user.is_active = False
        get_user_store().update(user)
        result = authenticate_user("testuser", "pass123")
        assert result is None


class TestJWTToken:
    """Tests for JWT token generation and verification."""

    def setup_method(self):
        """Reset user store before each test."""
        get_user_store().clear()

    def test_generate_token(self):
        """generate_token should return a JWT token string."""
        user = create_user(username="testuser", email="test@test.com", password="pass123")
        token = generate_token(user)
        assert token is not None
        assert isinstance(token, str)
        assert token.count('.') == 2  # JWT has 3 parts separated by dots

    def test_verify_token_valid(self):
        """verify_token should return user_id for valid token."""
        user = create_user(username="testuser", email="test@test.com", password="pass123")
        token = generate_token(user)
        user_id = verify_token(token)
        assert user_id == user.id

    def test_verify_token_invalid(self):
        """verify_token should return None for invalid token."""
        result = verify_token("invalid.token.here")
        assert result is None

    def test_verify_token_tampered(self):
        """verify_token should return None for tampered token."""
        user = create_user(username="testuser", email="test@test.com", password="pass123")
        token = generate_token(user)
        # Tamper with the token
        parts = token.split('.')
        parts[1] = parts[1] + "tampered"
        tampered_token = '.'.join(parts)
        result = verify_token(tampered_token)
        assert result is None

    def test_verify_token_expired(self):
        """verify_token should return None for expired token."""
        user = create_user(username="testuser", email="test@test.com", password="pass123")
        # Generate token with mocked time in the past
        with patch('knowledgebot.core.auth.time.time') as mock_time:
            # Set time to way in the past so token is expired
            mock_time.return_value = time.time() - JWT_EXPIRATION_SECONDS - 100
            token = generate_token(user)
        # Now verify with current time - should be expired
        result = verify_token(token)
        assert result is None

    def test_get_user_by_token(self):
        """get_user_by_token should return user for valid token."""
        user = create_user(username="testuser", email="test@test.com", password="pass123")
        token = generate_token(user)
        retrieved_user = get_user_by_token(token)
        assert retrieved_user is not None
        assert retrieved_user.id == user.id
        assert retrieved_user.username == user.username

    def test_get_user_by_token_invalid(self):
        """get_user_by_token should return None for invalid token."""
        result = get_user_by_token("invalid.token.here")
        assert result is None
