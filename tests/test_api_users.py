"""
Tests for user REST API endpoints.
"""
import pytest

from knowledgebot.api import HTTPStatus
from knowledgebot.api.users import (
    RegisterRequest,
    LoginRequest,
    UpdateProfileRequest,
    ChangePasswordRequest,
    register,
    login,
    get_current_user,
    get_user,
    update_profile,
    change_password,
    list_users,
    verify_auth_token,
    refresh_token,
)
from knowledgebot.core.auth import create_user, generate_token
from knowledgebot.core.models import User, UserRole, get_user_store


class TestRegisterRequest:
    """Tests for RegisterRequest validation."""

    def test_valid_request(self):
        """Should validate correct data."""
        request = RegisterRequest(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        errors = request.validate()
        assert errors == []

    def test_short_username(self):
        """Should reject short username."""
        request = RegisterRequest(
            username="ab",
            email="test@example.com",
            password="password123",
        )
        errors = request.validate()
        assert len(errors) == 1
        assert errors[0]["field"] == "username"

    def test_invalid_email(self):
        """Should reject invalid email."""
        request = RegisterRequest(
            username="testuser",
            email="invalid-email",
            password="password123",
        )
        errors = request.validate()
        assert len(errors) == 1
        assert errors[0]["field"] == "email"

    def test_short_password(self):
        """Should reject short password."""
        request = RegisterRequest(
            username="testuser",
            email="test@example.com",
            password="12345",
        )
        errors = request.validate()
        assert len(errors) == 1
        assert errors[0]["field"] == "password"


class TestLoginRequest:
    """Tests for LoginRequest validation."""

    def test_valid_request(self):
        """Should validate correct data."""
        request = LoginRequest(username="testuser", password="password123")
        errors = request.validate()
        assert errors == []

    def test_missing_username(self):
        """Should reject missing username."""
        request = LoginRequest(username="", password="password123")
        errors = request.validate()
        assert len(errors) == 1
        assert errors[0]["field"] == "username"


class TestRegister:
    """Tests for register endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_register_success(self):
        """Should register new user."""
        request = RegisterRequest(
            username="newuser",
            email="new@example.com",
            password="password123",
        )
        response = register(request)
        assert response.success is True
        assert response.status_code == HTTPStatus.CREATED
        assert response.data["username"] == "newuser"

    def test_register_with_role(self):
        """Should register with specified role."""
        request = RegisterRequest(
            username="admin",
            email="admin@example.com",
            password="password123",
            role="admin",
        )
        response = register(request)
        assert response.success is True
        assert response.data["role"] == "admin"

    def test_register_duplicate_username(self):
        """Should reject duplicate username."""
        request = RegisterRequest(
            username="existing",
            email="first@example.com",
            password="password123",
        )
        register(request)

        duplicate = RegisterRequest(
            username="existing",
            email="second@example.com",
            password="password123",
        )
        response = register(duplicate)
        assert response.success is False
        assert response.status_code == HTTPStatus.CONFLICT

    def test_register_validation_error(self):
        """Should return validation error for invalid data."""
        request = RegisterRequest(
            username="ab",  # Too short
            email="invalid",  # Invalid email
            password="123",  # Too short
        )
        response = register(request)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert len(response.errors) == 3


class TestLogin:
    """Tests for login endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_login_success(self):
        """Should login with correct credentials."""
        # Create user first
        create_user("testuser", "test@example.com", "password123")

        request = LoginRequest(username="testuser", password="password123")
        response = login(request)
        assert response.success is True
        assert "token" in response.data
        assert response.data["user"]["username"] == "testuser"

    def test_login_wrong_password(self):
        """Should reject wrong password."""
        create_user("testuser", "test@example.com", "password123")

        request = LoginRequest(username="testuser", password="wrongpassword")
        response = login(request)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_login_nonexistent_user(self):
        """Should reject nonexistent user."""
        request = LoginRequest(username="nonexistent", password="password123")
        response = login(request)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED


class TestGetCurrentUser:
    """Tests for get_current_user endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_get_current_user_success(self):
        """Should return user for valid token."""
        user = create_user("testuser", "test@example.com", "password123")
        token = generate_token(user)

        response = get_current_user(token)
        assert response.success is True
        assert response.data["username"] == "testuser"

    def test_get_current_user_invalid_token(self):
        """Should reject invalid token."""
        response = get_current_user("invalid-token")
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_get_current_user_no_token(self):
        """Should reject missing token."""
        response = get_current_user("")
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED


class TestGetUser:
    """Tests for get_user endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_get_own_profile(self):
        """Should return own profile."""
        user = create_user("testuser", "test@example.com", "password123")
        response = get_user(user.id, user.id)
        assert response.success is True
        assert response.data["username"] == "testuser"

    def test_get_other_user_as_admin(self):
        """Admin should get other user's profile."""
        admin = create_user("admin", "admin@example.com", "password123", UserRole.ADMIN)
        user = create_user("testuser", "test@example.com", "password123")

        response = get_user(user.id, admin.id)
        assert response.success is True
        assert response.data["username"] == "testuser"

    def test_get_other_user_as_regular(self):
        """Regular user should not get other user's profile."""
        user1 = create_user("user1", "user1@example.com", "password123")
        user2 = create_user("user2", "user2@example.com", "password123")

        response = get_user(user2.id, user1.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_get_nonexistent_user(self):
        """Should return not found for nonexistent user."""
        user = create_user("testuser", "test@example.com", "password123")
        response = get_user("nonexistent", user.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.NOT_FOUND


class TestUpdateProfile:
    """Tests for update_profile endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_update_own_email(self):
        """Should update own email."""
        user = create_user("testuser", "old@example.com", "password123")
        request = UpdateProfileRequest(email="new@example.com")

        response = update_profile(user.id, request, user.id)
        assert response.success is True
        assert response.data["email"] == "new@example.com"

    def test_update_other_user_as_admin(self):
        """Admin should update other user."""
        admin = create_user("admin", "admin@example.com", "password123", UserRole.ADMIN)
        user = create_user("testuser", "test@example.com", "password123")
        request = UpdateProfileRequest(is_active=False)

        response = update_profile(user.id, request, admin.id)
        assert response.success is True
        assert response.data["is_active"] is False

    def test_update_other_user_as_regular(self):
        """Regular user should not update other user."""
        user1 = create_user("user1", "user1@example.com", "password123")
        user2 = create_user("user2", "user2@example.com", "password123")
        request = UpdateProfileRequest(email="changed@example.com")

        response = update_profile(user2.id, request, user1.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_update_email_conflict(self):
        """Should reject duplicate email."""
        user1 = create_user("user1", "user1@example.com", "password123")
        create_user("user2", "user2@example.com", "password123")
        request = UpdateProfileRequest(email="user2@example.com")

        response = update_profile(user1.id, request, user1.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.CONFLICT

    def test_update_active_as_regular(self):
        """Regular user should not change active status."""
        user = create_user("testuser", "test@example.com", "password123")
        request = UpdateProfileRequest(is_active=False)

        response = update_profile(user.id, request, user.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED


class TestChangePassword:
    """Tests for change_password endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_change_own_password(self):
        """Should change own password."""
        user = create_user("testuser", "test@example.com", "oldpassword")
        request = ChangePasswordRequest(
            current_password="oldpassword",
            new_password="newpassword",
        )

        response = change_password(user.id, request, user.id)
        assert response.success is True

        # Verify new password works
        updated_user = get_user_store().get_by_id(user.id)
        assert updated_user.check_password("newpassword")

    def test_change_password_wrong_current(self):
        """Should reject wrong current password."""
        user = create_user("testuser", "test@example.com", "oldpassword")
        request = ChangePasswordRequest(
            current_password="wrongpassword",
            new_password="newpassword",
        )

        response = change_password(user.id, request, user.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_change_password_validation_error(self):
        """Should reject short new password."""
        user = create_user("testuser", "test@example.com", "oldpassword")
        request = ChangePasswordRequest(
            current_password="oldpassword",
            new_password="123",  # Too short
        )

        response = change_password(user.id, request, user.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


class TestListUsers:
    """Tests for list_users endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_list_users_as_admin(self):
        """Admin should list all users."""
        admin = create_user("admin", "admin@example.com", "password123", UserRole.ADMIN)
        create_user("user1", "user1@example.com", "password123")
        create_user("user2", "user2@example.com", "password123")

        response = list_users(admin.id)
        assert response.success is True
        assert len(response.data) == 3

    def test_list_users_as_regular(self):
        """Regular user should not list users."""
        user = create_user("testuser", "test@example.com", "password123")
        response = list_users(user.id)
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED


class TestVerifyAuthToken:
    """Tests for verify_auth_token endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_verify_valid_token(self):
        """Should verify valid token."""
        user = create_user("testuser", "test@example.com", "password123")
        token = generate_token(user)

        response = verify_auth_token(token)
        assert response.success is True
        assert response.data["valid"] is True
        assert response.data["user_id"] == user.id

    def test_verify_invalid_token(self):
        """Should return invalid for bad token."""
        response = verify_auth_token("invalid-token")
        assert response.success is True  # Request succeeded
        assert response.data["valid"] is False

    def test_verify_missing_token(self):
        """Should reject missing token."""
        response = verify_auth_token("")
        assert response.success is False
        assert response.status_code == HTTPStatus.BAD_REQUEST


class TestRefreshToken:
    """Tests for refresh_token endpoint."""

    def setup_method(self):
        """Reset store before each test."""
        get_user_store().clear()

    def test_refresh_valid_token(self):
        """Should refresh valid token."""
        user = create_user("testuser", "test@example.com", "password123")
        old_token = generate_token(user)

        response = refresh_token(old_token)
        assert response.success is True
        assert "token" in response.data
        assert response.data["user"]["username"] == "testuser"

    def test_refresh_invalid_token(self):
        """Should reject invalid token."""
        response = refresh_token("invalid-token")
        assert response.success is False
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_refresh_missing_token(self):
        """Should reject missing token."""
        response = refresh_token("")
        assert response.success is False
        assert response.status_code == HTTPStatus.BAD_REQUEST
