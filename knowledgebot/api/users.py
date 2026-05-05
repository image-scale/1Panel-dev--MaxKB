"""
REST API endpoints for user operations.

This module provides API handlers for user management:
- Registration
- Login
- Profile retrieval and update
- Password change
"""
from dataclasses import dataclass
from typing import Optional

from knowledgebot.api import (
    APIResponse,
    ok,
    created,
    bad_request,
    unauthorized,
    not_found,
    conflict,
    validation_error,
)
from knowledgebot.core.auth import (
    create_user,
    authenticate_user,
    generate_token,
    verify_token,
    get_user_by_token,
)
from knowledgebot.core.models import User, UserRole, get_user_store


@dataclass
class RegisterRequest:
    """Request data for user registration."""
    username: str
    email: str
    password: str
    role: str = "user"

    def validate(self) -> list[dict]:
        """Validate the request data."""
        errors = []
        if not self.username or len(self.username) < 3:
            errors.append({
                "field": "username",
                "message": "Username must be at least 3 characters"
            })
        if not self.email or "@" not in self.email:
            errors.append({
                "field": "email",
                "message": "Valid email is required"
            })
        if not self.password or len(self.password) < 6:
            errors.append({
                "field": "password",
                "message": "Password must be at least 6 characters"
            })
        if self.role not in ["user", "admin"]:
            errors.append({
                "field": "role",
                "message": "Role must be 'user' or 'admin'"
            })
        return errors


@dataclass
class LoginRequest:
    """Request data for user login."""
    username: str
    password: str

    def validate(self) -> list[dict]:
        """Validate the request data."""
        errors = []
        if not self.username:
            errors.append({
                "field": "username",
                "message": "Username is required"
            })
        if not self.password:
            errors.append({
                "field": "password",
                "message": "Password is required"
            })
        return errors


@dataclass
class UpdateProfileRequest:
    """Request data for profile update."""
    email: Optional[str] = None
    is_active: Optional[bool] = None

    def validate(self) -> list[dict]:
        """Validate the request data."""
        errors = []
        if self.email is not None and "@" not in self.email:
            errors.append({
                "field": "email",
                "message": "Valid email is required"
            })
        return errors


@dataclass
class ChangePasswordRequest:
    """Request data for password change."""
    current_password: str
    new_password: str

    def validate(self) -> list[dict]:
        """Validate the request data."""
        errors = []
        if not self.current_password:
            errors.append({
                "field": "current_password",
                "message": "Current password is required"
            })
        if not self.new_password or len(self.new_password) < 6:
            errors.append({
                "field": "new_password",
                "message": "New password must be at least 6 characters"
            })
        return errors


def register(request: RegisterRequest) -> APIResponse:
    """
    Register a new user.

    Args:
        request: Registration data.

    Returns:
        APIResponse with user data or error.
    """
    # Validate request
    errors = request.validate()
    if errors:
        return validation_error("Validation failed", errors)

    # Map role string to enum
    role = UserRole.ADMIN if request.role == "admin" else UserRole.USER

    try:
        user = create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            role=role,
        )
        return created(
            data=user.to_dict(),
            message="User registered successfully",
        )
    except ValueError as e:
        return conflict(str(e))


def login(request: LoginRequest) -> APIResponse:
    """
    Authenticate a user and return a token.

    Args:
        request: Login credentials.

    Returns:
        APIResponse with token or error.
    """
    # Validate request
    errors = request.validate()
    if errors:
        return validation_error("Validation failed", errors)

    # Authenticate
    user = authenticate_user(request.username, request.password)
    if not user:
        return unauthorized("Invalid username or password")

    # Generate token
    token = generate_token(user)

    return ok(
        data={
            "token": token,
            "user": user.to_dict(),
        },
        message="Login successful",
    )


def get_current_user(token: str) -> APIResponse:
    """
    Get the current user from a token.

    Args:
        token: JWT token.

    Returns:
        APIResponse with user data or error.
    """
    if not token:
        return unauthorized("Token is required")

    user = get_user_by_token(token)
    if not user:
        return unauthorized("Invalid or expired token")

    return ok(
        data=user.to_dict(),
        message="User retrieved successfully",
    )


def get_user(user_id: str, requesting_user_id: str) -> APIResponse:
    """
    Get a user by ID.

    Args:
        user_id: ID of user to retrieve.
        requesting_user_id: ID of the requesting user.

    Returns:
        APIResponse with user data or error.
    """
    store = get_user_store()
    user = store.get_by_id(user_id)

    if not user:
        return not_found("User not found")

    # Users can only view their own profile unless admin
    requesting_user = store.get_by_id(requesting_user_id)
    if not requesting_user:
        return unauthorized("Invalid requesting user")

    if requesting_user_id != user_id and requesting_user.role != UserRole.ADMIN:
        return unauthorized("Cannot view other users' profiles")

    return ok(
        data=user.to_dict(),
        message="User retrieved successfully",
    )


def update_profile(
    user_id: str,
    request: UpdateProfileRequest,
    requesting_user_id: str,
) -> APIResponse:
    """
    Update a user's profile.

    Args:
        user_id: ID of user to update.
        request: Update data.
        requesting_user_id: ID of the requesting user.

    Returns:
        APIResponse with updated user data or error.
    """
    # Validate request
    errors = request.validate()
    if errors:
        return validation_error("Validation failed", errors)

    store = get_user_store()
    user = store.get_by_id(user_id)

    if not user:
        return not_found("User not found")

    # Users can only update their own profile unless admin
    requesting_user = store.get_by_id(requesting_user_id)
    if not requesting_user:
        return unauthorized("Invalid requesting user")

    if requesting_user_id != user_id and requesting_user.role != UserRole.ADMIN:
        return unauthorized("Cannot update other users' profiles")

    # Update fields
    if request.email is not None:
        # Check if email already exists
        existing = store.get_by_email(request.email)
        if existing and existing.id != user_id:
            return conflict("Email already in use")
        user.email = request.email

    if request.is_active is not None:
        # Only admins can change active status
        if requesting_user.role != UserRole.ADMIN:
            return unauthorized("Only admins can change active status")
        user.is_active = request.is_active

    store.update(user)

    return ok(
        data=user.to_dict(),
        message="Profile updated successfully",
    )


def change_password(
    user_id: str,
    request: ChangePasswordRequest,
    requesting_user_id: str,
) -> APIResponse:
    """
    Change a user's password.

    Args:
        user_id: ID of user whose password to change.
        request: Password change data.
        requesting_user_id: ID of the requesting user.

    Returns:
        APIResponse with success or error.
    """
    # Validate request
    errors = request.validate()
    if errors:
        return validation_error("Validation failed", errors)

    store = get_user_store()
    user = store.get_by_id(user_id)

    if not user:
        return not_found("User not found")

    # Users can only change their own password unless admin
    requesting_user = store.get_by_id(requesting_user_id)
    if not requesting_user:
        return unauthorized("Invalid requesting user")

    if requesting_user_id != user_id and requesting_user.role != UserRole.ADMIN:
        return unauthorized("Cannot change other users' passwords")

    # Verify current password (skip for admin changing another user's password)
    if requesting_user_id == user_id:
        if not user.check_password(request.current_password):
            return bad_request("Current password is incorrect")

    # Set new password
    user.set_password(request.new_password)
    store.update(user)

    return ok(message="Password changed successfully")


def list_users(requesting_user_id: str) -> APIResponse:
    """
    List all users (admin only).

    Args:
        requesting_user_id: ID of the requesting user.

    Returns:
        APIResponse with list of users or error.
    """
    store = get_user_store()
    requesting_user = store.get_by_id(requesting_user_id)

    if not requesting_user:
        return unauthorized("Invalid requesting user")

    if requesting_user.role != UserRole.ADMIN:
        return unauthorized("Admin access required")

    users = store.list_all()
    return ok(
        data=[u.to_dict() for u in users],
        message=f"Found {len(users)} users",
    )


def verify_auth_token(token: str) -> APIResponse:
    """
    Verify if a token is valid.

    Args:
        token: JWT token to verify.

    Returns:
        APIResponse with validity status.
    """
    if not token:
        return bad_request("Token is required")

    user_id = verify_token(token)
    if not user_id:
        return ok(
            data={"valid": False},
            message="Token is invalid or expired",
        )

    return ok(
        data={
            "valid": True,
            "user_id": user_id,
        },
        message="Token is valid",
    )


def refresh_token(token: str) -> APIResponse:
    """
    Refresh an existing token.

    Args:
        token: Current JWT token.

    Returns:
        APIResponse with new token or error.
    """
    if not token:
        return bad_request("Token is required")

    user = get_user_by_token(token)
    if not user:
        return unauthorized("Invalid or expired token")

    # Generate new token
    new_token = generate_token(user)

    return ok(
        data={
            "token": new_token,
            "user": user.to_dict(),
        },
        message="Token refreshed successfully",
    )
