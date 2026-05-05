"""
API module - REST API request/response handlers.

This module provides framework-agnostic API handlers that can be integrated
with any web framework (Flask, FastAPI, Django REST, etc.).
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class HTTPStatus(int, Enum):
    """HTTP status codes."""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500


@dataclass
class APIResponse:
    """Standard API response wrapper."""
    success: bool = True
    data: Any = None
    message: str = ""
    status_code: int = 200
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.errors:
            result["errors"] = self.errors
        return result


def ok(data: Any = None, message: str = "Success") -> APIResponse:
    """Create a success response."""
    return APIResponse(
        success=True,
        data=data,
        message=message,
        status_code=HTTPStatus.OK,
    )


def created(data: Any = None, message: str = "Created") -> APIResponse:
    """Create a created response."""
    return APIResponse(
        success=True,
        data=data,
        message=message,
        status_code=HTTPStatus.CREATED,
    )


def no_content(message: str = "Deleted") -> APIResponse:
    """Create a no content response."""
    return APIResponse(
        success=True,
        message=message,
        status_code=HTTPStatus.NO_CONTENT,
    )


def bad_request(message: str, errors: Optional[list[dict]] = None) -> APIResponse:
    """Create a bad request response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.BAD_REQUEST,
        errors=errors or [],
    )


def unauthorized(message: str = "Unauthorized") -> APIResponse:
    """Create an unauthorized response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.UNAUTHORIZED,
    )


def forbidden(message: str = "Forbidden") -> APIResponse:
    """Create a forbidden response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.FORBIDDEN,
    )


def not_found(message: str = "Not found") -> APIResponse:
    """Create a not found response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.NOT_FOUND,
    )


def conflict(message: str) -> APIResponse:
    """Create a conflict response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.CONFLICT,
    )


def validation_error(message: str, errors: list[dict]) -> APIResponse:
    """Create a validation error response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        errors=errors,
    )


def internal_error(message: str = "Internal server error") -> APIResponse:
    """Create an internal server error response."""
    return APIResponse(
        success=False,
        message=message,
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
