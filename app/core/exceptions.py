from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class ChatbotException(Exception):
    """Base exception for chatbot application."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(ChatbotException):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTH_ERROR")


class AuthorizationError(ChatbotException):
    """Authorization failed."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, code="FORBIDDEN")


class NotFoundError(ChatbotException):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            f"{resource} not found: {identifier}",
            code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier}
        )


class ValidationError(ChatbotException):
    """Validation failed."""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class RateLimitError(ChatbotException):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds",
            code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after}
        )


class LLMError(ChatbotException):
    """LLM provider error."""

    def __init__(self, provider: str, message: str):
        super().__init__(
            f"LLM error ({provider}): {message}",
            code="LLM_ERROR",
            details={"provider": provider}
        )


class DocumentProcessingError(ChatbotException):
    """Document processing failed."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Failed to process document '{filename}': {reason}",
            code="DOCUMENT_ERROR",
            details={"filename": filename, "reason": reason}
        )


class ConversationNotFoundError(NotFoundError):
    """Conversation not found."""

    def __init__(self, conversation_id: str):
        super().__init__("Conversation", conversation_id)


class UserNotFoundError(NotFoundError):
    """User not found."""

    def __init__(self, identifier: str):
        super().__init__("User", identifier)


def raise_http_exception(exc: ChatbotException) -> None:
    """Convert ChatbotException to HTTPException."""
    status_map = {
        "AUTH_ERROR": status.HTTP_401_UNAUTHORIZED,
        "FORBIDDEN": status.HTTP_403_FORBIDDEN,
        "NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "RATE_LIMIT_EXCEEDED": status.HTTP_429_TOO_MANY_REQUESTS,
        "LLM_ERROR": status.HTTP_502_BAD_GATEWAY,
        "DOCUMENT_ERROR": status.HTTP_400_BAD_REQUEST,
        "INTERNAL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR
    }

    raise HTTPException(
        status_code=status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR),
        detail={
            "message": exc.message,
            "code": exc.code,
            "details": exc.details
        }
    )
