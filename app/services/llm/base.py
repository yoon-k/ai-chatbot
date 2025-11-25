from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on typical pricing."""
        # Rough estimate: $0.01/1K input, $0.03/1K output for GPT-4
        return (self.prompt_tokens * 0.00001) + (self.completion_tokens * 0.00003)


@dataclass
class LLMMessage:
    """Standard message format."""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standard LLM response format."""
    content: str
    finish_reason: FinishReason
    usage: TokenUsage
    model: str
    function_call: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "finish_reason": self.finish_reason.value,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "estimated_cost": self.usage.estimated_cost
            },
            "model": self.model,
            "function_call": self.function_call
        }


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    delta: str
    finish_reason: Optional[FinishReason] = None
    usage: Optional[TokenUsage] = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of supported models."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming completion."""
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    def _prepare_messages(
        self,
        messages: List[LLMMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LLMMessage to provider-specific format."""
        result = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.name:
                m["name"] = msg.name
            if msg.function_call:
                m["function_call"] = msg.function_call
            result.append(m)
        return result

    def validate_model(self, model: str) -> bool:
        """Check if model is supported."""
        return model in self.supported_models


class LLMError(Exception):
    """Base exception for LLM errors."""

    def __init__(self, message: str, provider: str, details: Optional[Dict] = None):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(self.message)


class RateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {provider}",
            provider,
            {"retry_after": retry_after}
        )


class ModelNotFoundError(LLMError):
    """Model not found or not accessible."""

    def __init__(self, provider: str, model: str):
        super().__init__(
            f"Model '{model}' not found for {provider}",
            provider,
            {"model": model}
        )


class ContentFilterError(LLMError):
    """Content was filtered."""

    def __init__(self, provider: str, reason: str):
        super().__init__(
            f"Content filtered: {reason}",
            provider,
            {"reason": reason}
        )
