from typing import Optional, Dict, Type
from app.services.llm.base import BaseLLM, ModelNotFoundError
from app.services.llm.openai import OpenAILLM
from app.services.llm.anthropic import AnthropicLLM
from app.core.config import settings


class LLMFactory:
    """Factory for creating LLM instances."""

    _providers: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM
    }

    _model_to_provider: Dict[str, str] = {
        # OpenAI models
        "gpt-4-turbo-preview": "openai",
        "gpt-4-0125-preview": "openai",
        "gpt-4-1106-preview": "openai",
        "gpt-4": "openai",
        "gpt-4-32k": "openai",
        "gpt-3.5-turbo": "openai",
        "gpt-3.5-turbo-16k": "openai",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",

        # Anthropic models
        "claude-3-opus-20240229": "anthropic",
        "claude-3-sonnet-20240229": "anthropic",
        "claude-3-haiku-20240307": "anthropic",
        "claude-3-5-sonnet-20241022": "anthropic",
        "claude-3-5-haiku-20241022": "anthropic"
    }

    @classmethod
    def get_provider_for_model(cls, model: str) -> str:
        """Get the provider name for a given model."""
        provider = cls._model_to_provider.get(model)
        if not provider:
            raise ModelNotFoundError("unknown", model)
        return provider

    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM instance.

        Args:
            provider: Provider name (openai, anthropic). Auto-detected if not provided.
            model: Model name. Uses default if not provided.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            Configured LLM instance.
        """
        # Use defaults from settings
        model = model or settings.DEFAULT_MODEL
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        max_tokens = max_tokens or settings.MAX_TOKENS

        # Auto-detect provider from model
        if not provider:
            provider = cls.get_provider_for_model(model)

        # Get provider class
        provider_class = cls._providers.get(provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider}")

        # Get API key
        api_key = cls._get_api_key(provider)
        if not api_key:
            raise ValueError(f"API key not configured for provider: {provider}")

        return provider_class(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @classmethod
    def _get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for provider from settings."""
        key_map = {
            "openai": settings.OPENAI_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
            "google": settings.GOOGLE_API_KEY
        }
        return key_map.get(provider)

    @classmethod
    def list_available_models(cls) -> Dict[str, list]:
        """List all available models by provider."""
        result = {}
        for provider_name, provider_class in cls._providers.items():
            api_key = cls._get_api_key(provider_name)
            if api_key:
                # Create temporary instance to get models
                instance = provider_class(
                    api_key=api_key,
                    model=provider_class.SUPPORTED_MODELS[0]
                )
                result[provider_name] = instance.supported_models
        return result

    @classmethod
    def is_model_available(cls, model: str) -> bool:
        """Check if a model is available."""
        try:
            provider = cls.get_provider_for_model(model)
            api_key = cls._get_api_key(provider)
            return api_key is not None
        except ModelNotFoundError:
            return False


# Convenience function
def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> BaseLLM:
    """Get an LLM instance with default settings."""
    return LLMFactory.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
