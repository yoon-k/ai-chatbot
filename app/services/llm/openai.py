from typing import AsyncGenerator, List, Dict, Any, Optional
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.services.llm.base import (
    BaseLLM, LLMMessage, LLMResponse, StreamChunk,
    TokenUsage, FinishReason, LLMError, RateLimitError
)


class OpenAILLM(BaseLLM):
    """OpenAI GPT implementation."""

    SUPPORTED_MODELS = [
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4o",
        "gpt-4o-mini"
    ]

    # Pricing per 1K tokens (USD)
    PRICING = {
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        organization_id: Optional[str] = None
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization_id
        )
        self._tokenizer = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))

    def _calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost based on model pricing."""
        pricing = self.PRICING.get(self.model, {"input": 0.01, "output": 0.03})
        input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError,))
    )
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion."""
        try:
            request_params = {
                "model": self.model,
                "messages": self._prepare_messages(messages),
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }

            if functions:
                request_params["tools"] = [
                    {"type": "function", "function": f} for f in functions
                ]
                if function_call:
                    if function_call == "auto":
                        request_params["tool_choice"] = "auto"
                    elif function_call == "none":
                        request_params["tool_choice"] = "none"
                    else:
                        request_params["tool_choice"] = {
                            "type": "function",
                            "function": {"name": function_call}
                        }

            response = await self.client.chat.completions.create(**request_params)

            choice = response.choices[0]
            message = choice.message

            # Handle function calls
            func_call = None
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_call = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )

            finish_reason_map = {
                "stop": FinishReason.STOP,
                "length": FinishReason.LENGTH,
                "tool_calls": FinishReason.FUNCTION_CALL,
                "content_filter": FinishReason.CONTENT_FILTER
            }

            return LLMResponse(
                content=message.content or "",
                finish_reason=finish_reason_map.get(
                    choice.finish_reason, FinishReason.STOP
                ),
                usage=usage,
                model=response.model,
                function_call=func_call
            )

        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError("openai")
            raise LLMError(error_message, "openai")

    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self._prepare_messages(messages),
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            async for chunk in response:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    finish_reason = None
                    if choice.finish_reason:
                        finish_reason_map = {
                            "stop": FinishReason.STOP,
                            "length": FinishReason.LENGTH
                        }
                        finish_reason = finish_reason_map.get(
                            choice.finish_reason, FinishReason.STOP
                        )

                    usage = None
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        )

                    yield StreamChunk(
                        delta=delta.content or "",
                        finish_reason=finish_reason,
                        usage=usage
                    )

        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError("openai")
            raise LLMError(error_message, "openai")
