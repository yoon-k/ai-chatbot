from typing import AsyncGenerator, List, Dict, Any, Optional
from anthropic import AsyncAnthropic

from app.services.llm.base import (
    BaseLLM, LLMMessage, LLMResponse, StreamChunk,
    TokenUsage, FinishReason, LLMError, RateLimitError
)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude implementation."""

    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ]

    PRICING = {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005}
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS

    async def count_tokens(self, text: str) -> int:
        """Estimate token count (Claude doesn't expose tokenizer)."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def _prepare_messages_for_claude(
        self,
        messages: List[LLMMessage]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Separate system message and convert to Claude format."""
        system_prompt = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                role = "user" if msg.role == "user" else "assistant"
                chat_messages.append({
                    "role": role,
                    "content": msg.content
                })

        return system_prompt, chat_messages

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
            system_prompt, chat_messages = self._prepare_messages_for_claude(messages)

            request_params = {
                "model": self.model,
                "messages": chat_messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }

            if system_prompt:
                request_params["system"] = system_prompt

            # Handle tools (function calling)
            if functions:
                request_params["tools"] = [
                    {
                        "name": f["name"],
                        "description": f["description"],
                        "input_schema": f["parameters"]
                    }
                    for f in functions
                ]

            response = await self.client.messages.create(**request_params)

            # Extract content
            content = ""
            func_call = None

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    func_call = {
                        "name": block.name,
                        "arguments": block.input
                    }

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )

            finish_reason_map = {
                "end_turn": FinishReason.STOP,
                "max_tokens": FinishReason.LENGTH,
                "tool_use": FinishReason.FUNCTION_CALL
            }

            return LLMResponse(
                content=content,
                finish_reason=finish_reason_map.get(
                    response.stop_reason, FinishReason.STOP
                ),
                usage=usage,
                model=self.model,
                function_call=func_call
            )

        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError("anthropic")
            raise LLMError(error_message, "anthropic")

    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            system_prompt, chat_messages = self._prepare_messages_for_claude(messages)

            request_params = {
                "model": self.model,
                "messages": chat_messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }

            if system_prompt:
                request_params["system"] = system_prompt

            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text)

                # Get final message for usage
                final_message = await stream.get_final_message()
                usage = TokenUsage(
                    prompt_tokens=final_message.usage.input_tokens,
                    completion_tokens=final_message.usage.output_tokens,
                    total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens
                )

                yield StreamChunk(
                    delta="",
                    finish_reason=FinishReason.STOP,
                    usage=usage
                )

        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError("anthropic")
            raise LLMError(error_message, "anthropic")
