import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
import json
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.factory import LLMFactory
from app.services.llm.base import LLMMessage, LLMResponse, StreamChunk
from app.services.rag.pipeline import RAGPipeline
from app.models.conversation import Conversation, Message, ConversationStatus
from app.schemas.chat import ChatRequest, ChatResponse, MessageResponse, TokenUsage
from app.core.exceptions import ConversationNotFoundError, LLMError
from app.core.config import settings

logger = structlog.get_logger()


class ChatService:
    """Main chat service handling conversations and LLM interactions."""

    # MUSE AI System Prompt
    DEFAULT_SYSTEM_PROMPT = """당신은 MUSE AI입니다.
MUSE는 창의적이고 지능적인 AI 어시스턴트로, 사용자의 질문에 정확하고 도움이 되는 답변을 제공합니다.

핵심 가치:
- 정확성: 사실에 기반한 정보를 제공합니다
- 창의성: 독창적이고 유용한 아이디어를 제시합니다
- 친절함: 항상 존중하고 친근한 태도로 대화합니다
- 전문성: 다양한 분야의 깊은 지식을 바탕으로 답변합니다

MUSE는 항상 사용자의 성공을 돕기 위해 최선을 다합니다."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.rag_pipeline = RAGPipeline()

    async def chat(
        self,
        request: ChatRequest,
        user_id: str
    ) -> ChatResponse:
        """Process a chat request and return response."""

        # Get or create conversation
        conversation = await self._get_or_create_conversation(
            request.conversation_id,
            user_id,
            request
        )

        # Build message history
        messages = await self._build_messages(
            conversation,
            request.message,
            request.system_prompt,
            request.include_history,
            request.history_limit
        )

        # Add RAG context if enabled
        if request.use_rag and request.document_ids:
            context = await self.rag_pipeline.retrieve(
                query=request.message,
                document_ids=request.document_ids,
                top_k=request.top_k
            )
            messages = self._inject_rag_context(messages, context)

        # Get LLM and generate response
        llm = LLMFactory.create(
            provider=request.provider.value if request.provider else None,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        try:
            response = await llm.complete(
                messages=messages,
                functions=self._convert_functions(request.functions) if request.functions else None,
                function_call=request.function_call
            )
        except Exception as e:
            logger.error("LLM error", error=str(e), user_id=user_id)
            raise LLMError(llm.provider_name, str(e))

        # Save messages
        user_message = await self._save_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message
        )

        assistant_message = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response.content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            function_call=response.function_call,
            finish_reason=response.finish_reason.value
        )

        # Update conversation stats
        await self._update_conversation_stats(
            conversation,
            response.usage.total_tokens
        )

        return ChatResponse(
            id=assistant_message.id,
            conversation_id=conversation.id,
            message=MessageResponse(
                id=assistant_message.id,
                role="assistant",
                content=response.content,
                model=response.model,
                created_at=assistant_message.created_at,
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    estimated_cost=response.usage.estimated_cost
                ),
                function_call=response.function_call
            ),
            model=response.model,
            provider=llm.provider_name,
            finish_reason=response.finish_reason.value
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a streaming chat request."""

        conversation = await self._get_or_create_conversation(
            request.conversation_id,
            user_id,
            request
        )

        messages = await self._build_messages(
            conversation,
            request.message,
            request.system_prompt,
            request.include_history,
            request.history_limit
        )

        if request.use_rag and request.document_ids:
            context = await self.rag_pipeline.retrieve(
                query=request.message,
                document_ids=request.document_ids,
                top_k=request.top_k
            )
            messages = self._inject_rag_context(messages, context)

        llm = LLMFactory.create(
            provider=request.provider.value if request.provider else None,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Save user message
        await self._save_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message
        )

        full_response = ""
        usage = None
        message_id = str(uuid.uuid4())

        try:
            async for chunk in llm.stream(messages=messages):
                full_response += chunk.delta

                yield {
                    "id": message_id,
                    "conversation_id": conversation.id,
                    "delta": chunk.delta,
                    "finish_reason": chunk.finish_reason.value if chunk.finish_reason else None
                }

                if chunk.usage:
                    usage = chunk.usage

        except Exception as e:
            logger.error("Streaming error", error=str(e))
            yield {"error": str(e)}
            return

        # Save assistant message
        await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=full_response,
            model=llm.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0
        )

        if usage:
            await self._update_conversation_stats(
                conversation,
                usage.total_tokens
            )

    async def _get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: str,
        request: ChatRequest
    ) -> Conversation:
        """Get existing conversation or create new one."""

        if conversation_id:
            conversation = await self.db.get(Conversation, conversation_id)
            if not conversation:
                raise ConversationNotFoundError(conversation_id)
            if conversation.user_id != user_id:
                raise ConversationNotFoundError(conversation_id)
            return conversation

        # Create new conversation
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            model=request.model or settings.DEFAULT_MODEL,
            system_prompt=request.system_prompt,
            temperature=request.temperature or settings.TEMPERATURE
        )
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)

        return conversation

    async def _build_messages(
        self,
        conversation: Conversation,
        user_message: str,
        system_prompt: Optional[str],
        include_history: bool,
        history_limit: int
    ) -> List[LLMMessage]:
        """Build message list for LLM."""

        messages = []

        # System prompt
        prompt = system_prompt or conversation.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        messages.append(LLMMessage(role="system", content=prompt))

        # History
        if include_history:
            history = await self._get_history(conversation.id, history_limit)
            for msg in history:
                messages.append(LLMMessage(role=msg.role, content=msg.content))

        # Current message
        messages.append(LLMMessage(role="user", content=user_message))

        return messages

    async def _get_history(
        self,
        conversation_id: str,
        limit: int
    ) -> List[Message]:
        """Get conversation history."""
        from sqlalchemy import select

        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))

    def _inject_rag_context(
        self,
        messages: List[LLMMessage],
        context: List[Dict[str, Any]]
    ) -> List[LLMMessage]:
        """Inject RAG context into messages."""

        if not context:
            return messages

        context_text = "\n\n---\n참고 문서:\n"
        for i, doc in enumerate(context, 1):
            context_text += f"\n[{i}] {doc['content'][:500]}...\n"

        # Insert before user message
        if len(messages) >= 2:
            messages[-2] = LLMMessage(
                role=messages[-2].role,
                content=messages[-2].content + context_text
            )

        return messages

    async def _save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        function_call: Optional[Dict] = None,
        finish_reason: Optional[str] = None
    ) -> Message:
        """Save a message to database."""

        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            function_call=function_call,
            finish_reason=finish_reason
        )
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)

        return message

    async def _update_conversation_stats(
        self,
        conversation: Conversation,
        tokens: int
    ) -> None:
        """Update conversation statistics."""

        conversation.message_count += 2  # user + assistant
        conversation.total_tokens += tokens
        conversation.last_message_at = datetime.utcnow()
        await self.db.commit()

    def _convert_functions(
        self,
        functions: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert function definitions to dict format."""
        return [
            {
                "name": f.name,
                "description": f.description,
                "parameters": f.parameters
            }
            for f in functions
        ]
