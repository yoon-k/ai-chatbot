from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.db.session import get_db
from app.api.deps import get_current_user_with_rate_limit
from app.core.security import TokenData
from app.schemas.chat import (
    ChatRequest, ChatResponse, RAGChatRequest,
    ConversationCreateRequest, ConversationUpdateRequest,
    ConversationResponse, ConversationListResponse,
    ConversationHistoryResponse, MessageFeedbackRequest
)
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Send a message and get AI response.

    MUSE AI will process your message and return an intelligent response.
    Supports multiple LLM providers (OpenAI GPT, Anthropic Claude).
    """
    service = ChatService(db)
    return await service.chat(request, current_user.user_id)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Send a message and get streaming AI response.

    Returns Server-Sent Events (SSE) stream for real-time response.
    """
    request.stream = True
    service = ChatService(db)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in service.chat_stream(request, current_user.user_id):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/rag", response_model=ChatResponse)
async def chat_with_rag(
    request: RAGChatRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Send a message with document context (RAG).

    MUSE AI will search through your uploaded documents and use
    relevant information to provide accurate, context-aware responses.
    """
    # Convert to ChatRequest with RAG enabled
    chat_request = ChatRequest(
        message=request.message,
        conversation_id=request.conversation_id,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        use_rag=True,
        document_ids=request.document_ids,
        top_k=request.top_k,
        stream=request.stream
    )

    service = ChatService(db)
    return await service.chat(chat_request, current_user.user_id)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """List user's conversations."""
    from sqlalchemy import select, func
    from app.models.conversation import Conversation, ConversationStatus

    # Count total
    count_query = select(func.count(Conversation.id)).where(
        Conversation.user_id == current_user.user_id,
        Conversation.status == ConversationStatus.ACTIVE
    )
    total = (await db.execute(count_query)).scalar()

    # Get page
    offset = (page - 1) * page_size
    query = (
        select(Conversation)
        .where(
            Conversation.user_id == current_user.user_id,
            Conversation.status == ConversationStatus.ACTIVE
        )
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(query)
    conversations = result.scalars().all()

    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.id,
                title=c.title,
                model=c.model,
                system_prompt=c.system_prompt,
                temperature=c.temperature,
                message_count=c.message_count,
                total_tokens=c.total_tokens,
                created_at=c.created_at,
                updated_at=c.updated_at,
                last_message_at=c.last_message_at
            )
            for c in conversations
        ],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(conversations)) < total
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Get conversation with message history."""
    from sqlalchemy import select
    from app.models.conversation import Conversation, Message
    from app.schemas.chat import MessageResponse

    # Get conversation
    conversation = await db.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    return ConversationHistoryResponse(
        conversation=ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            model=conversation.model,
            system_prompt=conversation.system_prompt,
            temperature=conversation.temperature,
            message_count=conversation.message_count,
            total_tokens=conversation.total_tokens,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at
        ),
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                model=m.model,
                created_at=m.created_at
            )
            for m in messages
        ]
    )


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreateRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Create a new conversation."""
    import uuid
    from app.models.conversation import Conversation
    from app.core.config import settings

    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        title=request.title,
        model=request.model or settings.DEFAULT_MODEL,
        system_prompt=request.system_prompt,
        temperature=request.temperature or settings.TEMPERATURE,
        metadata=request.metadata
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        model=conversation.model,
        system_prompt=conversation.system_prompt,
        temperature=conversation.temperature,
        message_count=conversation.message_count,
        total_tokens=conversation.total_tokens,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Update conversation settings."""
    from app.models.conversation import Conversation

    conversation = await db.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if request.title is not None:
        conversation.title = request.title
    if request.model is not None:
        conversation.model = request.model
    if request.system_prompt is not None:
        conversation.system_prompt = request.system_prompt
    if request.temperature is not None:
        conversation.temperature = request.temperature
    if request.metadata is not None:
        conversation.metadata = request.metadata

    await db.commit()
    await db.refresh(conversation)

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        model=conversation.model,
        system_prompt=conversation.system_prompt,
        temperature=conversation.temperature,
        message_count=conversation.message_count,
        total_tokens=conversation.total_tokens,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation (soft delete)."""
    from app.models.conversation import Conversation, ConversationStatus

    conversation = await db.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.status = ConversationStatus.DELETED
    await db.commit()

    return {"message": "Conversation deleted"}


@router.post("/messages/{message_id}/feedback")
async def submit_feedback(
    message_id: str,
    request: MessageFeedbackRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback for a message."""
    from app.models.conversation import Message, Conversation

    message = await db.get(Message, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Verify ownership
    conversation = await db.get(Conversation, message.conversation_id)
    if conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Message not found")

    message.rating = request.rating
    message.feedback = request.feedback
    await db.commit()

    return {"message": "Feedback submitted"}
