from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# ==================== Request Schemas ====================

class ChatMessage(BaseModel):
    """Single message in conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For function messages


class FunctionDefinition(BaseModel):
    """Function definition for function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ChatRequest(BaseModel):
    """Chat completion request."""
    message: str = Field(..., min_length=1, max_length=32000)
    conversation_id: Optional[str] = None

    # Model settings
    model: Optional[str] = None
    provider: Optional[LLMProvider] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)

    # Context
    system_prompt: Optional[str] = None
    include_history: bool = True
    history_limit: int = Field(default=20, ge=0, le=100)

    # Function calling
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[str] = None  # "auto", "none", or function name

    # RAG
    use_rag: bool = False
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)

    # Options
    stream: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "message": "안녕하세요, 오늘 날씨에 대해 알려주세요.",
                "conversation_id": "conv_abc123",
                "model": "gpt-4-turbo-preview",
                "temperature": 0.7,
                "stream": True
            }
        }


class RAGChatRequest(BaseModel):
    """RAG-enabled chat request."""
    message: str = Field(..., min_length=1, max_length=32000)
    document_ids: List[str] = Field(..., min_length=1)
    conversation_id: Optional[str] = None

    # Retrieval settings
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    include_metadata: bool = True

    # Generation settings
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)

    stream: bool = False


class ConversationCreateRequest(BaseModel):
    """Create new conversation."""
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    metadata: Optional[Dict[str, Any]] = None


class ConversationUpdateRequest(BaseModel):
    """Update conversation settings."""
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    metadata: Optional[Dict[str, Any]] = None


class MessageFeedbackRequest(BaseModel):
    """Submit feedback for a message."""
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None


# ==================== Response Schemas ====================

class TokenUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[float] = None


class MessageResponse(BaseModel):
    """Single message response."""
    id: str
    role: MessageRole
    content: str
    model: Optional[str] = None
    created_at: datetime

    # Token usage
    usage: Optional[TokenUsage] = None

    # Function call (if any)
    function_call: Optional[Dict[str, Any]] = None

    # RAG context (if used)
    sources: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    """Chat completion response."""
    id: str
    conversation_id: str
    message: MessageResponse

    # Metadata
    model: str
    provider: str
    finish_reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_xyz789",
                "conversation_id": "conv_abc123",
                "message": {
                    "id": "msg_xyz789",
                    "role": "assistant",
                    "content": "안녕하세요! 오늘 날씨에 대해 알려드릴게요...",
                    "model": "gpt-4-turbo-preview",
                    "created_at": "2024-01-15T10:30:00Z",
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 150,
                        "total_tokens": 200,
                        "estimated_cost": 0.006
                    }
                },
                "model": "gpt-4-turbo-preview",
                "provider": "openai",
                "finish_reason": "stop"
            }
        }


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    id: str
    conversation_id: str
    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


class ConversationResponse(BaseModel):
    """Conversation details."""
    id: str
    title: Optional[str]
    model: str
    system_prompt: Optional[str]
    temperature: float
    message_count: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime]


class ConversationListResponse(BaseModel):
    """List of conversations."""
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class ConversationHistoryResponse(BaseModel):
    """Conversation with message history."""
    conversation: ConversationResponse
    messages: List[MessageResponse]
