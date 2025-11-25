from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    CSV = "csv"


# ==================== Request Schemas ====================

class DocumentUploadRequest(BaseModel):
    """Document upload metadata."""
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: bool = False
    metadata: Optional[Dict[str, Any]] = None

    # Processing options
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)


class DocumentUpdateRequest(BaseModel):
    """Update document metadata."""
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentSearchRequest(BaseModel):
    """Search documents."""
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.5, ge=0, le=1)
    include_content: bool = True


class DocumentReprocessRequest(BaseModel):
    """Reprocess document with new settings."""
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    embedding_model: Optional[str] = None


# ==================== Response Schemas ====================

class DocumentResponse(BaseModel):
    """Document details."""
    id: str
    filename: str
    original_filename: str
    file_type: DocumentType
    file_size: int
    status: DocumentStatus

    # Content info
    title: Optional[str]
    description: Optional[str]
    page_count: Optional[int]
    word_count: Optional[int]
    chunk_count: int
    language: Optional[str]

    # Metadata
    tags: Optional[List[str]]
    is_public: bool
    metadata: Optional[Dict[str, Any]]

    # Timestamps
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]

    # Error info (if failed)
    error_message: Optional[str]

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "doc_abc123",
                "filename": "company_policy.pdf",
                "original_filename": "Company Policy 2024.pdf",
                "file_type": "pdf",
                "file_size": 1048576,
                "status": "completed",
                "title": "Company Policy 2024",
                "page_count": 25,
                "word_count": 12500,
                "chunk_count": 45,
                "language": "ko",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class DocumentListResponse(BaseModel):
    """Paginated document list."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class DocumentChunkResponse(BaseModel):
    """Document chunk details."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int]
    metadata: Optional[Dict[str, Any]]


class DocumentSearchResult(BaseModel):
    """Single search result."""
    document_id: str
    document_title: Optional[str]
    chunk_id: str
    content: str
    similarity_score: float
    page_number: Optional[int]
    metadata: Optional[Dict[str, Any]]


class DocumentSearchResponse(BaseModel):
    """Document search results."""
    query: str
    results: List[DocumentSearchResult]
    total_results: int
    search_time_ms: float


class DocumentStatsResponse(BaseModel):
    """Document statistics."""
    total_documents: int
    total_chunks: int
    total_storage_bytes: int

    # By status
    pending_count: int
    processing_count: int
    completed_count: int
    failed_count: int

    # By type
    by_file_type: Dict[str, int]

    # Recent activity
    documents_this_week: int
    queries_this_week: int


class UploadProgressResponse(BaseModel):
    """Upload/processing progress."""
    document_id: str
    status: DocumentStatus
    progress_percent: int
    current_step: str  # "uploading", "extracting", "chunking", "embedding"
    error_message: Optional[str]
