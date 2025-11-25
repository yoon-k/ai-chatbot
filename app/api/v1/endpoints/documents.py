import uuid
import os
import hashlib
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.session import get_db
from app.api.deps import get_current_user_with_rate_limit
from app.core.security import TokenData
from app.core.config import settings
from app.models.document import Document, DocumentStatus, DocumentType
from app.schemas.document import (
    DocumentResponse, DocumentListResponse, DocumentUpdateRequest,
    DocumentSearchRequest, DocumentSearchResponse, DocumentSearchResult,
    UploadProgressResponse
)
from app.services.rag.pipeline import RAGPipeline

router = APIRouter()

UPLOAD_DIR = "/tmp/muse_uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md", "html"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    is_public: bool = Form(False),
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Upload a document for RAG processing.

    Supported formats: PDF, DOCX, TXT, MD, HTML
    Maximum file size: 50MB

    The document will be processed asynchronously:
    1. Text extraction
    2. Chunking
    3. Embedding generation
    4. Vector storage
    """
    # Validate file extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )

    # Generate file hash
    file_hash = hashlib.sha256(contents).hexdigest()

    # Check for duplicate
    result = await db.execute(
        select(Document).where(
            Document.user_id == current_user.user_id,
            Document.file_hash == file_hash
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Document already exists"
        )

    # Save file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(contents)

    # Create document record
    document = Document(
        id=doc_id,
        user_id=current_user.user_id,
        filename=filename,
        original_filename=file.filename,
        file_type=DocumentType(ext),
        file_size=len(contents),
        file_hash=file_hash,
        storage_path=file_path,
        title=title or file.filename,
        description=description,
        tags=tags,
        is_public=is_public,
        status=DocumentStatus.PENDING
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Start background processing
    background_tasks.add_task(
        process_document_task,
        document.id,
        file_path,
        ext,
        current_user.user_id
    )

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        original_filename=document.original_filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        title=document.title,
        description=document.description,
        page_count=document.page_count,
        word_count=document.word_count,
        chunk_count=document.chunk_count,
        language=document.language,
        tags=document.tags.split(",") if document.tags else None,
        is_public=document.is_public,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
        processed_at=document.processed_at,
        error_message=document.error_message
    )


async def process_document_task(
    document_id: str,
    file_path: str,
    file_type: str,
    user_id: str
):
    """Background task for document processing."""
    from app.db.session import async_session_maker

    async with async_session_maker() as db:
        document = await db.get(Document, document_id)
        if not document:
            return

        try:
            document.status = DocumentStatus.PROCESSING
            await db.commit()

            # Process with RAG pipeline
            rag = RAGPipeline()
            result = await rag.process_document(
                file_path=file_path,
                file_type=file_type,
                document_id=document_id,
                user_id=user_id
            )

            # Update document
            document.status = DocumentStatus.COMPLETED
            document.chunk_count = result["chunk_count"]
            document.collection_name = result["collection_name"]
            document.processed_at = datetime.utcnow()
            await db.commit()

        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            await db.commit()


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: Optional[DocumentStatus] = None,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """List user's documents."""
    # Build query
    conditions = [Document.user_id == current_user.user_id]
    if status:
        conditions.append(Document.status == status)

    # Count total
    count_query = select(func.count(Document.id)).where(*conditions)
    total = (await db.execute(count_query)).scalar()

    # Get page
    offset = (page - 1) * page_size
    query = (
        select(Document)
        .where(*conditions)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                original_filename=d.original_filename,
                file_type=d.file_type,
                file_size=d.file_size,
                status=d.status,
                title=d.title,
                description=d.description,
                page_count=d.page_count,
                word_count=d.word_count,
                chunk_count=d.chunk_count,
                language=d.language,
                tags=d.tags.split(",") if d.tags else None,
                is_public=d.is_public,
                metadata=d.metadata,
                created_at=d.created_at,
                updated_at=d.updated_at,
                processed_at=d.processed_at,
                error_message=d.error_message
            )
            for d in documents
        ],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(documents)) < total
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Get document details."""
    document = await db.get(Document, document_id)
    if not document or (document.user_id != current_user.user_id and not document.is_public):
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        original_filename=document.original_filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        title=document.title,
        description=document.description,
        page_count=document.page_count,
        word_count=document.word_count,
        chunk_count=document.chunk_count,
        language=document.language,
        tags=document.tags.split(",") if document.tags else None,
        is_public=document.is_public,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
        processed_at=document.processed_at,
        error_message=document.error_message
    )


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Update document metadata."""
    document = await db.get(Document, document_id)
    if not document or document.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    if request.title is not None:
        document.title = request.title
    if request.description is not None:
        document.description = request.description
    if request.tags is not None:
        document.tags = ",".join(request.tags)
    if request.is_public is not None:
        document.is_public = request.is_public
    if request.metadata is not None:
        document.metadata = request.metadata

    await db.commit()
    await db.refresh(document)

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        original_filename=document.original_filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        title=document.title,
        description=document.description,
        page_count=document.page_count,
        word_count=document.word_count,
        chunk_count=document.chunk_count,
        language=document.language,
        tags=document.tags.split(",") if document.tags else None,
        is_public=document.is_public,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
        processed_at=document.processed_at,
        error_message=document.error_message
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and its vectors."""
    document = await db.get(Document, document_id)
    if not document or document.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from vector store
    rag = RAGPipeline()
    await rag.delete_document(document_id, current_user.user_id)

    # Delete file
    if os.path.exists(document.storage_path):
        os.remove(document.storage_path)

    # Delete from database
    await db.delete(document)
    await db.commit()

    return {"message": "Document deleted"}


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Search across documents using semantic search."""
    import time

    start_time = time.time()

    rag = RAGPipeline()
    results = await rag.retrieve(
        query=request.query,
        document_ids=request.document_ids or [],
        user_id=current_user.user_id,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold
    )

    search_time = (time.time() - start_time) * 1000

    # Get document titles
    doc_titles = {}
    if results:
        doc_ids = list(set(r["document_id"] for r in results))
        docs_result = await db.execute(
            select(Document.id, Document.title).where(Document.id.in_(doc_ids))
        )
        doc_titles = {row[0]: row[1] for row in docs_result}

    return DocumentSearchResponse(
        query=request.query,
        results=[
            DocumentSearchResult(
                document_id=r["document_id"],
                document_title=doc_titles.get(r["document_id"]),
                chunk_id=f"{r['document_id']}_{r['chunk_index']}",
                content=r["content"] if request.include_content else "",
                similarity_score=r["similarity_score"],
                page_number=r["metadata"].get("page_number"),
                metadata=r["metadata"]
            )
            for r in results
        ],
        total_results=len(results),
        search_time_ms=search_time
    )


@router.get("/{document_id}/status", response_model=UploadProgressResponse)
async def get_processing_status(
    document_id: str,
    current_user: TokenData = Depends(get_current_user_with_rate_limit),
    db: AsyncSession = Depends(get_db)
):
    """Get document processing status."""
    document = await db.get(Document, document_id)
    if not document or document.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    # Estimate progress based on status
    progress_map = {
        DocumentStatus.PENDING: 0,
        DocumentStatus.PROCESSING: 50,
        DocumentStatus.COMPLETED: 100,
        DocumentStatus.FAILED: 0
    }

    step_map = {
        DocumentStatus.PENDING: "waiting",
        DocumentStatus.PROCESSING: "embedding",
        DocumentStatus.COMPLETED: "completed",
        DocumentStatus.FAILED: "failed"
    }

    return UploadProgressResponse(
        document_id=document_id,
        status=document.status,
        progress_percent=progress_map.get(document.status, 0),
        current_step=step_map.get(document.status, "unknown"),
        error_message=document.error_message
    )
