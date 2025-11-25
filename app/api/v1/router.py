from fastapi import APIRouter

from app.api.v1.endpoints import chat, auth, documents

api_router = APIRouter()

# Authentication routes
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

# Chat routes
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"]
)

# Document/RAG routes
api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"]
)
