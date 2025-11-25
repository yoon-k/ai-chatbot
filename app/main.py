from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import sentry_sdk

from app.core.config import settings
from app.core.exceptions import ChatbotException, raise_http_exception
from app.api.v1.router import api_router
from app.db.session import init_db, close_db


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Initialize Sentry for error tracking
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        traces_sample_rate=0.1
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting MUSE AI Chatbot", version=settings.APP_VERSION)
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down MUSE AI Chatbot")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="MUSE AI Chatbot",
    description="""
# MUSE AI - Enterprise Chatbot Platform

MUSE는 엔터프라이즈급 AI 챗봇 플랫폼입니다.

## Features

- **Multi-LLM Support**: OpenAI GPT, Anthropic Claude 등 다양한 LLM 지원
- **RAG Pipeline**: 문서 기반 질의응답
- **Streaming**: 실시간 응답 스트리밍
- **Authentication**: JWT + API Key 인증
- **Rate Limiting**: 사용량 제한 및 모니터링

## Quick Start

1. 회원가입: `POST /api/v1/auth/register`
2. 로그인: `POST /api/v1/auth/login`
3. 채팅: `POST /api/v1/chat`

## Authentication

모든 API 요청에는 인증이 필요합니다:

```
Authorization: Bearer <access_token>
```

또는 API Key 사용:

```
X-API-Key: sk-...
```
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    """Handle custom chatbot exceptions."""
    logger.warning(
        "Chatbot exception",
        code=exc.code,
        message=exc.message,
        path=request.url.path
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        "Unexpected error",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        }
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "MUSE AI"}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check with dependency verification."""
    from app.db.session import engine

    checks = {
        "database": False,
        "redis": False
    }

    # Check database
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        checks["database"] = True
    except Exception:
        pass

    # Check Redis
    try:
        from app.api.deps import get_redis
        redis_client = await get_redis()
        await redis_client.ping()
        checks["redis"] = True
    except Exception:
        pass

    is_ready = all(checks.values())

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks
    }


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint."""
    return {
        "name": "MUSE AI Chatbot",
        "version": settings.APP_VERSION,
        "description": "Enterprise AI Chatbot Platform",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS
    )
