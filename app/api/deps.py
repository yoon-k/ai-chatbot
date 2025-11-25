from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from datetime import datetime

from app.db.session import get_db
from app.core.security import get_current_user, TokenData, verify_api_key
from app.core.config import settings
from app.core.exceptions import RateLimitError


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Redis connection pool
redis_pool: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get Redis connection."""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
    return redis_pool


async def get_current_user_or_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> TokenData:
    """Authenticate via JWT token or API key."""

    # Try JWT first
    if credentials:
        return await get_current_user(credentials)

    # Try API key
    if api_key:
        from app.models.user import User
        from sqlalchemy import select

        # Find user by API key hash
        from app.core.security import hash_api_key
        key_hash = hash_api_key(api_key)

        result = await db.execute(
            select(User).where(User.api_key_hash == key_hash)
        )
        user = result.scalar_one_or_none()

        if user and user.is_active:
            return TokenData(
                user_id=user.id,
                email=user.email,
                role=user.role.value,
                organization_id=user.organization_id
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )


class RateLimiter:
    """Rate limiting middleware."""

    def __init__(
        self,
        requests_per_minute: int = None,
        requests_per_day: int = None
    ):
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self.requests_per_day = requests_per_day or settings.RATE_LIMIT_PER_DAY

    async def __call__(
        self,
        request: Request,
        current_user: TokenData = Depends(get_current_user_or_api_key),
        redis_client: redis.Redis = Depends(get_redis)
    ) -> TokenData:
        """Check rate limits for current user."""

        if not settings.RATE_LIMIT_ENABLED:
            return current_user

        user_id = current_user.user_id
        now = datetime.utcnow()

        # Check minute limit
        minute_key = f"rate_limit:{user_id}:minute:{now.strftime('%Y%m%d%H%M')}"
        minute_count = await redis_client.incr(minute_key)
        if minute_count == 1:
            await redis_client.expire(minute_key, 60)

        if minute_count > self.requests_per_minute:
            raise RateLimitError(retry_after=60)

        # Check daily limit
        day_key = f"rate_limit:{user_id}:day:{now.strftime('%Y%m%d')}"
        day_count = await redis_client.incr(day_key)
        if day_count == 1:
            await redis_client.expire(day_key, 86400)

        if day_count > self.requests_per_day:
            raise RateLimitError(retry_after=3600)

        return current_user


# Default rate limiter
rate_limiter = RateLimiter()


async def get_current_user_with_rate_limit(
    request: Request,
    current_user: TokenData = Depends(get_current_user_or_api_key),
    redis_client: redis.Redis = Depends(get_redis)
) -> TokenData:
    """Get current user with rate limiting."""
    return await rate_limiter(request, current_user, redis_client)
