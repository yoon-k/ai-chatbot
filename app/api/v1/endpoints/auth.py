import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.core.security import (
    get_password_hash, verify_password,
    create_access_token, create_refresh_token,
    decode_token, get_current_user, TokenData,
    generate_api_key, hash_api_key
)
from app.core.config import settings
from app.models.user import User, UserRole, AuthProvider, RefreshToken
from app.schemas.user import (
    UserRegisterRequest, UserLoginRequest, TokenResponse,
    UserResponse, UserProfileResponse, PasswordChangeRequest,
    RefreshTokenRequest, APIKeyCreateRequest, APIKeyResponse,
    APIKeyListItem
)

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
async def register(
    request: UserRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user account."""

    # Check if email exists
    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    user = User(
        id=str(uuid.uuid4()),
        email=request.email,
        hashed_password=get_password_hash(request.password),
        full_name=request.full_name,
        role=UserRole.USER,
        auth_provider=AuthProvider.LOCAL
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Generate tokens
    token_data = {"sub": user.id, "email": user.email, "role": user.role.value}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Save refresh token
    await _save_refresh_token(db, user.id, refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: UserLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate and get access token."""

    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.commit()

    # Generate tokens
    token_data = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "organization_id": user.organization_id
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    await _save_refresh_token(db, user.id, refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""

    try:
        payload = decode_token(request.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )

    # Verify refresh token exists and is valid
    token_hash = hash_api_key(request.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    stored_token = result.scalar_one_or_none()

    if not stored_token or not stored_token.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired or revoked"
        )

    # Get user
    user = await db.get(User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Revoke old token
    stored_token.revoked_at = datetime.utcnow()

    # Generate new tokens
    token_data = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "organization_id": user.organization_id
    }
    new_access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)

    await _save_refresh_token(db, user.id, new_refresh_token)
    await db.commit()

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Logout and revoke all refresh tokens."""

    # Revoke all user's refresh tokens
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == current_user.user_id,
            RefreshToken.revoked_at.is_(None)
        )
    )
    tokens = result.scalars().all()

    for token in tokens:
        token.revoked_at = datetime.utcnow()

    await db.commit()

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserProfileResponse)
async def get_profile(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user profile."""

    user = await db.get(User, current_user.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    org_name = None
    if user.organization_id:
        from app.models.user import Organization
        org = await db.get(Organization, user.organization_id)
        org_name = org.name if org else None

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        role=user.role,
        monthly_token_usage=user.monthly_token_usage,
        token_limit=user.token_limit,
        usage_percentage=(user.monthly_token_usage / user.token_limit) * 100 if user.token_limit else 0,
        organization_id=user.organization_id,
        organization_name=org_name,
        created_at=user.created_at,
        last_login_at=user.last_login_at
    )


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    full_name: str = None,
    avatar_url: str = None,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user profile."""

    user = await db.get(User, current_user.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if full_name is not None:
        user.full_name = full_name
    if avatar_url is not None:
        user.avatar_url = avatar_url

    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        last_login_at=user.last_login_at
    )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change current user password."""

    user = await db.get(User, current_user.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(request.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    user.hashed_password = get_password_hash(request.new_password)
    await db.commit()

    return {"message": "Password changed successfully"}


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate a new API key."""

    user = await db.get(User, current_user.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Generate key
    api_key = generate_api_key()
    user.api_key_hash = hash_api_key(api_key)
    await db.commit()

    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    return APIKeyResponse(
        id=str(uuid.uuid4()),
        name=request.name,
        key=api_key,  # Only shown once!
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )


async def _save_refresh_token(
    db: AsyncSession,
    user_id: str,
    token: str
) -> None:
    """Save refresh token to database."""
    refresh_token = RefreshToken(
        id=str(uuid.uuid4()),
        user_id=user_id,
        token_hash=hash_api_key(token),
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    db.add(refresh_token)
    await db.commit()
