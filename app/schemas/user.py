from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


# ==================== Request Schemas ====================

class UserRegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123",
                "full_name": "John Doe"
            }
        }


class UserLoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)


class UserUpdateRequest(BaseModel):
    """Update user profile."""
    full_name: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = Field(None, max_length=500)


class AdminUserUpdateRequest(BaseModel):
    """Admin update user."""
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    token_limit: Optional[int] = Field(None, ge=0)


class RefreshTokenRequest(BaseModel):
    """Refresh access token."""
    refresh_token: str


class APIKeyCreateRequest(BaseModel):
    """Create API key."""
    name: str = Field(..., max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


# ==================== Response Schemas ====================

class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserResponse(BaseModel):
    """User information response."""
    id: str
    email: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserProfileResponse(BaseModel):
    """Detailed user profile."""
    id: str
    email: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    role: UserRole

    # Usage stats
    monthly_token_usage: int
    token_limit: int
    usage_percentage: float

    # Organization
    organization_id: Optional[str]
    organization_name: Optional[str]

    # Timestamps
    created_at: datetime
    last_login_at: Optional[datetime]


class APIKeyResponse(BaseModel):
    """API key response (only shown once on creation)."""
    id: str
    name: str
    key: str  # Only returned on creation
    created_at: datetime
    expires_at: Optional[datetime]


class APIKeyListItem(BaseModel):
    """API key list item (key masked)."""
    id: str
    name: str
    key_preview: str  # e.g., "sk-...abc123"
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]


class UsageStatsResponse(BaseModel):
    """User usage statistics."""
    period: str  # "daily", "weekly", "monthly"
    total_messages: int
    total_tokens: int
    total_conversations: int
    total_documents: int
    estimated_cost: float

    # Breakdown by model
    usage_by_model: dict

    # Daily breakdown
    daily_usage: List[dict]


class OrganizationResponse(BaseModel):
    """Organization details."""
    id: str
    name: str
    slug: str
    description: Optional[str]
    max_users: int
    current_users: int
    monthly_token_limit: int
    monthly_token_usage: int
    is_active: bool
    created_at: datetime


class UserListResponse(BaseModel):
    """Paginated user list."""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int
    has_more: bool
