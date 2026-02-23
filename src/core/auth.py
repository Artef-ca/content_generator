"""Authentication utilities for Mobily Content Generator.

This is a simple authentication system that will be replaced with a more robust 
solution in the future. It uses JWT tokens and bcrypt password hashing.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import User, get_db

# Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "mobily-content-generator-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours default

# Password hashing - using pbkdf2_sha256 with minimal rounds for fast demo auth
# Note: This is a simple demo auth system that will be replaced
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto", pbkdf2_sha256__rounds=1000)

# Bearer token security
security = HTTPBearer(auto_error=False)

# Pydantic models for auth
class UserCreate(BaseModel):
    """User registration request."""
    username: str
    password: str
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError('Username cannot be empty')
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if len(v) > 50:
            raise ValueError('Username must be at most 50 characters')
        # Allow alphanumeric, underscore, and hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v.lower().strip()


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        return v.lower().strip()


class UserResponse(BaseModel):
    """User response (without password)."""
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[int] = None
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        username: str = payload.get("username")
        if sub is None:
            return None
        # sub is stored as string, convert to int
        user_id = int(sub) if isinstance(sub, str) else sub
        return TokenData(user_id=user_id, username=username)
    except (JWTError, ValueError):
        return None


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Get a user by username."""
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """Create a new user."""
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        password_hash=hashed_password,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """Authenticate a user by username and password."""
    user = await get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get the current authenticated user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if credentials is None:
        raise credentials_exception
    
    token_data = decode_token(credentials.credentials)
    if token_data is None or token_data.user_id is None:
        raise credentials_exception
    
    user = await get_user_by_id(db, token_data.user_id)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get the current user if authenticated, otherwise return None."""
    import time
    auth_start = time.time()
    
    if credentials is None:
        print(f"[PERF] Auth check (no credentials): {time.time() - auth_start:.3f}s")
        return None
    
    token_data = decode_token(credentials.credentials)
    if token_data is None or token_data.user_id is None:
        print(f"[PERF] Auth check (invalid token): {time.time() - auth_start:.3f}s")
        return None
    
    t_db_query = time.time()
    user = await get_user_by_id(db, token_data.user_id)
    db_query_time = time.time() - t_db_query
    total_time = time.time() - auth_start
    print(f"[PERF] Auth check (DB query): {db_query_time:.3f}s, total: {total_time:.3f}s")
    return user
