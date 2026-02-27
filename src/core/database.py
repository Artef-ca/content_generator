"""Database configuration and models for Mobily Content Generator."""

import os
from datetime import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import String, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import enum

# Database URL - supports both Cloud SQL and local SQLite for development
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./mobily_content.db"  # Default to SQLite for local dev
)

# Detect if we're using Cloud SQL (Unix socket connection)
# Cloud SQL URL format: postgresql+asyncpg://user:password@/dbname?host=/cloudsql/PROJECT:REGION:INSTANCE
USE_CLOUD_SQL = DATABASE_URL.startswith("postgresql") and "/cloudsql/" in DATABASE_URL

# Create async engine with Cloud SQL connector if needed
if USE_CLOUD_SQL:
    # Extract Cloud SQL instance from DATABASE_URL
    # Format: postgresql+asyncpg://user:password@/dbname?host=/cloudsql/PROJECT:REGION:INSTANCE
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(DATABASE_URL)
    query_params = parse_qs(parsed.query)
    cloud_sql_instance = query_params.get("host", [None])[0]
    
    if cloud_sql_instance and cloud_sql_instance.startswith("/cloudsql/"):
        # Remove /cloudsql/ prefix to get instance name
        instance_name = cloud_sql_instance.replace("/cloudsql/", "")
        print(f"[INFO] Using Cloud SQL: {instance_name}")
        
        # Use Cloud SQL connector for proper connection handling
        from google.cloud.sql.connector import Connector
        import asyncpg
        
        connector = Connector()
        
        async def get_cloud_sql_conn():
            """Get Cloud SQL connection using the connector."""
            # Extract connection details from DATABASE_URL
            user = parsed.username
            # Password can come from URL or environment variable/secret
            password = parsed.password or os.environ.get("CLOUD_SQL_PASSWORD", "")
            database = parsed.path.lstrip("/")
            
            if not password:
                raise ValueError(
                    "Cloud SQL password not found. Set it in DATABASE_URL or CLOUD_SQL_PASSWORD environment variable."
                )
            
            conn = await connector.connect_async(
                instance_name,
                "asyncpg",
                user=user,
                password=password,
                db=database,
            )
            return conn
        
        # Create engine with Cloud SQL connector
        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=get_cloud_sql_conn,
            echo=os.environ.get("DEBUG", "false").lower() == "true",
        )
    else:
        # Direct PostgreSQL connection (not Cloud SQL)
        print(f"[INFO] Using PostgreSQL connection")
        engine = create_async_engine(
            DATABASE_URL,
            echo=os.environ.get("DEBUG", "false").lower() == "true",
        )
else:
    # SQLite for local development
    print(f"[INFO] Using SQLite for local development")
    engine = create_async_engine(
        DATABASE_URL,
        echo=os.environ.get("DEBUG", "false").lower() == "true",
    )

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class FeedbackRating(enum.Enum):
    """Enum for feedback ratings."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    content_history: Mapped[list["ContentHistory"]] = relationship(
        "ContentHistory", back_populates="user", cascade="all, delete-orphan"
    )
    feedback: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="user", cascade="all, delete-orphan"
    )


class ContentHistory(Base):
    """Content history model to store generated content."""
    __tablename__ = "content_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    request_params: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string of request
    generated_content: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(String(50), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="content_history")
    feedback: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="content_history", cascade="all, delete-orphan"
    )


class Feedback(Base):
    """Feedback model for user ratings on generated content."""
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content_history_id: Mapped[int] = mapped_column(
        ForeignKey("content_history.id"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    rating: Mapped[str] = mapped_column(
        SQLEnum(FeedbackRating, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    variation_index: Mapped[Optional[int]] = mapped_column(nullable=True)  # Which variation was rated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    content_history: Mapped["ContentHistory"] = relationship(
        "ContentHistory", back_populates="feedback"
    )
    user: Mapped["User"] = relationship("User", back_populates="feedback")


class ImageHistory(Base):
    """Stores every generated and refined image for session-based history."""
    __tablename__ = "image_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    gcs_uri: Mapped[str] = mapped_column(Text, nullable=False)
    public_url: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_commentary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("image_history.id"), nullable=True)
    variation_index: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


async def init_db():
    """Initialize database tables and create demo, prepaid, and postpaid users."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create demo, prepaid, and postpaid users if they don't exist
    from auth import get_password_hash
    async with async_session_maker() as session:
        from sqlalchemy import select
        
        # Create demo user
        result = await session.execute(select(User).where(User.username == "demo"))
        if not result.scalar_one_or_none():
            demo_user = User(
                username="demo",
                password_hash=get_password_hash("demo123")
            )
            session.add(demo_user)
            print("Demo user created: demo / demo123")
        
        # Create prepaid user
        result = await session.execute(select(User).where(User.username == "prepaid"))
        if not result.scalar_one_or_none():
            prepaid_user = User(
                username="prepaid",
                password_hash=get_password_hash("prepaid123")
            )
            session.add(prepaid_user)
            print("Prepaid user created: prepaid / prepaid123")
        
        # Create postpaid user
        result = await session.execute(select(User).where(User.username == "postpaid"))
        if not result.scalar_one_or_none():
            postpaid_user = User(
                username="postpaid",
                password_hash=get_password_hash("postpaid123")
            )
            session.add(postpaid_user)
            print("Postpaid user created: postpaid / postpaid123")
        
        await session.commit()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    import time
    db_start = time.time()
    async with async_session_maker() as session:
        session_time = time.time() - db_start
        if session_time > 0.01:  # Only log if it takes more than 10ms
            print(f"[PERF] DB session created: {session_time:.3f}s")
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
