"""Async engine and session factory for direct Postgres access.

This is the SQLAlchemy path, used by ingestion and by anything that needs real
SQL — hybrid retrieval runs pgvector and full-text queries the Supabase client
cannot express. Auth and user-scoped reads go through the Supabase client
instead; see app/database/supabase.py.

The engine is created lazily so importing this module does not open a
connection: Alembic imports the models, and tests import app code, neither of
which should dial the database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


@cache
def engine() -> AsyncEngine:
    """One engine per process. Cached because each call builds a new pool."""
    return create_async_engine(
        settings.alembic_url,
        # Supabase's session pooler already pools; a second pool on top of it
        # holds connections open against a quota we do not control.
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True,
    )


@cache
def _sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine(),
        expire_on_commit=False,  # ingestion reads attributes after commit
        autoflush=False,
    )


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    """A session that rolls back on error and always closes.

    Commits are explicit. A long ingestion commits per batch so a crash leaves
    finished work durable rather than discarding the whole filing.
    """
    async with _sessionmaker()() as s:
        try:
            yield s
        except Exception:
            await s.rollback()
            raise


async def dispose() -> None:
    """Close the pool. Scripts should call this before exiting."""
    if engine.cache_info().currsize:
        await engine().dispose()
