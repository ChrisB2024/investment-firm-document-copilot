"""Minimal stand-in for Supabase's `auth.users` table.

Supabase owns the `auth` schema; we neither create nor migrate it. But a foreign
key can only be compiled if SQLAlchemy can resolve its target, so the table has
to exist in the metadata as a stub — just the column we reference.

`alembic/env.py` filters the `auth` schema out of autogenerate, so this stub
never turns into a CREATE TABLE. Do not add columns here to "match" the real
table: nothing keeps them in sync, and a wrong stub is worse than a thin one.
"""

from sqlalchemy import Column, Table
from sqlalchemy.dialects.postgresql import UUID

from app.database.base import Base

AUTH_SCHEMA = "auth"

auth_users = Table(
    "users",
    Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    schema=AUTH_SCHEMA,
)

__all__ = ["AUTH_SCHEMA", "auth_users"]
