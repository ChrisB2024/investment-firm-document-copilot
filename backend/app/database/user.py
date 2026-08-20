"""One row per authenticated user."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

# Imported for the side effect of registering `auth.users` in the metadata,
# without which the foreign key below cannot be compiled.
from app.database.supabase_auth import auth_users  # noqa: F401

if TYPE_CHECKING:
    from app.database.chat_thread import ChatThread


class User(Base):
    __tablename__ = "users"

    # Supabase owns the identity: this is `auth.users.id`, not a value we mint.
    #
    # The FK crosses into Supabase's `auth` schema, which Alembic does not manage
    # — autogenerate cannot see the target, so this constraint is emitted but
    # never verified. It also means migrations require a database that has
    # Supabase Auth installed; a plain Postgres will fail here.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        # Named explicitly: the convention would produce "fk_users_id_users",
        # which reads as self-referential when it crosses schemas.
        ForeignKey("auth.users.id", ondelete="CASCADE", name="fk_users_id_auth_users"),
        primary_key=True,
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chat_threads: Mapped[list[ChatThread]] = relationship(back_populates="owner")
