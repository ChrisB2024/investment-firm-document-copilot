"""Links an assistant claim to the passage that supports it.

This is the product's trust guarantee, so it is a real foreign key rather than
JSON on the message: a citation that cannot be resolved must be a database
error, not a rendering surprise.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

if TYPE_CHECKING:
    from app.database.chat_message import ChatMessage
    from app.database.document_chunk import DocumentChunk
    from app.database.document_table import DocumentTable


class MessageCitation(Base):
    __tablename__ = "message_citations"
    __table_args__ = (
        Index("ix_message_citations_message_id", "message_id"),
        Index("ix_message_citations_chunk_id", "chunk_id"),
        Index("ix_message_citations_table_id", "table_id"),
        # Declared here *and* written in migration 4d46c2dda720: Alembic does
        # not compare check constraints, so one that lives only in the database
        # is dropped by the next autogenerate and one that lives only here is
        # never created. `num_nonnulls` over `(chunk_id IS NULL) <> (table_id IS
        # NULL)` because it reads as what it means. The naming convention in
        # base.py expands the name to ck_message_citations_one_source.
        CheckConstraint("num_nonnulls(chunk_id, table_id) = 1", name="one_source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Exactly one of these two is set, enforced by the CHECK above. Tables are a
    # second retrievable source type, and the figures the brief asks about live
    # only inside one — so a citation that cannot name a table is a citation the
    # agent cannot write. Both stay real foreign keys rather than collapsing to
    # a `(source_type, row_id)` pair, for the reason in the module docstring: an
    # unconstrained pair gives up resolvability to save one nullable column.
    #
    # RESTRICT on both: deleting a cited source must fail loudly rather than
    # cascade a hole into an analyst's saved research.
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="RESTRICT"),
    )
    table_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_tables.id", ondelete="RESTRICT"),
    )
    # What the prose in chat_messages.content actually contains: "[S3]".
    # citation_index cannot carry it — one passage legitimately supports two
    # claims, which is two rows under one handle.
    handle: Mapped[str] = mapped_column(String(8), nullable=False)
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255))
    form: Mapped[str] = mapped_column(String(16), nullable=False)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    message: Mapped[ChatMessage] = relationship(back_populates="citations")
    chunk: Mapped[DocumentChunk | None] = relationship(back_populates="citations")
    table: Mapped[DocumentTable | None] = relationship(back_populates="citations")
