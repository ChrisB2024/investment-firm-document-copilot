from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.constant import EMBEDDING_DIMENSIONS

if TYPE_CHECKING:
    from app.database.message_citation import MessageCitation
    from app.database.source_document import SourceDocument


class DocumentTable(Base):
    """Normalized full table extracted from a source filing."""

    __tablename__ = "document_tables"
    __table_args__ = (
        UniqueConstraint("document_id", "table_index", name="uq_document_tables_document_table"),
        Index("ix_document_tables_document_id", "document_id"),
        # Tables are a second retrievable source type alongside document_chunks:
        # financial figures live only here, and chunking a wide multi-year table
        # would split a row away from its header. Declared here as well as in the
        # migration, or autogenerate sees them only in the database and drops
        # them on the next revision.
        Index("ix_document_tables_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_document_tables_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    units: Mapped[str | None] = mapped_column(String(255))
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    table_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sql_text("'{}'::jsonb"),
    )
    source_html_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )
    # Title and markdown together: a query like "Apple revenue by product" has to
    # match the caption, while the figures themselves live in the rows.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(title, '') || ' ' || markdown)",
            persisted=True,
        ),
        nullable=True,
    )

    document: Mapped[SourceDocument] = relationship(back_populates="tables")
    citations: Mapped[list[MessageCitation]] = relationship(back_populates="table")

    @property
    def embed_text(self) -> str:
        """What gets embedded. Mirrors ingest.extract.Table.embed_text.

        Deliberately the same shape as the search_vector expression above, so
        semantic and lexical retrieval see the same text and cannot disagree
        about what this table says.
        """
        section = (self.table_data or {}).get("section")
        parts = [p for p in (section, self.title, self.units) if p]
        return "\n\n".join([*parts, self.markdown])