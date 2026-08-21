"""add source_documents.content_hash

Revision ID: b13d3bb1bb88
Revises: dedf64806f7f

sha256 of the source file plus the extractor version, so ingestion can tell an
unchanged filing from one that needs re-parsing.

Two things fall out of it. A re-run skips extraction and chunking for filings
that have not changed and resumes embedding wherever it stopped, which matters
because embedding is the slow, paid half. And bumping the extractor version in
the hash invalidates every document at once, which is what a parser fix needs:
the same HTML now produces different chunks.

Nullable: rows ingested before this column existed have no hash and will simply
be re-extracted once.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'b13d3bb1bb88'
down_revision: str | Sequence[str] | None = 'dedf64806f7f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "source_documents",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("source_documents", "content_hash")
