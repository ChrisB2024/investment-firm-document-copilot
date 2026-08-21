"""make document_tables retrievable

Revision ID: dedf64806f7f
Revises: b764413cd363

Tables become a second retrievable source type alongside document_chunks.

They have to be, because the figures analysts ask about exist nowhere else.
Apple's FY2024 iPhone revenue of $201,183M appears only inside a table: it is
absent from every section's prose, so it never reaches a chunk and hybrid
retrieval cannot find it. Four of the ten questions in the client brief are
numeric comparisons of exactly this kind.

Kept whole rather than chunked. A 10-K revenue table spans three fiscal years
across one row, and chunking would split a figure away from the header that
says which year it belongs to.

Measured across the corpus: 1,724 tables, median 177 tokens, max 1,762 — every
one fits the 8191-token embedding input limit whole, so no table needs
splitting.
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa

from alembic import op

revision: str = 'dedf64806f7f'
down_revision: str | Sequence[str] | None = 'b764413cd363'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_tables",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True),
    )
    # Generated rather than written by the application, so it cannot drift from
    # the content. The regconfig is inlined because a generated column requires
    # an immutable expression. Title and markdown are concatenated: a query like
    # "Apple revenue by product" matches the caption, the figures match the rows.
    op.execute(
        "ALTER TABLE document_tables "
        "ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS ("
        "to_tsvector('english', coalesce(title, '') || ' ' || markdown)"
        ") STORED"
    )
    op.execute(
        "CREATE INDEX ix_document_tables_search_vector "
        "ON document_tables USING gin (search_vector)"
    )
    # Cosine, matching document_chunks and the L2-normalised vectors OpenAI
    # returns.
    op.execute(
        "CREATE INDEX ix_document_tables_embedding_hnsw "
        "ON document_tables USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_tables_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_document_tables_search_vector")
    op.execute("ALTER TABLE document_tables DROP COLUMN IF EXISTS search_vector")
    op.drop_column("document_tables", "embedding")
