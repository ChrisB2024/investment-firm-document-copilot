"""drop the unused page column

Revision ID: e1f6c1db7782
Revises: 4d46c2dda720

`page` was declared on `document_chunks` and `message_citations` and never
populated: 0 of 2,321 chunks carry one, because `ingest.chunk` writes
`page=None` and always has.

Not an oversight to fix — the column cannot be filled meaningfully. These are
inline XBRL filings, and an HTML document has no pagination until something
renders it; what counts as "page 47" depends on the viewer. The client brief
asks for "the specific filing and the specific page", and the honest answer to
that requirement is `section`, which *is* populated: 100% of chunks and 94% of
tables. "Item 1A. Risk Factors" is something an analyst can navigate to in the
original filing, where a page number in a 6 MB HTML file is not.

Dropped rather than left empty: a column that always reads NULL looks like a
feature nobody finished, and the next person to touch the citation UI has to
rediscover that it never worked.

Both tables are empty of citations and no chunk has a value, so nothing is lost.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'e1f6c1db7782'
down_revision: str | Sequence[str] | None = '4d46c2dda720'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("document_chunks", "page")
    op.drop_column("message_citations", "page")


def downgrade() -> None:
    """Restores the columns, nullable, which is the state they were in.

    Lossless in both directions: every value was NULL, so re-adding a nullable
    column reproduces the old table exactly.
    """
    op.add_column(
        "message_citations", sa.Column("page", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "document_chunks", sa.Column("page", sa.String(length=64), nullable=True)
    )
