"""citations may reference a table

Revision ID: 4d46c2dda720
Revises: b13d3bb1bb88

A citation can point at a table, and until now the schema said it could not.

`message_citations.chunk_id` is NOT NULL referencing `document_chunks`, but 46%
of a filing's comma-formatted figures appear only inside a table — which is why
Phase 2 made tables a second retrievable source type in the first place. Brief
questions 1, 2 and 8 are numeric comparisons, so the agent cites tables on its
first real run and the citation table physically cannot hold the row.

Both stay real foreign keys rather than collapsing to a `(source_type, row_id)`
pair. message_citation.py's docstring is the reason: a citation that cannot be
resolved must be a database error, not a rendering surprise. An unconstrained
pair gives that up to save one nullable column.

Also adds `handle`. The assistant's prose contains "[S3]" and nothing persisted
maps S3 to a row. `citation_index` cannot: one passage legitimately supports two
claims, which is two rows carrying one handle, so the two are not the same
number. The alternative was renumbering the model's prose at persist time, which
is rewriting the answer to fit the schema.

Cheap now, expensive later: all four chat tables are empty (0 rows in
message_citations, chat_messages, chat_threads, users), so this is a plain
column change with no backfill. After the pilot has run it is a three-step
migration against real research history.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '4d46c2dda720'
down_revision: str | Sequence[str] | None = 'b13d3bb1bb88'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHECK = "ck_message_citations_one_source"
_FK = "fk_message_citations_table_id_document_tables"
_INDEX = "ix_message_citations_table_id"


def upgrade() -> None:
    op.alter_column(
        "message_citations", "chunk_id", existing_type=sa.UUID(), nullable=True
    )
    op.add_column("message_citations", sa.Column("table_id", sa.UUID(), nullable=True))
    # RESTRICT, matching chunk_id and for the same reason: deleting a cited
    # source must fail loudly rather than cascade a hole into an analyst's
    # saved research.
    op.create_foreign_key(
        op.f(_FK),
        "message_citations",
        "document_tables",
        ["table_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    # Every foreign key in this schema has one, and citation-by-source lookup
    # needs it.
    op.create_index(_INDEX, "message_citations", ["table_id"], unique=False)
    # Last, and free only because the table is empty: a CHECK cannot be added
    # while a row could violate it. Named through the convention in base.py,
    # which expands "one_source" to _CHECK, so `message_citation.py` can
    # declare the same constraint — it has to, because Alembic does not compare
    # check constraints and would otherwise never see this one.
    op.create_check_constraint(
        "one_source", "message_citations", "num_nonnulls(chunk_id, table_id) = 1"
    )
    # NOT NULL with no backfill and no default: a citation with no handle cannot
    # be rendered against the prose, so there is no valid row without one, and
    # there are no rows to rewrite.
    op.add_column(
        "message_citations", sa.Column("handle", sa.String(length=8), nullable=False)
    )


def downgrade() -> None:
    """Refuses while a table citation exists, rather than deleting it.

    Reversing this is lossy in a way the upgrade is not: a table citation has no
    representation in the old schema, so `chunk_id` cannot be made NOT NULL
    again while one exists. Silently deleting an analyst's citations is worse
    than failing, so this says which rows block it and stops.
    """
    blocking = (
        op.get_bind()
        .execute(
            sa.text("SELECT count(*) FROM message_citations WHERE table_id IS NOT NULL")
        )
        .scalar_one()
    )
    if blocking:
        raise RuntimeError(
            f"{blocking} citation(s) reference a table and cannot exist in the "
            "pre-4d46c2dda720 schema. Delete them deliberately, then rerun this "
            "downgrade."
        )

    op.drop_column("message_citations", "handle")
    # op.f() on every name below: these were written by the naming convention,
    # and without it Alembic runs them back through it and looks for
    # ck_message_citations_ck_message_citations_one_source.
    op.drop_constraint(op.f(_CHECK), "message_citations", type_="check")
    op.drop_index(_INDEX, table_name="message_citations")
    op.drop_constraint(op.f(_FK), "message_citations", type_="foreignkey")
    op.drop_column("message_citations", "table_id")
    op.alter_column(
        "message_citations", "chunk_id", existing_type=sa.UUID(), nullable=False
    )
