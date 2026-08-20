"""enable row level security

Revision ID: b764413cd363
Revises: aed0df9ad92c
Create Date: (see git history)

Supabase exposes every table in `public` through PostgREST using the anon key,
which ships inside the frontend bundle and is public by design. Without RLS,
anyone holding that key can read chats and citations directly, bypassing FastAPI
entirely.

Enabling RLS with no policies denies everything to `anon` and `authenticated`.
The backend connects with the service-role key, which bypasses RLS, so this
costs the application nothing.

This was originally switched on by hand in the Supabase dashboard. That does not
survive a rebuild: autogenerate never compares `relrowsecurity`, so a fresh
database would come up unprotected with nothing to warn you. Applying it here
makes it reproducible. `ENABLE ROW LEVEL SECURITY` on an already-enabled table
is a no-op, so this is safe against a database where it is already on.
"""

from collections.abc import Sequence

from alembic import op

revision: str = 'b764413cd363'
down_revision: str | Sequence[str] | None = 'aed0df9ad92c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Every application table. `alembic_version` is deliberately excluded: it is
# Alembic's bookkeeping, not application data, and only the migration role
# touches it.
TABLES = (
    "users",
    "chat_threads",
    "chat_messages",
    "message_citations",
    "source_documents",
    "document_chunks",
    "document_tables",
)


def upgrade() -> None:
    """Enable RLS on every application table."""
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    """Disable RLS.

    This reopens every table to the public anon key. It exists for symmetry;
    think before running it against anything real.
    """
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
