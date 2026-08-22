"""Tests for ingest.persist.

These write to a real database, so they are all integration tests. Run with:

    uv run pytest -m integration

Each must clean up after itself — a leftover row makes the next run's
idempotency assertions lie. Prefer a transaction rolled back at the end over
DELETEs that can themselves fail halfway.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_upsert_document_inserts_then_updates(manifest_entry):
    """TODO: upsert twice with different markdown and assert one row whose id
    is unchanged.

    The id has to survive: message_citations point at chunks that point at this
    document, so a delete-then-insert would orphan an analyst's citations on
    every re-ingest.
    """
    raise NotImplementedError


async def test_fiscal_year_comes_from_report_date(manifest_entry):
    """TODO: assert a filing dated 2025-02-07 reporting on 2024-12-31 stores
    fiscal_year 2024.

    Using filing_date would label every Amazon and Alphabet filing a year late,
    and "compare 2024 across companies" would silently mix periods.
    """
    raise NotImplementedError


async def test_replace_chunks_is_idempotent(manifest_entry):
    """TODO: write chunks twice and assert the count does not double.

    Getting this wrong is silent: retrieval returns duplicates, each with a
    plausible citation.
    """
    raise NotImplementedError


async def test_replace_chunks_removes_orphans(manifest_entry):
    """TODO: write 10 chunks, then 5, and assert the last 5 are gone.

    An upsert alone leaves chunks at indices the new extraction no longer
    produces — stale text that still retrieves.
    """
    raise NotImplementedError


async def test_pending_chunks_returns_only_unembedded(manifest_entry):
    """TODO: assert chunks with an embedding are excluded and the rest come
    back in chunk_index order.

    This is what makes a run resumable rather than re-paying for a filing.
    """
    raise NotImplementedError


async def test_generated_search_vector_populates(manifest_entry):
    """TODO: insert a chunk and assert search_vector is non-null and matches a
    plainto_tsquery for a word in its text.

    Postgres maintains it, so this is really a guard against the migration
    drifting away from the model.
    """
    raise NotImplementedError


async def test_duplicate_chunk_index_is_rejected(manifest_entry):
    """TODO: assert inserting two chunks with the same (document_id,
    chunk_index) raises IntegrityError."""
    raise NotImplementedError


async def test_tables_persist_with_rows_and_hash(manifest_entry):
    """TODO: assert replace_tables stores markdown, rows, section and
    source_html_hash, and that re-running does not duplicate."""
    raise NotImplementedError
