"""Tests for ingest.persist.

These write to a real database, so they are all integration tests. Run with:

    uv run pytest -m integration

Each must clean up after itself — a leftover row makes the next run's
idempotency assertions lie. Prefer a transaction rolled back at the end over
DELETEs that can themselves fail halfway.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.database.constant import EMBEDDING_DIMENSIONS
from app.database.models import DocumentChunk, DocumentTable, SourceDocument
from app.database.session import dispose, engine, session
from ingest.chunk import Chunk
from ingest.extract import Table
from ingest.persist import (
    counts,
    fetch_document,
    pending_chunks,
    pending_tables,
    replace_chunks,
    replace_tables,
    upsert_document,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def db():
    """A session whose work is always discarded.

    Nothing here commits, so rolling back at the end undoes every write. This
    is why the tests need no cleanup code: a DELETE in a finally block can
    itself fail halfway and leave exactly the rows it was meant to remove.

    The engine cache is cleared afterwards because it is process-wide while
    pytest-asyncio gives each test its own event loop — a pooled connection
    opened on a closed loop fails on the next test, not this one.
    """
    try:
        async with session() as connection:
            try:
                yield connection
            finally:
                await connection.rollback()
    finally:
        await dispose()
        engine.cache_clear()


@pytest.fixture
def entry(manifest_entry) -> dict:
    """manifest_entry with an accession number unique to this run.

    The rollback should make this unnecessary. It is here for when it is not:
    a unique number means a row that escapes cannot collide with real corpus
    data or with the next run's assertions.
    """
    return {**manifest_entry, "accession_number": f"TEST-{uuid4().hex[:20]}"}


def _chunks(count: int, start: int = 0) -> list[Chunk]:
    return [
        Chunk(
            chunk_index=i,
            text=f"Chunk {i}: the Company depends on a single supplier.",
            token_count=12,
            section="Item 1A. Risk Factors",
            metadata={"ticker": "AAPL", "fiscal_year": 2024},
        )
        for i in range(start, start + count)
    ]


async def test_upsert_document_inserts_then_updates(db, entry):
    """Upserting twice leaves one row whose id is unchanged.

    The id has to survive: message_citations point at chunks that point at this
    document, so a delete-then-insert would orphan an analyst's citations on
    every re-ingest.
    """
    first = await upsert_document(db, entry, "# First pass", "hash-1")
    second = await upsert_document(db, entry, "# Second pass", "hash-2")

    assert first == second, "the row was replaced, not updated"

    total = await db.execute(
        select(func.count()).select_from(SourceDocument).where(
            SourceDocument.accession_number == entry["accession_number"]
        )
    )
    assert total.scalar_one() == 1

    row = await fetch_document(db, entry["accession_number"])
    assert row.id == first
    assert row.markdown_content == "# Second pass"
    assert row.content_hash == "hash-2", "a re-run must move the hash forward"


async def test_fiscal_year_comes_from_report_date(db, entry):
    """A filing dated 2025-02-07 reporting on 2024-12-31 stores fiscal_year 2024.

    Using filing_date would label every Amazon and Alphabet filing a year late,
    and "compare 2024 across companies" would silently mix periods.
    """
    amazon = {**entry, "ticker": "AMZN", "filing_date": "2025-02-07",
              "report_date": "2024-12-31"}

    await upsert_document(db, amazon, "", "hash")
    row = await fetch_document(db, amazon["accession_number"])

    assert row.fiscal_year == 2024
    # Without this the test passes just as well on a filing_date read, because
    # for AAPL the two dates fall in the same calendar year.
    assert row.filing_date.year == 2025


async def test_replace_chunks_is_idempotent(db, entry):
    """Writing the same chunks twice does not double the count.

    Getting this wrong is silent: retrieval returns duplicates, each with a
    plausible citation.
    """
    document_id = await upsert_document(db, entry, "", "hash")

    await replace_chunks(db, document_id, _chunks(10))
    await replace_chunks(db, document_id, _chunks(10))

    chunk_count, _ = await counts(db, document_id)
    assert chunk_count == 10


async def test_replace_chunks_removes_orphans(db, entry):
    """A re-extraction producing fewer chunks leaves none behind.

    An upsert alone leaves chunks at indices the new extraction no longer
    produces — stale text that still retrieves, still cites, and no longer
    appears anywhere in the filing.
    """
    document_id = await upsert_document(db, entry, "", "hash")

    await replace_chunks(db, document_id, _chunks(10))
    await replace_chunks(db, document_id, _chunks(5))

    remaining = await db.execute(
        select(DocumentChunk.chunk_index)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    assert list(remaining.scalars()) == [0, 1, 2, 3, 4]


async def test_pending_chunks_returns_only_unembedded(db, entry):
    """Embedded chunks are excluded; the rest come back in chunk_index order.

    This is what makes a run resumable rather than re-paying for a filing.
    Order matters because embed_chunks batches positionally — the caller zips
    vectors back onto exactly this list.
    """
    document_id = await upsert_document(db, entry, "", "hash")

    chunks = _chunks(4)
    chunks[1].embedding = [0.5] * EMBEDDING_DIMENSIONS
    # Inserted out of order, so ORDER BY is doing the work rather than the
    # insertion sequence happening to agree with it.
    await replace_chunks(db, document_id, [chunks[3], chunks[1], chunks[0], chunks[2]])

    pending = await pending_chunks(db, document_id)

    assert [c.chunk_index for c in pending] == [0, 2, 3]
    assert all(c.embedding is None for c in pending)


async def test_generated_search_vector_populates(db, entry):
    """Postgres fills search_vector from text, and it is queryable.

    Postgres maintains it, so this is really a guard against the migration
    drifting away from the model: the column is declared Computed on
    DocumentChunk and created by hand in the migration, and nothing but a
    live insert proves the two still agree.
    """
    document_id = await upsert_document(db, entry, "", "hash")
    await replace_chunks(db, document_id, _chunks(1))

    vector = await db.execute(
        select(DocumentChunk.search_vector).where(DocumentChunk.document_id == document_id)
    )
    assert vector.scalar_one() is not None

    async def matches(query: str) -> int:
        result = await db.execute(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.search_vector.op("@@")(func.plainto_tsquery("english", query)),
            )
        )
        return result.scalar_one()

    assert await matches("single supplier") == 1
    # Stemming is what makes "depends" find "depend"; a plain LIKE would not.
    assert await matches("depend") == 1
    assert await matches("semiconductor litigation") == 0


async def test_duplicate_chunk_index_is_rejected(db, entry):
    """Two chunks sharing (document_id, chunk_index) violate the constraint.

    That constraint is what makes replace_chunks idempotent and what makes
    neighbour lookup unambiguous, so it has to be enforced by the database
    rather than assumed by the writer.
    """
    document_id = await upsert_document(db, entry, "", "hash")
    collision = _chunks(1) + _chunks(1)

    with pytest.raises(IntegrityError):
        await replace_chunks(db, document_id, collision)


async def test_tables_persist_with_rows_and_hash(db, entry):
    """replace_tables stores markdown, rows, section and hash, and re-runs clean.

    The rows and the section live in one JSONB column, and the model's
    embed_text reads them back out. It has to reproduce what
    ingest.extract.Table.embed_text sent to the embedding API, or lexical and
    semantic retrieval disagree about what the table says.
    """
    document_id = await upsert_document(db, entry, "", "hash")
    table = Table(
        table_index=0,
        title="Net sales by category",
        units="in millions",
        markdown="| Product | 2024 |\n| --- | --- |\n| iPhone | $201,183 |",
        rows=[["Product", "2024"], ["iPhone", "$201,183"]],
        source_html_hash="a" * 64,
        section="Item 7. Management's Discussion and Analysis",
    )

    await replace_tables(db, document_id, [table])

    row = (
        await db.execute(
            select(DocumentTable).where(DocumentTable.document_id == document_id)
        )
    ).scalar_one()
    assert row.markdown == table.markdown
    assert row.table_data["rows"] == [["Product", "2024"], ["iPhone", "$201,183"]]
    assert row.table_data["section"] == "Item 7. Management's Discussion and Analysis"
    assert row.source_html_hash == "a" * 64
    assert row.embed_text == table.embed_text

    assert [t.table_index for t in await pending_tables(db, document_id)] == [0]

    await replace_tables(db, document_id, [table])
    _, table_count = await counts(db, document_id)
    assert table_count == 1
