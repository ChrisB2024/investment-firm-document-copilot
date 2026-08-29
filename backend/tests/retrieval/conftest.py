"""Fixtures for the retrieval tests.

Most of this package is pure: `Filters` normalisation, `fuse`, `_interleave`,
`_arm_limit`, and everything in the CLI that formats rather than queries. Those
belong in the fast suite and need nothing here.

What does need a fixture is the other half, which cannot be mocked into
meaning anything. A stubbed `vector_search` proves the caller passes its
arguments along; it says nothing about whether `<=>` matches the HNSW opclass,
whether a filter binds or silently returns the whole corpus, or whether
`row_number() OVER (PARTITION BY ...)` partitions the way the grid needs. Those
are the failures worth catching, and only Postgres can answer them.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.database.models import SourceDocument
from app.database.session import dispose, engine, session
from app.retrieval.queries import Hit, SourceType

# Enough of the corpus that a partition has something to partition. Retrieval
# tests assert on shape — every company covered, every year present — which is
# meaningless against two filings.
MIN_DOCUMENTS = 25

# Document ids live in their own range, so `hit(3, ...)` has row_id
# UUID(int=3) and document_id UUID(int=2**64 + 3) rather than the same value
# twice. A fused result that carried the row id through as its document id
# would otherwise look correct.
DOCUMENT_OFFSET = 2**64


@pytest.fixture
async def corpus():
    """A read-only session over the ingested corpus, or skip.

    The schema existing is not the same as the corpus existing. A fresh clone
    that has run migrations but not ingestion has every table and no rows, and
    a retrieval suite failing there says nothing about the code — so count the
    documents first and skip with the command that fixes it.

    Nothing here writes, so unlike `db` in test_persist.py there is nothing to
    roll back. The engine is still disposed and its cache cleared: `engine()`
    is process-wide and cached while pytest-asyncio gives each test its own
    event loop, so a pooled connection opened in one test fails in the next
    rather than where the mistake was.
    """
    try:
        async with session() as connection:
            documents = await connection.scalar(
                select(func.count()).select_from(SourceDocument)
            )
            if documents < MIN_DOCUMENTS:
                pytest.skip(
                    f"corpus has {documents} documents, needs at least "
                    f"{MIN_DOCUMENTS}; run `uv run python -m ingest.run`"
                )
            yield connection
    finally:
        await dispose()
        engine.cache_clear()


@pytest.fixture
def hit():
    """Build a `Hit` with a deterministic id, for the pure fusion tests.

    `hit(n, rank)` puts `UUID(int=n)` on the row, so a test asserts on identity
    by writing back the integer it passed in. Fusion's tie-break sorts on
    `str(row_id)`, and zero-padded hex orders the same way the integers do —
    so the small numbers a test writes also read as the order it should expect.

    Fusion and interleaving are pure functions over ranks. Driving them with
    real search results would make the test depend on the corpus, on the
    embedding model, and on the tie-break of two floats — none of which is what
    is being tested.
    """
    def _hit(n: int, rank: int, source_type: SourceType = "chunk") -> Hit:
        return Hit(
            source_type=source_type,
            row_id=UUID(int=n),
            document_id=UUID(int=DOCUMENT_OFFSET + n),
            rank=rank,
            # Shaped like the vector arm's cosine distance, where lower is
            # better. Nothing in `fuse` reads it — and an implementation that
            # started to would rank everything backwards here rather than
            # passing by coincidence off a score that agreed with the rank.
            score=float(rank),
        )
    return _hit
