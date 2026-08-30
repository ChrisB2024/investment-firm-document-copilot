"""Fixtures for the retrieval tests.

Most of this package is pure: `Filters` normalisation, `fuse`, `_interleave`,
`_arm_limit`, and everything in the CLI that formats rather than queries. Those
belong in the fast suite and need nothing here.

What does need a fixture is the other half, which cannot be mocked into
meaning anything. A stubbed `vector_search` proves the caller passes its
arguments along; it says nothing about whether `<=>` matches the HNSW opclass,
whether a filter binds or silently returns the whole corpus, or whether
`row_number() OVER (PARTITION BY ...)` partitions the way the grid needs. Those
are the failures worth catching, and only Postgres can answer them. The
session they run against is the `corpus` fixture, which sits in
tests/conftest.py because tests/assistant/ needs it too.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.retrieval.queries import Hit, SourceType

# Document ids live in their own range, so `hit(3, ...)` has row_id
# UUID(int=3) and document_id UUID(int=2**64 + 3) rather than the same value
# twice. A fused result that carried the row id through as its document id
# would otherwise look correct.
DOCUMENT_OFFSET = 2**64


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
