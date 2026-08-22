"""Tests for ingest.embed.

All of these run against the fake_embeddings fixture. None should reach the
network; a test here that needs a key is in the wrong file.
"""

from __future__ import annotations

from ingest.embed import embed_chunks


def test_batches_respect_the_input_cap(fake_embeddings):
    """TODO: embed 250 chunks and assert the batch sizes are [100, 100, 50]."""
    raise NotImplementedError


def test_batches_respect_the_token_cap(fake_embeddings):
    """TODO: embed chunks whose token counts force smaller batches than
    BATCH_SIZE, and assert the split honours MAX_BATCH_TOKENS."""
    raise NotImplementedError


def test_out_of_order_response_is_remapped(fake_embeddings):
    """TODO: install the stub with shuffle=True and assert each chunk still
    receives its own vector.

    This is the single most important test in the file. Without the sort by
    item.index, every embedding lands on the wrong passage — and nothing
    fails. Retrieval keeps working and returns confident nonsense.
    """
    raise NotImplementedError


def test_dimension_mismatch_raises(fake_embeddings):
    """TODO: install with dimensions=512 and assert RuntimeError naming the
    chunk and the model.

    The vector(1536) column would reject it anyway, but the error there says
    nothing about which setting drifted.
    """
    raise NotImplementedError


def test_short_response_raises(fake_embeddings):
    """TODO: install with drop=1 and assert RuntimeError.

    zip(strict=True) would catch the pairing, but the explicit count check
    reports how many were sent versus returned.
    """
    raise NotImplementedError


def test_blank_text_is_rejected_before_any_call(fake_embeddings):
    """TODO: assert ValueError and that no request was made — the returned
    calls list stays empty. The endpoint rejects empty input and fails the
    whole batch, so one blank chunk would lose ~100 good ones."""
    raise NotImplementedError


def test_tables_embed_through_the_same_path(fake_embeddings):
    """TODO: assert embed_tables populates Table.embedding and uses
    embed_text — section, caption and units included, not markdown alone.

    The caption carries the semantics a user types ("revenue by product"); the
    rows carry the figures. embed_text must mirror the search_vector
    expression on document_tables so lexical and semantic retrieval agree.
    """
    raise NotImplementedError


async def test_empty_input_makes_no_call(fake_embeddings):
    """WORKED EXAMPLE — shows the fake_embeddings pattern; the rest are yours.

    `calls` records the size of every batch actually requested, so an empty
    list proves no request was issued rather than merely that none succeeded.
    """
    calls = fake_embeddings()

    await embed_chunks([])

    assert calls == []
