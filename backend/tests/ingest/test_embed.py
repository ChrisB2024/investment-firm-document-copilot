"""Tests for ingest.embed.

All of these run against the fake_embeddings fixture. None should reach the
network; a test here that needs a key is in the wrong file.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.database.constant import EMBEDDING_DIMENSIONS
from ingest.chunk import Chunk
from ingest.embed import (
    MAX_BATCH_TOKENS,
    MAX_INPUT_TOKENS,
    embed_chunks,
    embed_tables,
)
from ingest.extract import Table


def _chunks(count: int, tokens: int = 10) -> list[Chunk]:
    return [
        Chunk(chunk_index=i, text=f"Chunk {i} body text.", token_count=tokens)
        for i in range(count)
    ]


async def test_batches_respect_the_input_cap(fake_embeddings):
    """250 chunks go out as [100, 100, 50]."""
    calls = fake_embeddings()
    chunks = _chunks(250)

    await embed_chunks(chunks)

    assert calls == [100, 100, 50]
    assert all(c.embedding is not None for c in chunks)


async def test_batches_respect_the_token_cap(fake_embeddings):
    """The token cap binds before BATCH_SIZE when chunks are large.

    Same 250 chunks as the test above, batched into 24s instead of hundreds
    purely because of their token counts. Either cap can bind first, and only
    the input cap is visible from a count of items.

    The per-chunk size is MAX_INPUT_TOKENS, the largest an item may legally be.
    Anything above it is rejected pre-flight, so this is the widest gap the two
    caps can have — and it is still a factor of four.
    """
    calls = fake_embeddings()

    await embed_chunks(_chunks(250, tokens=MAX_INPUT_TOKENS))

    assert calls == [24] * 10 + [10]
    assert max(calls) * MAX_INPUT_TOKENS <= MAX_BATCH_TOKENS


async def test_out_of_order_response_is_remapped(fake_embeddings):
    """Each chunk receives its own vector even when the response is reordered.

    This is the single most important test in the file. Without the sort by
    item.index, every embedding lands on the wrong passage — and nothing
    fails. Retrieval keeps working and returns confident nonsense.

    The stub fills each vector with its position in the batch and hands them
    back reversed, so an unsorted response would give chunk 0 the vector 4.0.
    One batch, because positions restart at 0 in the next one.
    """
    fake_embeddings(shuffle=True)
    chunks = _chunks(5)

    await embed_chunks(chunks)

    for position, chunk in enumerate(chunks):
        assert chunk.embedding == [float(position)] * EMBEDDING_DIMENSIONS


async def test_dimension_mismatch_raises(fake_embeddings):
    """A short vector raises, naming the item, the width and the model.

    The vector(1536) column would reject it anyway, but the error there says
    nothing about which setting drifted.
    """
    fake_embeddings(dimensions=512)
    chunks = [Chunk(chunk_index=7, text="Chunk 7 body text.", token_count=10)]

    with pytest.raises(RuntimeError) as excinfo:
        await embed_chunks(chunks)

    message = str(excinfo.value)
    assert "chunk 7" in message
    assert str(EMBEDDING_DIMENSIONS) in message
    assert "512" in message
    assert settings.openai_embedding_model in message
    assert chunks[0].embedding is None


async def test_short_response_raises(fake_embeddings):
    """A response with fewer vectors than texts raises with both counts.

    zip(strict=True) would catch the pairing, but the explicit count check
    reports how many were sent versus returned.
    """
    fake_embeddings(drop=1)
    chunks = _chunks(5)

    with pytest.raises(RuntimeError) as excinfo:
        await embed_chunks(chunks)

    message = str(excinfo.value)
    assert "sent 5" in message
    assert "got 4" in message
    assert all(c.embedding is None for c in chunks)


async def test_blank_text_is_rejected_before_any_call(fake_embeddings):
    """One blank chunk raises before anything is sent.

    The endpoint rejects empty input and fails the whole batch, so a single
    blank chunk would lose the ~100 good ones travelling with it. The empty
    calls list is the assertion that matters: it proves no request was issued,
    not merely that none succeeded.
    """
    calls = fake_embeddings()
    chunks = _chunks(3)
    chunks[1].text = "   "

    with pytest.raises(ValueError, match=r"\[1\]"):
        await embed_chunks(chunks)

    assert calls == []
    assert all(c.embedding is None for c in chunks)


async def test_oversized_input_is_rejected_before_any_call(fake_embeddings):
    """A chunk over the per-input cap raises before anything is sent.

    The cap is per string, not per request: the endpoint rejects the whole
    request over one bad input, so an oversized chunk costs the ~100 good ones
    batched with it and returns an error that does not name the culprit.

    This is the far end of chunk.py's unsplittable-paragraph xfail. A paragraph
    with no sentence boundary is emitted whole, and once one exceeds this cap
    the failure moves from chunking, where it is visible, to the OpenAI call,
    where it is not.
    """
    calls = fake_embeddings()
    chunks = _chunks(3)
    chunks[1].token_count = MAX_INPUT_TOKENS + 1

    with pytest.raises(ValueError) as excinfo:
        await embed_chunks(chunks)

    message = str(excinfo.value)
    assert f"chunk 1 at {MAX_INPUT_TOKENS + 1}" in message
    assert calls == []
    assert all(c.embedding is None for c in chunks)

    # A budget set wrong puts every item over the cap; the message lists a few
    # and counts the rest rather than dumping thousands of them.
    many = _chunks(250, tokens=MAX_INPUT_TOKENS + 1)
    with pytest.raises(ValueError) as excinfo:
        await embed_chunks(many)

    message = str(excinfo.value)
    assert message.startswith("250 chunk(s) over")
    assert "(+245 more)" in message
    assert len(message) < 500


async def test_tables_embed_through_the_same_path(fake_embeddings):
    """embed_tables sends embed_text, not markdown, and fills in .embedding.

    The caption carries the semantics a user types ("revenue by product"); the
    rows carry the figures. embed_text must mirror the search_vector
    expression on document_tables so lexical and semantic retrieval agree.

    Asserting on the text actually sent is the point — populating .embedding
    would look identical if markdown alone had gone out, and the caption is the
    half users search on.
    """
    calls = fake_embeddings()
    table = Table(
        table_index=0,
        title="Net sales by category",
        units="in millions",
        markdown="| Product | 2024 |\n| --- | --- |\n| iPhone | $201,183 |",
        section="Item 7. Management's Discussion and Analysis",
    )

    await embed_tables([table])

    assert calls == [1]
    assert table.embedding == [0.0] * EMBEDDING_DIMENSIONS

    sent = calls.texts[0][0]
    assert sent == table.embed_text
    assert sent != table.markdown
    for part in ("Item 7.", "Net sales by category", "in millions", "$201,183"):
        assert part in sent


async def test_empty_input_makes_no_call(fake_embeddings):
    """WORKED EXAMPLE — shows the fake_embeddings pattern; the rest are yours.

    `calls` records the size of every batch actually requested, so an empty
    list proves no request was issued rather than merely that none succeeded.
    """
    calls = fake_embeddings()

    await embed_chunks([])

    assert calls == []
