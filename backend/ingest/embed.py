"""Chunks -> embedding vectors.

25 filings produce thousands of chunks. One request per chunk is slow enough to
matter and will hit rate limits; the API takes batches.

The client and the single request live in `app.embeddings`; this module is the
batching policy over them. `embed_texts` and `MAX_INPUT_TOKENS` are re-exported
so existing callers keep their import path.
"""

from __future__ import annotations

from collections.abc import Iterator

from app.config import settings
from app.embeddings import MAX_INPUT_TOKENS, embed_texts
from ingest.chunk import token_count

# The API caps inputs per request and total tokens per request. Batch size is a
# trade-off, not a constant to guess at: too large and a single failure costs
# the whole batch, too small and the run crawls.
BATCH_SIZE = 100


MAX_BATCH_TOKENS = 200_000

def _batches(items: list, tokens_of) -> Iterator[list]:
    """Group items under both the per-request input cap and the token cap.

    Either can bind first: a hundred 700-token chunks sit well under the token
    cap, while a hundred wide tables would not.
    """
    batch: list = []
    tokens = 0
    for item in items:
        size = tokens_of(item)
        if batch and (len(batch) >= BATCH_SIZE or tokens + size > MAX_BATCH_TOKENS):
            yield batch
            batch, tokens = [], 0
        batch.append(item)
        tokens += size
    if batch:
        yield batch


async def embed_chunks(chunks: list) -> None:
    """Embed chunks in batches, in place."""
    await _embed_items(
        chunks, lambda c: c.text, lambda c: c.chunk_index, "chunk",
        lambda c: c.token_count,
    )


async def embed_tables(tables: list) -> None:
    """Embed tables in batches, in place.

    Tables are a second retrievable source type, not part of the chunk stream:
    they are embedded whole so a multi-year row keeps the header naming its
    years. `embed_text` mirrors the search_vector expression on document_tables
    so semantic and lexical retrieval see the same text.
    """
    await _embed_items(
        tables, lambda t: t.embed_text, lambda t: t.table_index, "table",
        # Counted here rather than stored: the ORM row has no token_count, and
        # a stored count would be one more thing to keep in step with the text.
        lambda t: token_count(t.embed_text),
    )


async def _embed_items(items: list, text_of, index_of, label: str, tokens_of) -> None:
    """Embed anything with a text and an index, in place.

    Both pre-flight checks fail before any request is issued, because the
    endpoint rejects a bad input by failing the whole request: one blank or
    oversized item would otherwise cost the ~100 good ones batched with it.

    Verifies the returned vector length equals
    `settings.openai_embedding_dimensions`: the column is vector(1536) and a
    mismatch fails at insert, but far more usefully it says which model and
    which setting drifted.
    """
    blank = [index_of(i) for i in items if not text_of(i).strip()]
    if blank:
        raise ValueError(
            f"{label}(s) {blank} have empty text; the embeddings endpoint rejects "
            "empty input and would fail the entire batch."
        )

    oversized = [(index_of(i), size) for i in items
                 if (size := tokens_of(i)) > MAX_INPUT_TOKENS]
    if oversized:
        # Listed rather than dumped: a mis-set chunk budget puts every item over
        # the cap, and an error naming thousands of them is an error no one reads.
        shown = ", ".join(f"{label} {index} at {size}" for index, size in oversized[:5])
        more = f" (+{len(oversized) - 5} more)" if len(oversized) > 5 else ""
        raise ValueError(
            f"{len(oversized)} {label}(s) over the {MAX_INPUT_TOKENS}-token input "
            f"cap: {shown}{more}. The endpoint rejects the whole request, so one "
            f"oversized {label} takes its entire batch of up to {BATCH_SIZE} with "
            f"it — and the error it returns does not name the culprit."
        )

    expected = settings.openai_embedding_dimensions
    model = settings.openai_embedding_model

    for batch in _batches(items, tokens_of):
        vectors = await embed_texts([text_of(i) for i in batch])

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding count mismatch: sent {len(batch)} texts, "
                f"got {len(vectors)} vectors from {model}"
            )

        for item, vector in zip(batch, vectors, strict=True):
            if len(vector) != expected:
                raise RuntimeError(
                    f"embedding dimension mismatch on {label} {index_of(item)}: "
                    f"expected {expected}, got {len(vector)} from {model}. "
                    "Model or openai_embedding_dimensions has drifted from the "
                    f"vector({expected}) column."
                )
            item.embedding = vector

