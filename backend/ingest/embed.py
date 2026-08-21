"""Chunks -> embedding vectors.

25 filings produce thousands of chunks. One request per chunk is slow enough to
matter and will hit rate limits; the API takes batches.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from openai import AsyncOpenAI

from app.config import settings

_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value(),
    max_retries=8
)

# The API caps inputs per request and total tokens per request. Batch size is a
# trade-off, not a constant to guess at: too large and a single failure costs
# the whole batch, too small and the run crawls.
BATCH_SIZE = 100


async def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch, preserving input order.

    Uses the OpenAI SDK with
    `settings.openai_embedding_model` and `settings.openai_embedding_dimensions`.

    Order is a correctness requirement, not a nicety: results are zipped back
    onto chunks positionally, so a reordered response silently attaches every
    embedding to the wrong passage. Retrieval would still "work" and return
    confident nonsense.

    Retry on rate limits with backoff. Let anything else propagate — a partial
    corpus that looks complete is worse than a failed run.
    """
    if not texts:
        return []

    response = await _client.embeddings.create(
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
        input=list(texts)
    )
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


MAX_BATCH_TOKENS = 200_000

def _batches(chunks: list) -> Iterator[list]:
    """Group chunks under both the per-request input cap and the token cap."""
    batch: list = []
    tokens = 0
    for chunk in chunks:
        if batch and (len(batch) >= BATCH_SIZE
                      or tokens + chunk.token_count > MAX_BATCH_TOKENS):
            yield batch
            batch, tokens = [], 0
        batch.append(chunk)
        tokens += chunk.token_count
    if batch:
        yield batch


async def embed_chunks(chunks: list) -> None:
    """Embed chunks in batches, in place.

    Verify the returned vector length equals
    `settings.openai_embedding_dimensions` before writing: the column is
    vector(1536) and a mismatch fails at insert, but far more usefully it tells
    you the model or dimension setting drifted.
    """
    blank = [c.chunk_index for c in chunks if not c.text.strip()]
    if blank:
        raise ValueError(
            f"chunk(s) {blank} have empty text; the embeddings endpoint rejects "
            "empty input and would fail the entire batch."
        )

    expected = settings.openai_embedding_dimensions
    model = settings.openai_embedding_model

    for batch in _batches(chunks):
        vectors = await embed_texts([c.text for c in batch])

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding count mismatch: sent {len(batch)} texts, "
                f"got {len(vectors)} vectors from {model}"
            )

        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != expected:
                raise RuntimeError(
                    f"embedding dimension mismatch on chunk {chunk.chunk_index}: "
                    f"expected {expected}, got {len(vector)} from {model}. "
                    "Model or openai_embedding_dimensions has drifted from the "
                    f"vector({expected}) column."
                )
            chunk.embedding = vector

