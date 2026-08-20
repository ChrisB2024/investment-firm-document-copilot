"""Chunks -> embedding vectors.

25 filings produce thousands of chunks. One request per chunk is slow enough to
matter and will hit rate limits; the API takes batches.
"""

from __future__ import annotations

from collections.abc import Sequence

# The API caps inputs per request and total tokens per request. Batch size is a
# trade-off, not a constant to guess at: too large and a single failure costs
# the whole batch, too small and the run crawls.
BATCH_SIZE = 100


async def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch, preserving input order.

    TODO: implement against the OpenAI SDK using
    `settings.openai_embedding_model` and `settings.openai_embedding_dimensions`.

    Order is a correctness requirement, not a nicety: results are zipped back
    onto chunks positionally, so a reordered response silently attaches every
    embedding to the wrong passage. Retrieval would still "work" and return
    confident nonsense.

    Retry on rate limits with backoff. Let anything else propagate — a partial
    corpus that looks complete is worse than a failed run.
    """
    raise NotImplementedError


async def embed_chunks(chunks: list) -> None:
    """Embed chunks in batches, in place.

    TODO: implement. Verify the returned vector length equals
    `settings.openai_embedding_dimensions` before writing: the column is
    vector(1536) and a mismatch fails at insert, but far more usefully it tells
    you the model or dimension setting drifted.
    """
    raise NotImplementedError
