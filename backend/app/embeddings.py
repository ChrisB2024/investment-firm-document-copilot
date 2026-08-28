"""The OpenAI embedding client and the single-batch call.

Split out of `ingest/embed.py` because retrieval needs it too, and `app/`
depending on `ingest/` would make the service import a one-off scripts package.
What lives here is what any caller needs: the client, one request, and the
endpoint's own limit. Batching policy for bulk ingestion stays in `ingest`.
"""

from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI

from app.config import settings

_client = AsyncOpenAI(
    api_key=settings.openai_api_key.get_secret_value(),
    max_retries=8,
)

# Per *input*, not per request: text-embedding-3-small accepts 8191 tokens for
# any one string. Exceeding it fails the whole request, not just that string.
MAX_INPUT_TOKENS = 8191


async def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch, preserving input order.

    Order is a correctness requirement, not a nicety: results are zipped back
    onto chunks positionally, so a reordered response silently attaches every
    embedding to the wrong passage. Retrieval would still "work" and return
    confident nonsense.

    Rate limits are retried with backoff by the client's `max_retries`. Anything
    else propagates — a partial corpus that looks complete is worse than a
    failed run.
    """
    response = await _client.embeddings.create(
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
        input=list(texts),
    )
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
