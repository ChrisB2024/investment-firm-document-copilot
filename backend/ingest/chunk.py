"""Sections -> retrieval-ready chunks.

Chunking is the decision that most determines whether retrieval works. A chunk
that splits a sentence, a table row, or a risk factor away from its heading is a
chunk the model cannot use to answer with a citation.

Measured from the corpus: Apple's 2024 Item 1A alone is 68,735 characters, so
big sections must split; Item 1B is 5 characters ("None."), so tiny sections
must survive without being merged into a neighbour and mislabelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ingest.extract import Section

# text-embedding-3-small accepts 8191 tokens, far more than we want per chunk.
# The limit that matters is retrieval quality: large chunks dilute the embedding
# and make citations vague, small ones lose the context that makes them true.
TARGET_TOKENS = 700
OVERLAP_TOKENS = 100
MIN_TOKENS = 50

ENCODING = "cl100k_base"


@dataclass
class Chunk:
    chunk_index: int
    text: str
    token_count: int
    page: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def token_count(text: str, encoding: str = ENCODING) -> int:
    """Exact token count, not a characters/4 estimate.

    TODO: implement with tiktoken (`import tiktoken`, `get_encoding(ENCODING)`).
    Cache the encoder at module level —
    constructing it per call dominates the runtime over 25 filings.
    """
    raise NotImplementedError


def chunk_section(section: Section, start_index: int) -> list[Chunk]:
    """Split one section into overlapping chunks.

    TODO: implement. The decisions that matter:

      - Split on paragraph boundaries first, sentences only if a paragraph
        alone exceeds TARGET_TOKENS. Never split mid-sentence: a half-sentence
        citation destroys the analyst's trust faster than a missing answer.
      - Never split a table row across chunks.
      - Prepend the section heading to every chunk. Retrieval matches on the
        chunk's own text, so a chunk reading "The Company relies on a single
        supplier..." with no company or section attached is unfindable and
        uncitable.
      - A section shorter than MIN_TOKENS is still its own chunk. Do not merge
        it into a neighbour — "None." belongs to Item 1B and saying otherwise
        is a wrong answer with a citation attached.
    """
    raise NotImplementedError


def chunk_filing(sections: list[Section], filing_metadata: dict[str, Any]) -> list[Chunk]:
    """Chunk a whole filing, assigning contiguous chunk_index values.

    TODO: implement. `chunk_index` must be contiguous and gap-free across the
    filing: the unique (document_id, chunk_index) constraint depends on it, and
    so does neighbour lookup, which reads index-1 and index+1 to give the model
    surrounding context.

    Every chunk carries filing_metadata into `chunk_metadata` — ticker, company,
    form, filing_date, fiscal_year, accession_number, section. Retrieval filters
    on that JSONB with the jsonb_path_ops GIN index, and the citation UI reads
    it to render a source without a second query.
    """
    raise NotImplementedError
