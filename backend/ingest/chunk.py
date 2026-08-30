"""Sections -> retrieval-ready chunks.

Chunking is the decision that most determines whether retrieval works. A chunk
that splits a sentence, a table row, or a risk factor away from its heading is a
chunk the model cannot use to answer with a citation.

Measured from the corpus: Apple's 2024 Item 1A alone is 68,735 characters, so
big sections must split; Item 1B is 5 characters ("None."), so tiny sections
must survive without being merged into a neighbour and mislabelled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import tiktoken

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
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Populated by ingest.embed. Nullable so a run that dies mid-embedding
    # leaves chunks a re-run can finish, matching the nullable DB column.
    embedding: list[float] | None = None


@cache
def _get_encoder(encoding: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding)

def token_count(text: str, encoding: str = ENCODING) -> int:
    """Exact token count, not a characters/4 estimate.

    The encoder is cached: constructing one per call dominates the runtime over
    25 filings.
    """
    return len(_get_encoder(encoding).encode_ordinary(text))

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_TABLE_ROW = re.compile(r"\|| {2,}\S|\t")
_ABBREVIATION_TAIL = re.compile(
    r"(?:^|[\s(\[])(?:"
    r"[A-Z]|(?:[A-Z]\.)+[A-Z]|"
    r"No|Nos|Inc|Corp|Co|Ltd|LLC|LP|Mr|Mrs|Ms|Dr|Jr|Sr|St|vs|etc|e\.g|i\.e"
    r"|Fig|Sec|Art|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec"
    r")\.$"
)

def _trim(text: str, a: int, b: int) -> tuple[int, int] | None:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return (a, b) if b > a else None


def _paragraph_spans(body: str) -> list[tuple[int, int]]:
    raw, pos = [], 0
    for m in _PARAGRAPH_BREAK.finditer(body):
        raw.append((pos, m.start()))
        pos = m.end()
    raw.append((pos, len(body)))
    return [s for s in (_trim(body, a, b) for a, b in raw) if s]

def _looks_like_table(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for ln in lines if _TABLE_ROW.search(ln)) >= max(2, len(lines) // 2)

def _line_spans(body: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = span
    out, pos = [], a
    for ln in body[a:b].splitlines(keepends=True):
        s = _trim(body, pos, pos + len(ln))
        if s:
            out.append(s)
        pos += len(ln)
    return out

def _sentence_spans(body: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = span
    para = body[a:b]
    out, start = [], 0
    for m in _SENTENCE_BOUNDARY.finditer(para):
        if _ABBREVIATION_TAIL.search(para[start:m.start()]):
            continue
        s = _trim(para, start, m.start())
        if s:
            out.append((a + s[0], a + s[1]))
        start = m.end()
    s = _trim(para, start, len(para))
    if s:
        out.append((a + s[0], a + s[1]))
    return out

def _atomize(body: str, limit: int) -> list[tuple[int, int]]:
    """Indivisible units: paragraphs, or their lines/sentences if oversized."""
    atoms: list[tuple[int, int]] = []
    for a, b in _paragraph_spans(body):
        block = body[a:b]
        if token_count(block) <= limit:
            atoms.append((a, b))
        elif _looks_like_table(block):
            atoms.extend(_line_spans(body, (a, b)))
        else:
            atoms.extend(_sentence_spans(body, (a, b)))
    return atoms
        

def chunk_section(section: Section, start_index: int) -> list[Chunk]:
    """Split one section into overlapping chunks.

    Atoms are paragraphs, falling back to lines for tables and sentences for
    oversized prose, so no chunk ever splits a table row or a sentence.

    Every chunk is prefixed with the section heading. Retrieval matches on the
    chunk's own text, so a passage reading "The Company relies on a single
    supplier..." with no company or section attached is unfindable and
    uncitable.

    A section shorter than MIN_TOKENS still becomes its own chunk rather than
    merging into a neighbour: "None." belongs to Item 1B, and saying otherwise
    is a wrong answer with a citation attached.

    Note: the budget is enforced on the sum of atom token counts, but the chunk
    text is a span slice, so inter-atom separators are not counted. Table-heavy
    sections overshoot TARGET_TOKENS by roughly 25%.
    """
    heading = (section.heading or "").strip()
    body = section.text or ""
    prefix = f"{heading}\n\n" if heading else ""
    budget = max(TARGET_TOKENS - token_count(prefix) if prefix else TARGET_TOKENS, MIN_TOKENS)

    atoms = _atomize(body, budget)
    if not atoms:
        # Keywords, not positions: this is a seven-field dataclass, and a
        # positional call here silently shifted `heading` into `metadata` when
        # the unused `page` field was removed rather than failing.
        return [
            Chunk(
                chunk_index=start_index,
                text=heading,
                token_count=token_count(heading),
                section=heading or None,
            )
        ]

    counts = [token_count(body[a:b]) for a, b in atoms]
    chunks: list[Chunk] = []
    i = 0
    while i < len(atoms):
        total, j = 0, i
        while j < len(atoms) and (j == i or total + counts[j] <= budget):
            total += counts[j]
            j += 1
        text = prefix + body[atoms[i][0]:atoms[j - 1][1]]
        chunks.append(Chunk(
            chunk_index=start_index + len(chunks),
            text=text,
            token_count=token_count(text),
            section=heading or None,
        ))

        if j >= len(atoms):
            break
        k, overlap = j, 0
        while k - 1 > i and overlap + counts[k - 1] <=OVERLAP_TOKENS:
            overlap += counts[k - 1]
            k -= 1
        i = k
    return chunks
            
        

_REQUIRED_METADATA = (
    "ticker", "company", "form", "filing_date", "fiscal_year", "accession_number",
)
def chunk_filing(sections: list[Section], filing_metadata: dict[str, Any]) -> list[Chunk]:
    """Chunk a whole filing, assigning contiguous chunk_index values.

    `chunk_index` is contiguous and gap-free across the filing: the unique
    (document_id, chunk_index) constraint depends on it, and so does neighbour
    lookup, which reads index-1 and index+1 to give the model surrounding
    context.

    Every chunk carries filing_metadata into `chunk_metadata`. Retrieval filters
    on that JSONB with the jsonb_path_ops GIN index, and the citation UI reads it
    to render a source without a second query.
    """
    missing = [key for key in _REQUIRED_METADATA if key not in filing_metadata]
    if missing:
        raise ValueError(
            f"filing_metadata missing required key(s): {', '.join(missing)}. "
            "Every chunk carries these into chunk_metadata; a filing ingested "
            "without them is unfilterable and uncitable."
        )

    chunks: list[Chunk] = []
    for section in sections:
        produced = chunk_section(section, len(chunks))
        for chunk in produced:
            chunk.metadata = {**filing_metadata, "section": chunk.section}
        chunks.extend(produced)
    return chunks
