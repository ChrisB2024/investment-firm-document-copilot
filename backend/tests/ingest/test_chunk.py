"""Tests for ingest.chunk."""


from __future__ import annotations

import re

import pytest

from ingest.chunk import TARGET_TOKENS, chunk_section, token_count
from ingest.extract import Section


def test_heading_is_prepended_to_every_chunk():
    """Every chunk starts with its section heading.

    Retrieval matches on the chunk's own text. A passage reading "The Company
    relies on a single supplier..." names neither company nor section, so it is
    both unfindable and uncitable.
    """
    body = "\n\n".join(
        f"Paragraph {i}. The Company relies on a single supplier. " * 10
        for i in range(20)
    )
    section = Section(item="1A", title="Risk Factors", text=body)
    assert token_count(body) > TARGET_TOKENS * 2
    chunks = chunk_section(section, 0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith(section.heading)



def test_chunks_respect_the_token_budget():
    """Chunk token counts stay within TARGET_TOKENS for a body of prose.

    This pins current behaviour, not intended behaviour. The budget is enforced
    on summed atom counts while the chunk text is a span slice, so inter-atom
    separators go uncounted. Prose paragraphs are few enough and large enough
    that the gap stays under the limit here; test_never_splits_a_table_row feeds
    300 short rows, where the same gap puts every chunk over it.

    The lower bound matters as much as the upper one: without it a regression
    that emitted one atom per chunk would satisfy the upper bound trivially.
    """
    body = "\n\n".join(
        f"Paragraph {i}. The Company relies on a single supplier for this component. " * 10
        for i in range(20)
    )
    section = Section(item="1A", title="Risk Factors", text=body)
    chunks = chunk_section(section, 0)
    assert len(chunks) > 1
    assert max(c.token_count for c in chunks) <= TARGET_TOKENS
    assert min(c.token_count for c in chunks[:-1]) >= TARGET_TOKENS * 0.6


@pytest.mark.xfail(reason="j == i takes the first atom unconditionally; a paragraph with no sentence boundary is never split", strict=True)
def test_an_unsplittable_paragraph_still_blows_the_budget():
    blob = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_section(Section(item="1A", title="Risk Factors", text=blob), 0)
    assert chunks[0].token_count <= TARGET_TOKENS

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")

def test_never_splits_mid_sentence():
    """Every chunk body ends at a sentence boundary.

    A half-sentence citation costs trust faster than a missing answer.

    The body is a single paragraph with no blank line, which is what sends
    _atomize down _sentence_spans instead of _paragraph_spans. Reformatted into
    paragraphs this test still passes while no longer touching sentence
    splitting at all.

    The offset checks are not redundant with the regex: a cut after the "." in
    "$3.5 billion" satisfies the terminator but leaves a non-space next, which
    is what _SENTENCE_BOUNDARY's trailing-whitespace requirement exists to
    prevent. Neither check catches an abbreviation cut — that is
    test_abbreviations_do_not_end_a_sentence's job.
    """
    body = " ".join(
        f"Sentence {i} describes a distinct risk to the business of the Company."
        for i in range(200)
    )
    section = Section(item="1A", title="Risk Factors", text=body)
    chunks = chunk_section(section, 0)

    assert len(chunks) > 1
    for chunk in chunks:
        chunk_body = chunk.text.removeprefix(f"{section.heading}\n\n")
        assert _SENTENCE_END.search(chunk_body), f"cut mid-sentence: ... {chunk_body[-60:]!r}"

        start = body.find(chunk_body)
        assert start != -1, "chunk text is not a verbatim slice of the body"
        end = start + len(chunk_body)
        assert start == 0 or body[start - 1].isspace()
        assert end == len(body) or body[end].isspace()



def test_never_splits_a_table_row():
    """No chunk boundary falls inside a table row.

    Membership catches a cut inside a row. The slice check catches rows dropped
    or reordered, which membership alone would pass.
    """
    body = "\n".join(
        f"Product line {i:<8}{i * 1101:>12,}{i * 1303:>12,}{i * 1709:>12,}"
        for i in range(300)
    )
    section = Section(item="7", title="Management's Discussion and Analysis", text=body)
    chunks = chunk_section(section, 0)

    # Rows have no sentence terminator, so >1 chunk means _looks_like_table won.
    assert len(chunks) > 1

    rows = set(body.splitlines())
    for chunk in chunks:
        chunk_body = chunk.text.removeprefix(f"{section.heading}\n\n")
        for line in chunk_body.splitlines():
            assert line in rows, f"boundary split a row: {line!r}"

        start = body.find(chunk_body)
        assert start != -1, "chunk text is not a verbatim slice of the body"
        end = start + len(chunk_body)
        assert start == 0 or body[start - 1] == "\n"
        assert end == len(body) or body[end] == "\n"



def test_abbreviations_do_not_end_a_sentence():
    """TODO: assert "Inc." and "U.S." do not create a split.

    _ABBREVIATION_TAIL exists for this; without it a chunk can end at
    "Apple Inc." mid-clause.
    """
    raise NotImplementedError


def test_tiny_section_becomes_its_own_chunk():
    """TODO: assert a 5-character section ("None.") yields exactly one chunk
    and is not merged into a neighbour.

    Merging would attribute Item 1B's answer to another Item — a wrong answer
    carrying a real citation, which is the failure mode the whole product is
    built to avoid.
    """
    raise NotImplementedError


def test_chunk_index_is_contiguous_across_the_filing():
    """TODO: assert indices are exactly range(len(chunks)).

    UNIQUE (document_id, chunk_index) depends on it, and so does neighbour
    lookup, which reads index-1 and index+1 for surrounding context.
    """
    raise NotImplementedError


def test_metadata_is_carried_onto_every_chunk(manifest_entry):
    """TODO: assert each chunk's metadata has ticker, company, form,
    filing_date, fiscal_year, accession_number and section — and that it is
    JSON-serialisable.

    It lands in a JSONB column, so a non-serialisable value (a function, a date
    object) fails at write time, long after chunking looked fine.
    """
    raise NotImplementedError


def test_missing_required_metadata_raises(manifest_entry):
    """TODO: drop a required key and assert ValueError naming it.

    Failing at ingest beats producing chunks that retrieval cannot filter and
    the citation UI cannot render.
    """
    raise NotImplementedError


def test_overlap_carries_context_between_chunks():
    """TODO: assert consecutive chunks share trailing/leading content, and that
    the walk-back always makes progress (no infinite loop on pathological
    atom sizes)."""
    raise NotImplementedError


def test_token_count_matches_the_encoder():
    """TODO: assert token_count agrees with tiktoken for a known string, and
    that the cached encoder returns the same object across calls."""
    raise NotImplementedError
