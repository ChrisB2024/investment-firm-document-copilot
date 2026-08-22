"""Tests for ingest.chunk."""

from __future__ import annotations


def test_heading_is_prepended_to_every_chunk():
    """TODO: assert every chunk starts with its section heading.

    Retrieval matches on the chunk's own text. A passage reading "The Company
    relies on a single supplier..." names neither company nor section, so it is
    both unfindable and uncitable.
    """
    raise NotImplementedError


def test_chunks_respect_the_token_budget():
    """TODO: chunk a long section and assert token counts stay near
    TARGET_TOKENS.

    Note the known gap before writing the assertion: the budget is enforced on
    summed atom counts while the chunk text is a span slice, so inter-atom
    separators go uncounted and table-heavy sections overshoot by ~25%. Decide
    whether this test pins current behaviour or the intended behaviour, and say
    which in the assertion message.
    """
    raise NotImplementedError


def test_never_splits_mid_sentence():
    """TODO: assert every chunk body ends at a sentence or block boundary.

    A half-sentence citation costs trust faster than a missing answer.
    """
    raise NotImplementedError


def test_never_splits_a_table_row():
    """TODO: feed a section whose body holds a multi-line table and assert no
    chunk boundary falls inside a row."""
    raise NotImplementedError


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
