"""Tests for app.retrieval.queries.

`Filters` is pure and belongs in the fast suite. Everything else here is SQL,
and SQL is the thing a mock cannot check: a stub proves the caller passed its
arguments along, not that `<=>` matches the HNSW opclass, that a filter binds
rather than silently returning the whole corpus, or that a window partitions
the way the grid needs.
"""

from __future__ import annotations

import pytest


def test_filters_normalise_tickers():
    """TODO: assert "aapl" and " AAPL " both become "AAPL", and duplicates
    collapse while order is preserved.

    Both sides of the comparison normalise, so a caller passing lower case must
    match — and the grid keys cells on the ticker, so "aapl" and "AAPL" reaching
    the database as two values would produce two half-filled grids.
    """
    raise NotImplementedError


def test_filters_reject_an_inverted_year_range():
    """TODO: assert ValueError when `fiscal_year_from` is after
    `fiscal_year_to`.

    That range matches nothing, and an empty result is indistinguishable from a
    corpus gap. Failing at construction is the only place it can be told apart.
    """
    raise NotImplementedError


def test_matches_everything_is_about_predicates_not_the_corpus():
    """TODO: assert `NO_FILTERS.matches_everything` and that naming all five
    tickers is *not* `matches_everything`.

    It answers the narrow question the caller can act on — whether the join to
    `source_documents` can be skipped — not whether the corpus happens to be
    fully covered, which only the corpus knows.
    """
    raise NotImplementedError


def test_vector_literal_survives_scientific_notation():
    """TODO: assert `_vector_literal` renders a component like 7.8e-05 in a form
    pgvector parses.

    A real query embedding carries a handful of components small enough that
    `str(float(x))` switches to exponent form. pgvector accepts them — this
    pins that, so a future switch to fixed-point formatting is a deliberate
    change rather than a silent one.
    """
    raise NotImplementedError


pytestmark = pytest.mark.integration


async def test_filters_that_match_nothing_return_nothing(corpus):
    """TODO: search with `forms=("10-Q",)`, which the corpus has none of, and
    assert an empty list.

    This is the filter failure worth catching. A predicate that fails to bind
    does not error — it returns the unfiltered corpus, which looks like working
    retrieval and answers about the wrong companies.
    """
    raise NotImplementedError


async def test_every_filter_field_binds(corpus):
    """TODO: assert tickers, fiscal_year_from, fiscal_year_to and their
    combination each restrict the results.

    Check each field separately. A shared-parameter typo binds one of them and
    not the others, and any single-field test passes over it.
    """
    raise NotImplementedError


async def test_both_source_types_are_reachable(corpus):
    """TODO: assert a search over both types can return a table, and that
    `source_types=("chunk",)` returns none.

    46% of a filing's figures appear only inside a table, so a chunk-only
    result set cannot answer a numeric question however well it reads.
    """
    raise NotImplementedError


async def test_text_search_falls_back_when_every_term_fails(corpus):
    """TODO: assert a long question — one whose terms cannot all co-occur —
    still returns results, and that a short precise one does not lose precision
    to the fallback.

    `plainto_tsquery` ANDs every lexeme, so two real analyst questions match
    nothing at all. Falling back on *zero* rather than on "fewer than limit" is
    what keeps the precise case precise.
    """
    raise NotImplementedError


async def test_an_all_stopword_question_matches_nothing(corpus):
    """TODO: assert "what about it" returns no text hits under either mode.

    It yields a NULL tsquery, and that is the honest answer — there is no
    lexical signal to rank on. The caller treats it as a normal empty arm.
    """
    raise NotImplementedError


async def test_grid_search_partitions_per_cell(corpus):
    """TODO: assert every requested (ticker, fiscal_year) appears as its own
    key, and that each arm's ranks run 1..n within each cell.

    The partition is what makes this different from a filtered search: ranking
    happens inside the cell, so a company-year cannot be crowded out by a
    stronger one.
    """
    raise NotImplementedError


async def test_grid_search_falls_back_per_cell_not_per_statement(corpus):
    """TODO: use a question whose AND form reaches some cells but not all, and
    assert *every* cell still has a text arm.

    Measured across the 25 cells, `plainto_tsquery` reaches 10 for "supplier
    concentration risk" and 20 for "capital expenditures". A statement-wide
    fallback fires only when every cell is empty, so those queries would leave
    15 and 5 cells ranked by the vector arm alone — decided by whichever company
    happened to match. This is the assertion that catches that, and it passes
    trivially on a question where AND reaches everything or nothing.
    """
    raise NotImplementedError


async def test_grid_search_omits_cells_with_no_rows(corpus):
    """TODO: ask for a year the corpus does not have and assert the cell is
    absent rather than present and empty.

    The caller has to tell "nothing matched" from "nothing exists" — the second
    is a company that did not file, and reporting it as a miss invites a
    fabricated negative.
    """
    raise NotImplementedError


async def test_hydrate_returns_both_source_types(corpus):
    """TODO: hydrate a chunk key and a table key together; assert both come
    back with their citation fields, and that a key for a deleted row is simply
    absent."""
    raise NotImplementedError
