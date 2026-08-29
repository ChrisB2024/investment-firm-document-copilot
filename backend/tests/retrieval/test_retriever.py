"""Tests for app.retrieval.retriever.

Split by what each part needs. `_arm_limit` and `_interleave` are pure and
belong in the fast suite; the three `retrieve*` entry points and `neighbours`
need the ingested corpus and are marked integration.
"""

from __future__ import annotations

import pytest


def test_arm_limit_floors_below_the_multiplier():
    """TODO: assert `_arm_limit(3)` is MIN_ARM_LIMIT, not 6.

    The floor matters more than the multiplier. At limit=3 a bare 2x asks each
    arm for six, and a passage the other arm ranked 15th can never be rewarded
    for agreement it was never asked about — which is the whole reason both
    arms run.
    """
    raise NotImplementedError


def test_interleave_round_robins_across_groups(hit):
    """TODO: three groups of three; assert the output is every group's best,
    then every group's second, then every group's third."""
    raise NotImplementedError


def test_interleave_balance_survives_truncation(hit):
    """TODO: cut the merged list at several lengths and assert no group is ever
    more than one passage ahead of another.

    This is the property the whole fan-out rests on, and it has already been
    wrong once. A flat round-robin over (ticker, fiscal_year) cells looked
    correct and gave `{AAPL: 4, NVDA: 5, MSFT: 1}` at a cut of ten — NVIDIA
    taking all five of its years before Alphabet or Amazon got one, which is
    exactly the shape the grid exists to prevent.

    Cut at several lengths, not one. The bug passed at a cut of 25 (every group
    complete) and failed at 5 and 10, so a single generous cut proves nothing.

    Assert the *spread*, not an exact arrangement: which group leads a round is
    a score comparison across populations and may legitimately change.
    """
    raise NotImplementedError


def test_interleave_is_deterministic(hit):
    """TODO: assert the same input in a differently-ordered mapping produces
    identical output.

    Determinism is what makes two runs of one question diffable. The round sort
    is `(-score, key)`, so it holds only because the key is orderable — which
    is why `Key` is constrained rather than left open.
    """
    raise NotImplementedError


def test_interleave_handles_ragged_and_empty_groups(hit):
    """TODO: groups of unequal length, an empty group, and an empty mapping.

    A company that filed in three of five years is ragged by construction, not
    an edge case — `grid_search` omits cells with no rows.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_retrieve_returns_ranked_passages(corpus):
    """TODO: assert ranks are contiguous from 1, scores are populated, and
    every result carries at least one arm in `contributions`."""
    raise NotImplementedError


@pytest.mark.integration
async def test_retrieve_survives_a_question_with_no_lexical_signal(corpus):
    """TODO: retrieve for "what about it" and assert results come back with
    `contributions` naming the vector arm only.

    An all-stopword question yields a NULL tsquery and an empty text arm. That
    is not an error — fusion over one arm is still a valid ranking — and the
    honest answer for a question with no lexical signal.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_retrieve_per_ticker_covers_every_company(corpus):
    """TODO: assert every requested ticker appears, and that a ticker absent
    from the corpus is skipped rather than raising."""
    raise NotImplementedError


@pytest.mark.integration
async def test_retrieve_per_ticker_rejects_tickers_in_filters(corpus):
    """TODO: assert ValueError when `filters.tickers` is set.

    It is overwritten per arm, so passing it there would silently discard the
    caller's restriction — the failure returns plausible results for the wrong
    companies.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_retrieve_grid_covers_every_cell(corpus):
    """TODO: five companies across five years; assert 25 distinct
    (ticker, fiscal_year) cells and five years per company.

    This is the Phase 3 exit criterion in a test. `retrieve(limit=10)` returns
    two companies for the same question, and `retrieve(limit=100)` still misses
    Amazon at 64,889 tokens — no value of `limit` reaches it.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_retrieve_grid_leads_with_one_passage_per_company(corpus):
    """TODO: assert the first N results of an N-company grid name N distinct
    companies.

    The integration half of the truncation property: the pure test pins
    `_interleave`, this pins that `retrieve_grid` groups by company before
    handing it over.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_neighbours_never_cross_a_section(corpus):
    """TODO: widen a sample of chunks and assert no window contains text from a
    different section.

    23.3% of adjacent chunk pairs straddle a section boundary, so a
    document-only constraint attaches another Item's prose to this passage's
    citation one time in four — and the citation still reads as the original
    Item. Sample enough anchors that the 23.3% is near-certain to be hit.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_neighbours_carry_the_heading_once(corpus):
    """TODO: assert a widened window contains its section heading exactly once.

    Every chunk is prefixed with its heading by `ingest.chunk`, so joining the
    rows raw repeats it per neighbour — 2.5 times per window, measured.
    """
    raise NotImplementedError


@pytest.mark.integration
async def test_tables_are_never_widened(corpus):
    """TODO: assert a table passage gets no context.

    Tables are whole by construction, and a table's `table_index` neighbour is
    a different table, not more of this one.
    """
    raise NotImplementedError
