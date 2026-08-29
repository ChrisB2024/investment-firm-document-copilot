"""Tests for app.retrieval.retriever.

Split by what each part needs. `_arm_limit` and `_interleave` are pure and
belong in the fast suite; the three `retrieve*` entry points and `neighbours`
need the ingested corpus and are marked integration.
"""

from __future__ import annotations

import pytest
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from app.config import settings
from app.retrieval.fusion import FusedHit, fuse
from app.retrieval.queries import Filters, hydrate
from app.retrieval.retriever import (
    ARM_MULTIPLIER,
    MIN_ARM_LIMIT,
    _arm_limit,
    _interleave,
    neighbours,
    retrieve,
    retrieve_grid,
    retrieve_per_ticker,
)

QUESTION = "supplier concentration risk"


def _group(hit, base: int, depth: int, *, start: int = 1) -> list[FusedHit]:
    """One group's ranked list, ids running `base`, `base + 1`, ...

    Built through `fuse` rather than by constructing `FusedHit` by hand: a
    round is ordered by the RRF score, and inventing those scores here would
    let a test pass on an ordering the real pipeline never produces.

    `start` is the arm rank the group's best hit had, and so sets how strong
    the whole group scores. Groups that all start at 1 score identically, which
    makes a global sort by score indistinguishable from a round-robin — the
    two orderings only diverge when one group outscores another.
    """
    return fuse(
        {"vector": [hit(base + i, start + i) for i in range(depth)]}, limit=depth
    )


def _spread(merged: list[FusedHit], cut: int, size: int) -> int:
    """How far apart the best- and worst-served groups are in `merged[:cut]`.

    Groups are recovered from the row ids, which `_group` lays out in blocks of
    `size` — so this counts what a caller truncating to fit a context window
    would actually get.
    """
    counts = [0] * size
    for fused in merged[:cut]:
        counts[fused.row_id.int // size - 1] += 1
    return max(counts) - min(counts)


def test_arm_limit_floors_below_the_multiplier():
    """Below the floor, each arm still searches `MIN_ARM_LIMIT` deep.

    The floor matters more than the multiplier. At limit=3 a bare 2x asks each
    arm for six, and a passage the other arm ranked 15th can never be rewarded
    for agreement it was never asked about — which is the whole reason both
    arms run.
    """
    assert _arm_limit(3) == MIN_ARM_LIMIT
    assert _arm_limit(3) != 3 * ARM_MULTIPLIER

    # Above the floor the multiplier takes over, or the floor would be a cap.
    assert _arm_limit(50) == 50 * ARM_MULTIPLIER

    # The crossover itself, where both rules agree — the point an off-by-one in
    # the comparison would move without changing either side of it.
    crossover = MIN_ARM_LIMIT // ARM_MULTIPLIER
    assert _arm_limit(crossover) == MIN_ARM_LIMIT
    assert _arm_limit(crossover + 1) == (crossover + 1) * ARM_MULTIPLIER


def test_interleave_round_robins_across_groups(hit):
    """Every group's best, then every group's second, then every group's third."""
    merged = _interleave(
        {
            "AAPL": _group(hit, 10, 3),
            "MSFT": _group(hit, 20, 3),
            "NVDA": _group(hit, 30, 3),
        }
    )

    # Within a round every group's hit holds the same RRF score, so the round
    # falls to the group key — alphabetical, and predictable from the ids.
    assert [f.row_id.int for f in merged] == [10, 20, 30, 11, 21, 31, 12, 22, 32]


def test_interleave_balance_survives_truncation(hit):
    """No group runs more than one passage ahead of another, at any cut.

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
    size = 5
    # Every group outscores the next, which is the shape that produced the bug:
    # NVIDIA's five years all ranked above Alphabet's first. With equal scores
    # a global sort happens to agree with a round-robin and nothing is tested.
    merged = _interleave({
        f"T{n}": _group(hit, size * n, size, start=1 + size * (n - 1))
        for n in range(1, size + 1)
    })
    assert len(merged) == size * size

    # Every cut, not a chosen few. The bug this guards passed at 25 and failed
    # at 5 and 10, so which cuts get checked is exactly what decided whether it
    # was caught.
    for cut in range(1, size * size + 1):
        assert _spread(merged, cut, size) <= 1, cut


def test_interleave_is_deterministic(hit):
    """The same groups in a differently-ordered mapping merge identically.

    Determinism is what makes two runs of one question diffable. The round sort
    is `(-score, key)`, so it holds only because the key is orderable — which
    is why `Key` is constrained rather than left open.
    """
    aapl, msft, nvda = _group(hit, 10, 3), _group(hit, 20, 3), _group(hit, 30, 3)

    assert _interleave({"AAPL": aapl, "MSFT": msft, "NVDA": nvda}) == _interleave(
        {"NVDA": nvda, "AAPL": aapl, "MSFT": msft}
    )


def test_interleave_handles_ragged_and_empty_groups(hit):
    """Unequal groups, an empty group, and no groups at all.

    A company that filed in three of five years is ragged by construction, not
    an edge case — `grid_search` omits cells with no rows.
    """
    ragged = _interleave(
        {
            "AAPL": _group(hit, 10, 3),
            "MSFT": _group(hit, 20, 1),
            "NVDA": _group(hit, 30, 2),
        }
    )
    # The short groups drop out of later rounds rather than padding them, so
    # every hit appears exactly once and the long group still finishes.
    assert [f.row_id.int for f in ragged] == [10, 20, 30, 11, 31, 12]

    # An empty group is not a round of its own, and not a hole in the others.
    sparse = _interleave({"AAPL": _group(hit, 10, 2), "MSFT": []})
    assert [f.row_id.int for f in sparse] == [10, 11]

    assert _interleave({}) == []


@pytest.mark.integration
async def test_retrieve_returns_ranked_passages(corpus):
    """Contiguous ranks from 1, real scores, and every result names an arm."""
    results = await retrieve(corpus, QUESTION, limit=5)
    assert len(results) == 5

    # Contiguous from 1 because `_materialise` renumbers: a row lost between
    # search and hydrate must close the gap, not leave a rank nobody can
    # account for.
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert all(r.score > 0 for r in results)

    # Every result was found by at least one arm — an empty `contributions`
    # would mean a passage fusion cannot explain, which is the field the CLI
    # exists to read.
    assert all(r.contributions for r in results)
    assert all(set(r.contributions) <= {"vector", "text"} for r in results)

    # Not widened unless asked: neighbour expansion costs about 3x the tokens.
    assert all(r.context is None for r in results)


@pytest.mark.integration
async def test_retrieve_survives_a_question_with_no_lexical_signal(corpus):
    """An all-stopword question ranks on the vector arm alone.

    An all-stopword question yields a NULL tsquery and an empty text arm. That
    is not an error — fusion over one arm is still a valid ranking — and the
    honest answer for a question with no lexical signal.
    """
    results = await retrieve(corpus, "what about it", limit=5)

    assert len(results) == 5
    assert all(set(r.contributions) == {"vector"} for r in results)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


@pytest.mark.integration
async def test_retrieve_per_ticker_covers_every_company(corpus):
    """Every requested company appears; one the corpus lacks is skipped."""
    asked = ("AAPL", "NVDA", "GOOGL")
    results = await retrieve_per_ticker(corpus, QUESTION, asked, per_ticker=2)

    assert {r.passage.ticker for r in results} == set(asked)

    # Lower case on the way in, because both sides normalise and a caller
    # typing "aapl" must not open a second, half-filled arm.
    assert {
        r.passage.ticker
        for r in await retrieve_per_ticker(corpus, QUESTION, ("aapl",), per_ticker=2)
    } == {"AAPL"}

    # The same company twice, spelled differently, is one arm. Normalising only
    # inside `Filters` is not enough: both arms would then scope to AAPL and
    # return the same rows, and `_interleave` would emit every passage twice —
    # a duplicate the caller cannot tell from two genuinely similar passages,
    # and one that reads to the model as corroboration.
    doubled = await retrieve_per_ticker(
        corpus, QUESTION, ("aapl", "AAPL"), per_ticker=2
    )
    assert len(doubled) == len({r.passage.row_id for r in doubled})
    assert len(doubled) == len(
        await retrieve_per_ticker(corpus, QUESTION, ("AAPL",), per_ticker=2)
    )

    # A company the corpus has never heard of contributes nothing rather than
    # raising: the agent picks the companies from the question, and one that
    # was not ingested is a gap to report, not a crash.
    mixed = await retrieve_per_ticker(corpus, QUESTION, ("AAPL", "ZZZZ"), per_ticker=2)
    assert {r.passage.ticker for r in mixed} == {"AAPL"}


@pytest.mark.integration
async def test_retrieve_per_ticker_rejects_tickers_in_filters(corpus):
    """Companies belong in the `tickers` argument, never in `filters`.

    It is overwritten per arm, so passing it there would silently discard the
    caller's restriction — the failure returns plausible results for the wrong
    companies.
    """
    with pytest.raises(ValueError) as excinfo:
        await retrieve_per_ticker(
            corpus, QUESTION, ("AAPL",), filters=Filters(tickers=("NVDA",))
        )

    # Naming the offending value, because the mistake is easy to make and the
    # results it would have produced look entirely reasonable.
    message = str(excinfo.value)
    assert "NVDA" in message
    assert "tickers" in message

    # Everything else in `filters` is still welcome — a year range applies to
    # every arm of the fan-out, and rejecting it would defeat the parameter.
    scoped = await retrieve_per_ticker(
        corpus,
        QUESTION,
        ("AAPL",),
        per_ticker=2,
        filters=Filters(fiscal_year_from=2024),
    )
    assert scoped
    assert all(r.passage.fiscal_year >= 2024 for r in scoped)


@pytest.mark.integration
async def test_retrieve_grid_covers_every_cell(corpus):
    """Five companies across five years come back as 25 distinct cells.

    This is the Phase 3 exit criterion in a test. `retrieve(limit=10)` returns
    two companies for the same question, and `retrieve(limit=100)` still misses
    Amazon at 64,889 tokens — no value of `limit` reaches it.
    """
    tickers = [t for (t,) in await corpus.execute(
        text("SELECT DISTINCT ticker FROM source_documents ORDER BY ticker")
    )]
    years = [y for (y,) in await corpus.execute(
        text("SELECT DISTINCT fiscal_year FROM source_documents ORDER BY fiscal_year")
    )]

    results = await retrieve_grid(
        corpus, QUESTION, tickers=tickers, years=years, per_cell=1
    )

    cells = {(r.passage.ticker, r.passage.fiscal_year) for r in results}
    assert cells == {(t, y) for t in tickers for y in years}

    # Per company, not just in total: 25 cells could also be five companies
    # with the wrong years, or one company counted 25 ways.
    for ticker in tickers:
        assert {y for (t, y) in cells if t == ticker} == set(years)


@pytest.mark.integration
async def test_retrieve_grid_leads_with_one_passage_per_company(corpus):
    """The first N results of an N-company grid name N distinct companies.

    The integration half of the truncation property: the pure test pins
    `_interleave`, this pins that `retrieve_grid` groups by company before
    handing it over.
    """
    tickers = [t for (t,) in await corpus.execute(
        text("SELECT DISTINCT ticker FROM source_documents ORDER BY ticker")
    )]
    years = [y for (y,) in await corpus.execute(
        text("SELECT DISTINCT fiscal_year FROM source_documents ORDER BY fiscal_year")
    )]

    results = await retrieve_grid(
        corpus, QUESTION, tickers=tickers, years=years, per_cell=1
    )

    # A caller truncating to fit a context window cuts here, and this is the
    # cut where a flat pass over cells gave one company five of the first ten.
    assert {r.passage.ticker for r in results[: len(tickers)]} == set(tickers)


@pytest.mark.integration
async def test_neighbours_never_cross_a_section(corpus):
    """No widened window carries prose from another section.

    23.3% of adjacent chunk pairs straddle a section boundary, so a
    document-only constraint attaches another Item's prose to this passage's
    citation one time in four — and the citation still reads as the original
    Item. Sample enough anchors that the 23.3% is near-certain to be hit.
    """
    anchor_ids = [
        row_id
        for (row_id,) in await corpus.execute(
            text("SELECT id FROM document_chunks ORDER BY id LIMIT 80")
        )
    ]
    passages = await hydrate(corpus, [("chunk", i) for i in anchor_ids])
    windows = await neighbours(corpus, list(passages.values()))

    # Exactly the rows a same-document window would have swept in and a
    # same-section one must not: adjacent by index, different section.
    strays = (
        await corpus.execute(
            text("""
                SELECT a.id AS anchor_id, n.section AS section, n.text AS text
                  FROM document_chunks a
                  JOIN document_chunks n
                    ON n.document_id = a.document_id
                   AND n.chunk_index BETWEEN a.chunk_index - :radius
                                         AND a.chunk_index + :radius
                 WHERE a.id = ANY(:anchor_ids)
                   AND n.section IS DISTINCT FROM a.section
            """).bindparams(
                bindparam("anchor_ids", type_=ARRAY(PgUUID(as_uuid=True)))
            ),
            {"anchor_ids": anchor_ids, "radius": settings.retrieval_neighbor_radius},
        )
    ).all()

    # The sample has to contain the failure for the assertion to mean anything.
    # At 23.3% of pairs, 80 anchors that produced none would be the surprise.
    assert strays, "sampled no section boundary; this proves nothing"

    checked = 0
    for stray in strays:
        window = windows.get(("chunk", stray.anchor_id))
        if window is None:
            continue

        if stray.section is not None:
            # The heading in the form `ingest.chunk` writes it. A stray joined
            # in raw keeps its own prefix — `neighbours` strips only the
            # anchor's — so this is the mark it leaves. The blank line matters:
            # a filing quotes "Item 1A. Risk Factors" in prose all the time.
            assert f"{stray.section}\n\n" not in window
            checked += 1

        body = stray.text.split("\n\n", 1)[-1].strip()
        # Long bodies only. "Not applicable." is an entire chunk in a 10-K and
        # appears under several Items, so a short body cannot identify the
        # chunk it came from and a substring test on one proves nothing.
        if len(body) >= 200:
            assert body[:200] not in window
            checked += 1

    assert checked, "no stray was distinctive enough to check"


@pytest.mark.integration
async def test_neighbours_carry_the_heading_once(corpus):
    """A widened window states its section heading once, at the top.

    Every chunk is prefixed with its heading by `ingest.chunk`, so joining the
    rows raw repeats it per neighbour — 2.5 times per window, measured.
    """
    rows = (
        await corpus.execute(
            text(
                "SELECT id, section, text FROM document_chunks"
                " WHERE section IS NOT NULL ORDER BY id LIMIT 40"
            )
        )
    ).all()
    passages = await hydrate(corpus, [("chunk", row.id) for row in rows])
    windows = await neighbours(corpus, list(passages.values()))
    assert windows

    widened = 0
    for row in rows:
        window = windows.get(("chunk", row.id))
        if window is None:
            continue
        assert window.startswith(row.section)
        # The prefix form specifically. A heading may legitimately be quoted in
        # prose ("see Item 1A. Risk Factors"); what must not recur is the
        # heading-then-blank-line that `ingest.chunk` writes.
        assert window.count(f"{row.section}\n\n") == 1
        if len(window) > len(row.text):
            widened += 1

    # Some window actually joined a neighbour, or "once" is just the anchor's
    # own heading and nothing was ever at risk of being repeated.
    assert widened


@pytest.mark.integration
async def test_tables_are_never_widened(corpus):
    """A table passage gets no context.

    Tables are whole by construction, and a table's `table_index` neighbour is
    a different table, not more of this one.
    """
    table_id = await corpus.scalar(text("SELECT id FROM document_tables LIMIT 1"))
    chunk_id = await corpus.scalar(
        text("SELECT id FROM document_chunks WHERE section IS NOT NULL LIMIT 1")
    )
    passages = await hydrate(corpus, [("table", table_id), ("chunk", chunk_id)])

    mixed = await neighbours(corpus, list(passages.values()))

    # The chunk is here to prove the call did something. Without it an empty
    # result would say only that `neighbours` returned nothing at all.
    assert ("chunk", chunk_id) in mixed
    assert ("table", table_id) not in mixed

    # Tables alone: no anchors, so no statement is issued and the caller's
    # `.get` leaves `context` None.
    assert await neighbours(corpus, [passages[("table", table_id)]]) == {}
