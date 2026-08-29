"""Tests for app.retrieval.queries.

`Filters` is pure and belongs in the fast suite. Everything else here is SQL,
and SQL is the thing a mock cannot check: a stub proves the caller passed its
arguments along, not that `<=>` matches the HNSW opclass, that a filter binds
rather than silently returning the whole corpus, or that a window partitions
the way the grid needs.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text

from app.database.constant import EMBEDDING_DIMENSIONS
from app.retrieval.queries import (
    NO_FILTERS,
    Filters,
    _vector_literal,
    grid_search,
    hydrate,
    text_search,
    vector_search,
)

# A question whose lexemes cannot all co-occur. Brief question 1, trimmed:
# `plainto_tsquery` ANDs every one of them and matches nothing in the corpus.
LONG_QUESTION = (
    "how did the revenue mix between iPhone, Services, Mac, iPad and Wearables change"
)

# AND-matches, and reaches 10 of the corpus's 25 cells — partial by measurement,
# which is what the per-cell fallback test needs.
PRECISE_QUESTION = "supplier concentration risk"


def _embedding() -> list[float]:
    """A deterministic query vector, not one from OpenAI.

    Every assertion below is about shape — that a filter binds, that a window
    partitions, that a cell is absent — and none about relevance. A live
    embedding would spend a network call per test to change nothing, and would
    move the results whenever the model does.

    The first component is deliberately small enough that `str(float(x))`
    renders it in exponent form, so every vector search here also exercises
    pgvector parsing what `_vector_literal` emits.
    """
    vector = [((i % 97) + 1) / 1000 for i in range(EMBEDDING_DIMENSIONS)]
    vector[0] = 7.8e-05
    return vector


def test_filters_normalise_tickers():
    """Case and surrounding space are stripped, duplicates collapse, order holds.

    Both sides of the comparison normalise, so a caller passing lower case must
    match — and the grid keys cells on the ticker, so "aapl" and "AAPL" reaching
    the database as two values would produce two half-filled grids.
    """
    filters = Filters(tickers=("aapl", " AAPL ", "nvda", "AAPL"))

    # Order preserved rather than sorted: it is the order the caller wrote, and
    # the fan-out reports companies in it.
    assert filters.tickers == ("AAPL", "NVDA")


def test_filters_reject_an_inverted_year_range():
    """A range whose start is after its end is refused at construction.

    That range matches nothing, and an empty result is indistinguishable from a
    corpus gap. Failing at construction is the only place it can be told apart.
    """
    with pytest.raises(ValueError) as excinfo:
        Filters(fiscal_year_from=2025, fiscal_year_to=2021)

    message = str(excinfo.value)
    assert "2025" in message
    assert "2021" in message

    # The legal shapes either side of the boundary. A single year is a closed
    # range with equal ends, and a half-open range is what "everything since
    # 2024" means — an off-by-one in the comparison rejects both.
    assert Filters(fiscal_year_from=2024, fiscal_year_to=2024).fiscal_year_to == 2024
    assert Filters(fiscal_year_from=2024).fiscal_year_to is None
    assert Filters(fiscal_year_to=2024).fiscal_year_from is None


def test_matches_everything_is_about_predicates_not_the_corpus():
    """It reports whether a predicate would be emitted, not whether all rows match.

    It answers the narrow question the caller can act on — whether the join to
    `source_documents` can be skipped — not whether the corpus happens to be
    fully covered, which only the corpus knows.
    """
    assert NO_FILTERS.matches_everything

    # Naming every issuer in the corpus selects every row, and is still not
    # `matches_everything`: the predicate exists, and only the corpus knows it
    # happens to be redundant.
    assert not Filters(
        tickers=("AAPL", "AMZN", "GOOGL", "MSFT", "NVDA")
    ).matches_everything

    # Each field on its own, because this is an `and` chain and a missing
    # clause makes it claim there is nothing to apply. The join is then skipped
    # and the filter silently disappears — a whole-corpus answer to a scoped
    # question, which is the failure mode with no visible symptom.
    assert not Filters(tickers=("AAPL",)).matches_everything
    assert not Filters(forms=("10-K",)).matches_everything
    assert not Filters(fiscal_year_from=2021).matches_everything
    assert not Filters(fiscal_year_to=2025).matches_everything


def test_vector_literal_survives_scientific_notation():
    """A component small enough for exponent form is emitted in exponent form.

    A real query embedding carries a handful of components small enough that
    `str(float(x))` switches to exponent form. pgvector accepts them — this
    pins that, so a future switch to fixed-point formatting is a deliberate
    change rather than a silent one.
    """
    literal = _vector_literal([7.8e-05, -7.83e-05, 0.5])

    assert literal == "[7.8e-05,-7.83e-05,0.5]"

    # Integers arrive as floats, since pgvector's parser wants a numeric list
    # and `1` from a caller's hand-built vector must not render as `1`-the-int
    # in a way a future formatter change could treat differently.
    assert _vector_literal([1, 2]) == "[1.0,2.0]"


@pytest.mark.integration
async def test_filters_that_match_nothing_return_nothing(corpus):
    """A filter matching no document returns nothing, from both arms.

    This is the filter failure worth catching. A predicate that fails to bind
    does not error — it returns the unfiltered corpus, which looks like working
    retrieval and answers about the wrong companies.
    """
    # The corpus is 10-K only, so this filter is satisfiable in principle and
    # matched by nothing in fact — which is the case a predicate that fails to
    # bind gets wrong.
    absent = Filters(forms=("10-Q",))

    assert await vector_search(corpus, _embedding(), limit=10, filters=absent) == []
    assert await text_search(corpus, PRECISE_QUESTION, limit=10, filters=absent) == []

    # Both arms, and both unfiltered, or the assertions above are satisfied by
    # a search that returns nothing whatever it is asked.
    assert await vector_search(corpus, _embedding(), limit=10) != []
    assert await text_search(corpus, PRECISE_QUESTION, limit=10) != []


@pytest.mark.integration
async def test_every_filter_field_binds(corpus):
    """Tickers, both year bounds, and their combination each narrow the result.

    Check each field separately. A shared-parameter typo binds one of them and
    not the others, and any single-field test passes over it.
    """
    async def documents(filters: Filters) -> list:
        """The passages a filtered search actually returned.

        `Hit` carries no ticker or year — only ids — so the check has to go
        back through `hydrate`. That is the same path the retriever takes, and
        it is the only way to see what a filtered search really selected.
        """
        hits = await vector_search(corpus, _embedding(), limit=20, filters=filters)
        assert hits, f"{filters} returned nothing; the assertions below are vacuous"
        return list((await hydrate(corpus, [h.key for h in hits])).values())

    assert {p.ticker for p in await documents(Filters(tickers=("AAPL",)))} == {"AAPL"}

    assert all(p.fiscal_year >= 2025 for p in await documents(
        Filters(fiscal_year_from=2025)
    ))
    assert all(p.fiscal_year <= 2021 for p in await documents(
        Filters(fiscal_year_to=2021)
    ))

    # Together, because binding each alone still allows a shared bind-parameter
    # name where the last one written wins and the rest quietly vanish.
    combined = await documents(
        Filters(tickers=("NVDA",), fiscal_year_from=2023, fiscal_year_to=2024)
    )
    assert {(p.ticker, p.fiscal_year) for p in combined} <= {
        ("NVDA", 2023),
        ("NVDA", 2024),
    }


@pytest.mark.integration
async def test_both_arms_rank_best_first_and_respect_limit(corpus):
    """Results arrive best-first, and never more than `limit` of them.

    Neither is implied by the rest of this file, which asserts *which* rows come
    back and which cell they belong to. Reverse every ORDER BY and drop the
    outer LIMIT and the other twelve tests still pass — while `retrieve` fuses
    on rank, so a reversed arm feeds RRF the worst matches labelled rank 1 and
    every coverage assertion downstream stays green.

    Both arms, because they order by opposite things: cosine distance ascending
    for the vector arm, `ts_rank_cd` descending for the text arm. A single
    direction copied to both is a plausible edit that only one of these catches.
    """
    for hits in (
        await vector_search(corpus, _embedding(), limit=7),
        await text_search(corpus, PRECISE_QUESTION, limit=7),
    ):
        assert hits, "an empty arm makes the assertions below vacuous"
        assert len(hits) <= 7

        # Rank numbers the position, so it must count 1..n whatever the order.
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

        # And the score has to agree with it. The vector arm reports
        # `1 - distance` precisely so both arms read the same direction —
        # higher is better — and a caller can compare within one arm without
        # knowing which produced it.
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), scores

    # Monotonic scores are not enough on their own. Every branch is re-sorted
    # by the outer query, so a branch whose own ORDER BY runs backwards selects
    # the *worst* candidates and then presents them in tidy descending order —
    # the assertions above all hold while the search returns the opposite of
    # what was asked for. What catches it is that a smaller limit must return a
    # prefix of a larger one: the five best are the first five of the fifty
    # best, where the five worst sit at the far end of the fifty worst.
    for shallow, deep in (
        (
            await vector_search(corpus, _embedding(), limit=5),
            await vector_search(corpus, _embedding(), limit=50),
        ),
        (
            await text_search(corpus, PRECISE_QUESTION, limit=5),
            await text_search(corpus, PRECISE_QUESTION, limit=50),
        ),
    ):
        assert len(deep) > len(shallow), "too few matches to tell a prefix from a set"
        assert [h.key for h in shallow] == [h.key for h in deep[: len(shallow)]]

    # The grid ranks inside each cell, so its ordering is a separate claim from
    # the two above and needs its own check.
    grid = await grid_search(
        corpus,
        _embedding(),
        PRECISE_QUESTION,
        tickers=("AAPL",),
        years=(2024,),
        depth=5,
    )
    for arm, hits in grid[("AAPL", 2024)].items():
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), (arm, scores)


@pytest.mark.integration
async def test_both_source_types_are_reachable(corpus):
    """A search over both types can return a table; a chunk-only one cannot.

    46% of a filing's figures appear only inside a table, so a chunk-only
    result set cannot answer a numeric question however well it reads.
    """
    both = await vector_search(corpus, _embedding(), limit=50)
    # That a table *can* surface, not that one always must: the per-branch
    # LIMIT bounds what the union materialises and cannot change which rows
    # survive, so a question with one kind of answer legitimately returns one
    # kind of row.
    assert "table" in {h.source_type for h in both}

    chunks_only = await vector_search(
        corpus, _embedding(), limit=50, source_types=("chunk",)
    )
    assert chunks_only, "the corpus has chunks; an empty list here is a bug"
    assert {h.source_type for h in chunks_only} == {"chunk"}

    # No source types at all is an empty search rather than an unfiltered one —
    # the difference between asking for nothing and asking for everything.
    assert await vector_search(corpus, _embedding(), limit=50, source_types=()) == []


@pytest.mark.integration
async def test_text_search_falls_back_when_every_term_fails(corpus):
    """The fallback fires on zero rows, and only on zero rows.

    `plainto_tsquery` ANDs every lexeme, so two real analyst questions match
    nothing at all. Falling back on *zero* rather than on "fewer than limit" is
    what keeps the precise case precise.
    """
    # Every lexeme required, so this matches nothing under AND. Reaching the
    # caller non-empty is the fallback doing its job.
    assert await text_search(corpus, LONG_QUESTION, limit=10) != []

    # And the other half: a question AND *can* satisfy must not be widened.
    # Chunks only, so the check below is one table rather than a union.
    precise = await text_search(
        corpus, PRECISE_QUESTION, limit=10, source_types=("chunk",)
    )
    assert precise

    matched_under_and = await corpus.scalar(
        text(
            "SELECT count(*) FROM document_chunks"
            " WHERE id = ANY(:ids)"
            "   AND search_vector @@ plainto_tsquery('english', :query)"
        ),
        {"ids": [h.row_id for h in precise], "query": PRECISE_QUESTION},
    )
    # Every row still satisfies AND. A fallback on "fewer than limit" would
    # drag in OR's rows here, and OR returns Item 1C Cybersecurity — dense in
    # "risk" and about something else entirely.
    assert matched_under_and == len(precise)


@pytest.mark.integration
async def test_an_all_stopword_question_matches_nothing(corpus):
    """A question with no lexemes returns nothing, under either mode.

    It yields a NULL tsquery, and that is the honest answer — there is no
    lexical signal to rank on. The caller treats it as a normal empty arm.
    """
    # `text_search` runs AND and then OR, so an empty list is both modes
    # answering the same way rather than the fallback being skipped.
    assert await text_search(corpus, "what about it", limit=10) == []


@pytest.mark.integration
async def test_grid_search_partitions_per_cell(corpus):
    """Every requested cell is its own key, ranked 1..n inside itself.

    The partition is what makes this different from a filtered search: ranking
    happens inside the cell, so a company-year cannot be crowded out by a
    stronger one.
    """
    tickers = ("AAPL", "NVDA")
    years = (2023, 2024)
    grid = await grid_search(
        corpus,
        _embedding(),
        PRECISE_QUESTION,
        tickers=tickers,
        years=years,
        depth=3,
    )

    assert set(grid) == {(t, y) for t in tickers for y in years}

    for cell, arms in grid.items():
        assert arms, f"{cell} has no arms"
        for arm, hits in arms.items():
            # Ranks restart at 1 in every cell. Continuing across cells would
            # mean the window partitioned by nothing, which a set of hits per
            # cell would not reveal on its own.
            assert [h.rank for h in hits] == list(range(1, len(hits) + 1)), (cell, arm)
            assert len(hits) <= 3

        # The rows really belong to the cell they are filed under. A window
        # partitioned by the wrong column still produces tidy 1..n ranks under
        # keys that look right, and only the documents say otherwise.
        keys = [h.key for arm in arms.values() for h in arm]
        passages = await hydrate(corpus, keys)
        assert {(p.ticker, p.fiscal_year) for p in passages.values()} == {cell}


@pytest.mark.integration
async def test_grid_search_falls_back_per_cell_not_per_statement(corpus):
    """A question AND reaches in only some cells still gets a text arm everywhere.

    Measured across the 25 cells, `plainto_tsquery` reaches 10 for "supplier
    concentration risk" and 20 for "capital expenditures". A statement-wide
    fallback fires only when every cell is empty, so those queries would leave
    15 and 5 cells ranked by the vector arm alone — decided by whichever company
    happened to match. This is the assertion that catches that, and it passes
    trivially on a question where AND reaches everything or nothing.
    """
    tickers = [t for (t,) in await corpus.execute(
        text("SELECT DISTINCT ticker FROM source_documents ORDER BY ticker")
    )]
    years = [y for (y,) in await corpus.execute(
        text("SELECT DISTINCT fiscal_year FROM source_documents ORDER BY fiscal_year")
    )]

    # The premise, asserted rather than assumed. On a question AND reaches in
    # every cell — or in none — the assertion below holds however the fallback
    # is scoped, and the test would be worthless without ever saying so.
    reached = await corpus.scalar(
        text(
            "SELECT count(*) FROM ("
            "  SELECT d.ticker, d.fiscal_year"
            "    FROM document_chunks c"
            "    JOIN source_documents d ON d.id = c.document_id"
            "   WHERE c.search_vector @@ plainto_tsquery('english', :query)"
            "   GROUP BY 1, 2"
            ") cells"
        ),
        {"query": PRECISE_QUESTION},
    )
    assert 0 < reached < len(tickers) * len(years)

    grid = await grid_search(
        corpus,
        _embedding(),
        PRECISE_QUESTION,
        tickers=tickers,
        years=years,
        depth=2,
    )

    assert len(grid) == len(tickers) * len(years)
    # Every cell, not most of them. The cells AND missed are exactly the ones a
    # statement-wide fallback would leave ranked by the vector arm alone.
    missing = [cell for cell, arms in grid.items() if "text" not in arms]
    assert missing == []


@pytest.mark.integration
async def test_grid_search_omits_cells_with_no_rows(corpus):
    """A cell the corpus has no rows for is absent, not present and empty.

    The caller has to tell "nothing matched" from "nothing exists" — the second
    is a company that did not file, and reporting it as a miss invites a
    fabricated negative.
    """
    grid = await grid_search(
        corpus,
        _embedding(),
        PRECISE_QUESTION,
        tickers=("AAPL",),
        years=(2024, 1999),
        depth=2,
    )

    # Absent, not `{("AAPL", 1999): {}}`. An empty cell reads as "we looked and
    # found nothing", which is a different claim about a year nobody filed in.
    assert set(grid) == {("AAPL", 2024)}


@pytest.mark.integration
async def test_hydrate_returns_both_source_types(corpus):
    """Chunks and tables come back together, and a vanished row is just absent."""
    chunk_id = await corpus.scalar(text("SELECT id FROM document_chunks LIMIT 1"))
    table_id = await corpus.scalar(text("SELECT id FROM document_tables LIMIT 1"))
    missing = ("chunk", uuid4())

    passages = await hydrate(
        corpus, [("chunk", chunk_id), ("table", table_id), missing]
    )

    assert set(passages) == {("chunk", chunk_id), ("table", table_id)}

    chunk = passages[("chunk", chunk_id)]
    table = passages[("table", table_id)]

    # The citation fields, on both types. A passage that cannot say which
    # filing it came from cannot be cited, and an answer it supports cannot be
    # checked.
    for passage in (chunk, table):
        assert passage.text
        assert passage.ticker
        assert passage.fiscal_year
        assert passage.form
        assert passage.document_id

    # A table's caption is its own column; a chunk has none, and carries its
    # heading inside the text instead.
    assert table.title is not None
    assert chunk.title is None

    assert await hydrate(corpus, []) == {}
