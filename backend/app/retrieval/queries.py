"""Bounded pgvector and full-text queries over chunks and tables.

Two retrievable source types, queried the same way: `document_chunks` holds
prose, `document_tables` holds financial tables kept whole. 46% of a filing's
figures appear only in a table, so a chunk-only search cannot answer half the
questions in the brief.

Measured against the ingested corpus (25 filings, 2,321 chunks, 1,724 tables),
not assumed:

- **Rank full-text with `ts_rank_cd`, never `ts_rank`.** `ts_rank` sums term
  weights and ignores where the terms sit, so "supplier concentration risk"
  returned four NVDA *Item 15* chunks tied at exactly 0.2464 — one scattered
  occurrence of each term, and no length normalisation to break the tie.
  `ts_rank_cd` scores cover density and returns Apple's Item 1A instead.
  `setweight` on heading vs body was tested and changed nothing; skip it.
- **`ts_rank_cd` needs positions in the tsvector.** `search_vector` is a stored
  generated column built by `to_tsvector`, so they are there. Never `strip()` it.
- **A filtered vector search does not use the HNSW index.** Postgres narrows on
  the filter, then sorts exactly. At this corpus size that is both faster and
  more accurate than approximate search — every shape below runs in 0.7-7.4 ms
  warm. It stops being true somewhere north of ~100k rows, where the filter
  stops being selective enough to save the scan.
- **Filter by joining `source_documents`, not by `chunk_metadata @>`.** Warm,
  the two are indistinguishable (3.0 ms each). The join wins on correctness:
  `fiscal_year BETWEEN` is one predicate against a typed column, where
  jsonb_path_ops supports only containment and needs a year-per-term OR
  expansion. `chunk_metadata` is a denormalised copy that can drift; the
  document row cannot.

Every query here is bounded. An unbounded retrieval that returns 2,000 passages
is not a retrieval — the caller cannot fuse it, the model cannot read it, and
the cost lands on whoever forgot the LIMIT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from sqlalchemy import Integer, Text, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.ext.asyncio import AsyncSession

SourceType = Literal["chunk", "table"]

# Cover density is meaningless without normalisation: rank/(rank+1) with flag 32
# keeps scores in [0,1) but destroys the spread RRF needs. Flag 1 divides by the
# log of document length, which is what stops a 700-token chunk beating a
# 200-token one purely by having more room for the terms to appear in.
TS_RANK_NORMALIZATION = 1


@dataclass(frozen=True)
class Filters:
    """What to narrow the corpus to before ranking.

    Empty means "no filter on this field", not "match nothing" — the questions
    in the brief are mostly unfiltered or company-scoped, and a filter that
    silently excluded everything would look identical to a corpus gap.

    Years are a closed range because nine of the ten brief questions span
    2021-2025. A set of discrete years would make "how did X change over time"
    the awkward case rather than the normal one.
    """

    tickers: tuple[str, ...] = ()
    fiscal_year_from: int | None = None
    fiscal_year_to: int | None = None
    forms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tickers", tuple(dict.fromkeys(t.strip().upper() for t in self.tickers))
        )
        if (
            self.fiscal_year_from is not None
            and self.fiscal_year_to is not None
            and self.fiscal_year_from > self.fiscal_year_to
        ):
            raise ValueError(
                f"fiscal_year_from={self.fiscal_year_from} is after "
                f"fiscal_year_to={self.fiscal_year_to}; this range matches nothing"
            )

    @property
    def matches_everything(self) -> bool:
        """True when no predicate would be emitted at all.

        Deliberately not a claim about the corpus: tickers naming all five
        issuers also matches every row, but only the corpus knows that. This
        answers the narrower question the caller can act on — whether the join
        to `source_documents` can be skipped entirely.
        """
        return (
            not self.tickers
            and not self.forms
            and self.fiscal_year_from is None
            and self.fiscal_year_to is None
        )


# A frozen instance is safe as a default, but ruff flags any call in a default
# argument (B008) and the rule is worth keeping for the mutable cases.
NO_FILTERS = Filters()


@dataclass
class Hit:
    """One ranked result, from either source type and either search.

    Carries `rank` as well as `score` because RRF consumes rank alone — the
    scores of a cosine search and a ts_rank_cd search are not comparable and
    must never be added.
    """

    source_type: SourceType
    row_id: UUID
    document_id: UUID
    rank: int
    score: float

    # Text is fetched separately by `hydrate`, after fusion has discarded most
    # of these. Carrying it here would mean selecting 2 x limit bodies to throw
    # away most of them.
    @property
    def key(self) -> tuple[SourceType, UUID]:
        return (self.source_type, self.row_id)


@dataclass(frozen=True)
class Passage:
    """
    A surviving hit with its content and everything a citation needs.
    'title' stays separate from `text` rather than concatenated the way
    `search_vector` concatenates them for tables: indexing wants one blob, a
    citation header wants the caption on its own.
    """

    source_type: SourceType
    row_id: UUID
    document_id: UUID
    text: str
    title: str | None
    ticker: str
    fiscal_year: int
    form: str

    # The rest of what a `message_citations` row needs, carried here because
    # `_HYDRATE` already has the joined document row in hand. Fetching them when
    # the citation is written instead would be a second query per citation, on
    # the request path, for columns this statement just read.
    filing_date: date
    company_name: str | None
    # For a chunk a column; for a table the section recorded in `table_data`.
    # Also what `_supports` checks a quote against, so carrying it widens the
    # surface a legitimate quote can come from — see grounding/validator.py.
    section: str | None

_HYDRATE = text("""
    SELECT 'chunk'::text  AS source_type,
           c.id           AS row_id,
           c.document_id  AS document_id,
           c.text         AS text,
           NULL::text     AS title,
           d.ticker       AS ticker,
           d.fiscal_year  AS fiscal_year,
           d.form         AS form,
           d.filing_date  AS filing_date,
           d.company_name AS company_name,
           c.section      AS section
      FROM document_chunks c
      JOIN source_documents d ON d.id = c.document_id
     WHERE c.id = ANY(:chunk_ids)
    UNION ALL
    SELECT 'table'::text  AS source_type,
           t.id           AS row_id,
           t.document_id  AS document_id,
           t.markdown     AS text,
           t.title        AS title,
           d.ticker       AS ticker,
           d.fiscal_year  AS fiscal_year,
           d.form         AS form,
           d.filing_date  AS filing_date,
           d.company_name AS company_name,
           t.table_data->>'section' AS section
      FROM document_tables t
      JOIN source_documents d ON d.id = t.document_id
     WHERE t.id = ANY(:table_ids)
""").bindparams(
    bindparam("chunk_ids", type_=ARRAY(PgUUID(as_uuid=True))),
    bindparam("table_ids", type_=ARRAY(PgUUID(as_uuid=True))),
)

async def hydrate(
        session: AsyncSession,
        keys: Sequence[tuple[SourceType, UUID]],
) -> Mapping[tuple[SourceType, UUID], Passage]:
    """Fetch the content and citation fields for already-ranked rows.

    Takes keys rather than hits because identity is all it needs, and both a
    `Hit` and a `FusedHit` expose `.key`. Anything missing from the result is a
    row that disappeared between the search and this call; the caller decides
    what that means.
    """
    if not keys:
        return {}

    ids: dict[SourceType, list[UUID]] = {"chunk": [], "table": []}
    seen: set[tuple[SourceType, UUID]] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            ids[key[0]].append(key[1])

    result = await session.execute(
        _HYDRATE,
        {"chunk_ids": ids["chunk"], "table_ids": ids["table"]},
    )
    return {
        (row["source_type"], row["row_id"]): Passage(**row)
        for row in result.mappings()
    }
    
    

_SOURCES: dict[SourceType, tuple[str, str]] = {
    "chunk": ("document_chunks", "c"),
    "table": ("document_tables", "t"),
}

_QVEC = "(:embedding)::vector"

def _filter_predicates(filters: Filters) -> tuple[list[str], dict[str, object]]:
    """WHERE fragments against `source_documents d`, plus their params.

    Shared with `text_search` deliberately. The two searches must narrow to the
    same corpus subset or fusion is combining rankings over different
    populations — and unlike a tsquery that drifts, that failure is invisible
    in the output.

    Emits `>=` / `<=` rather than the `BETWEEN` the module docstring names: a
    half-open range (from 2023, no upper bound) is a legal `Filters` and
    BETWEEN cannot express it. With both bounds present they plan identically.
    """
    if filters.matches_everything:
        return [], {}

    preds: list[str] = []
    params: dict[str, object] = {}
    if filters.tickers:
        preds.append("d.ticker = ANY(:tickers)")
        params["tickers"] = list(filters.tickers)
    if filters.forms:
        preds.append("d.form = ANY(:forms)")
        params["forms"] = list(filters.forms)
    if filters.fiscal_year_from is not None:
        preds.append("d.fiscal_year >= :fy_from")
        params["fy_from"] = filters.fiscal_year_from
    if filters.fiscal_year_to is not None:
        preds.append("d.fiscal_year <= :fy_to")
        params["fy_to"] = filters.fiscal_year_to
    return preds, params

def _vector_literal(embedding: list[float]) -> str:
    """pgvector's text input format.

    `str(float(x))` emits scientific notation for the smallest components — a
    real query embedding has a handful like -7.83e-05 — and pgvector's parser
    accepts them, so no formatting is needed.
    """
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _union(branches: list[str]) -> str:
    """Combine per-source-type branches.

    Each branch is parenthesised because it carries its own LIMIT, and Postgres
    rejects a bare LIMIT inside a UNION arm. A single-source-type search has one
    branch and parses either way, which is exactly why this is easy to miss.
    """
    return " UNION ALL ".join(f"({branch})" for branch in branches)


def _branches(
    template: str, source_types: tuple[SourceType, ...], preds: list[str], extra: str = ""
) -> list[str]:
    """Fill `template` once per source type, with the shared filter predicates.

    The JOIN is emitted only when something actually filters on the document —
    an unfiltered search has no reason to touch `source_documents` at all.
    """
    out = []
    for source_type in source_types:
        table, alias = _SOURCES[source_type]
        where = [*([extra.format(a=alias)] if extra else []), *preds]
        out.append(template.format(
            source_type=source_type,
            table=table,
            a=alias,
            join=(f"JOIN source_documents d ON d.id = {alias}.document_id" if preds else ""),
            where=(f"WHERE {' AND '.join(where)}" if where else ""),
        ))
    return out

def _array_params(params: dict[str, object]) -> list:
    """Type the array binds. `= ANY(:tickers)` needs a real text[], not a list."""
    return [
        bindparam(name, type_=ARRAY(Text()))
        for name in ("tickers", "forms")
        if name in params
    ]


# Table and alias names come from _SOURCES, never from a caller — the only
# interpolated values here are module constants. Every caller-supplied value is
# a bound parameter.
_VECTOR_BRANCH = f"""
    SELECT '{{source_type}}'::text          AS source_type,
           {{a}}.id                         AS row_id,
           {{a}}.document_id                AS document_id,
           {{a}}.embedding <=> {_QVEC}      AS distance
      FROM {{table}} {{a}}
      {{join}}
     {{where}}
     ORDER BY {{a}}.embedding <=> {_QVEC}
     LIMIT :limit
"""


async def vector_search(
    session: AsyncSession,
    embedding: list[float],
    *,
    limit: int,
    filters: Filters = NO_FILTERS,
    source_types: tuple[SourceType, ...] = ("chunk", "table"),
) -> list[Hit]:
    """Top-`limit` rows by cosine distance, nearest first.

    Use the `<=>` operator, not `<->` or `<#>`: the HNSW indexes are built with
    `vector_cosine_ops`, and an operator that does not match the index opclass
    silently falls back to a sequential scan.

    Distance is not similarity. `<=>` returns 0 for identical and 2 for
    opposite, so `score` should be `1 - distance` if you want it to read the
    same direction as the full-text score. Rank is what fusion uses either way.

    Both source types are ranked in one list, so a table and a chunk compete on
    the same footing. That is deliberate: for "Apple net sales by category" the
    right answer is a table, and for "how did they describe supply risk" it is
    prose. Deciding by source type up front would be guessing at the question.
    """
    if limit < 1:
        raise ValueError(f"limit={limit} is not a retrieval")
    if not source_types:
        return []

    preds, params = _filter_predicates(filters)
    branches = _branches(_VECTOR_BRANCH, source_types, preds)

    stmt = text(f"""
        SELECT source_type, row_id, document_id, distance
          FROM ({_union(branches)}) AS candidates
         ORDER BY distance
         LIMIT :limit
    """).bindparams(bindparam("embedding", type_=Text()), *_array_params(params))

    result = await session.execute(
        stmt,
        {"embedding": _vector_literal(embedding), "limit": limit, **params},
    )
    return [
        Hit(
            source_type=row.source_type,
            row_id=row.row_id,
            document_id=row.document_id,
            rank=i,
            score=1.0 - row.distance,
        )
        for i, row in enumerate(result, start=1)
    ]



# Two ways to turn a question into a tsquery. `:query` is bound once and reused
# by both the `@@` filter and the `ts_rank_cd` ordering in every branch, so the
# two cannot drift into ranking a row that does not match.
#
# `websearch_to_tsquery` is not a third option: measured on the corpus it
# returns exactly what `plainto_tsquery` returns for every question tried. Its
# extras are quoting, OR and negation in the *user's* syntax, which an analyst
# writing prose never types. It does not soften the AND.
_TSQUERY_ALL = "plainto_tsquery('english', :query)"
_TSQUERY_ANY = (
    "(SELECT string_agg(lexeme, ' | ') FROM unnest("
    "tsvector_to_array(to_tsvector('english', :query))) AS lexeme)::tsquery"
)

_TEXT_BRANCH = """
    SELECT '{{source_type}}'::text AS source_type,
           {{a}}.id                AS row_id,
           {{a}}.document_id       AS document_id,
           ts_rank_cd({{a}}.search_vector, {tsquery}, {norm}) AS score
      FROM {{table}} {{a}}
      {{join}}
     {{where}}
     ORDER BY score DESC
     LIMIT :limit
"""


async def text_search(
    session: AsyncSession,
    query: str,
    *,
    limit: int,
    filters: Filters = NO_FILTERS,
    source_types: tuple[SourceType, ...] = ("chunk", "table"),
) -> list[Hit]:
    """Top-`limit` rows by `ts_rank_cd`, highest first.

    Requires every term, then falls back to requiring any of them only when
    that returns nothing at all. Both halves are measured, and neither is
    optional:

    - AND is more precise when it matches. "supplier concentration risk" under
      AND returns Apple's Item 1A Risk Factors; under OR it returns Item 1C
      Cybersecurity, which is dense in "risk" and about something else.
    - AND returns *zero rows* for a real analyst question. "how did the revenue
      mix between iPhone, Services, Mac, iPad and Wearables change" and
      "NVIDIA demand drivers, customer concentration and supply constraints for
      Data Center" both match nothing, because `plainto_tsquery` ANDs all
      fifteen-odd lexemes. Under OR they return Apple's Item 1 Business and
      NVIDIA's Item 1A/Item 7 respectively — the right passages.

    Falling back only on zero is what keeps the precise case precise. A
    threshold of "fewer than limit" would drag OR's noise into every narrow
    query that legitimately has few matches.

    An all-stopword query yields a NULL tsquery, which matches nothing under
    both modes. That is the honest answer: there is no lexical signal in "what
    about it".
    """
    if limit < 1:
        raise ValueError(f"limit={limit} is not a retrieval")
    if not source_types:
        return []

    preds, params = _filter_predicates(filters)

    async def run(tsquery: str) -> list[Hit]:
        template = _TEXT_BRANCH.format(tsquery=tsquery, norm=TS_RANK_NORMALIZATION)
        branches = _branches(
            template, source_types, preds, extra=f"{{a}}.search_vector @@ {tsquery}"
        )
        stmt = text(f"""
            SELECT source_type, row_id, document_id, score
              FROM ({_union(branches)}) AS candidates
             ORDER BY score DESC
             LIMIT :limit
        """).bindparams(*_array_params(params))
        result = await session.execute(
            stmt, {"query": query, "limit": limit, **params}
        )
        return [
            Hit(
                source_type=row.source_type,
                row_id=row.row_id,
                document_id=row.document_id,
                rank=i,
                score=row.score,
            )
            for i, row in enumerate(result, start=1)
        ]

    return await run(_TSQUERY_ALL) or await run(_TSQUERY_ANY)


# One (ticker, fiscal_year) pair. The unit a comparative question is really
# asking about: "how did this change from 2021 to 2025" is twenty-five of these
# for five companies, not one ranking of ten.
Cell = tuple[str, int]


# One pool branch per source type. Both tables carry `embedding` and
# `search_vector`, so a single pool feeds both arms and the filtering join to
# `source_documents` happens once instead of four times.
_GRID_POOL_BRANCH = """
    SELECT '{source_type}'::text AS source_type,
           {a}.id                AS row_id,
           {a}.document_id       AS document_id,
           d.ticker              AS ticker,
           d.fiscal_year         AS fiscal_year,
           {a}.embedding         AS embedding,
           {a}.search_vector     AS search_vector
      FROM {table} {a}
      JOIN source_documents d ON d.id = {a}.document_id
     WHERE d.ticker = ANY(:tickers) AND d.fiscal_year = ANY(:years)
"""

# Both arms rank inside the cell, and both break ties on row_id. Ties are not
# hypothetical here: ts_rank_cd returned four NVDA chunks at exactly 0.2464
# corpus-wide, and inside a ~100-row cell an exact tie is likelier still.
# Without a total order `rn` would be arbitrary, and `rn` is what RRF consumes —
# an unstable rank means an unstable answer between two runs of one question.
_GRID_ARM = """
    SELECT '{arm}'::text   AS arm,
           p.source_type   AS source_type,
           p.row_id        AS row_id,
           p.document_id   AS document_id,
           p.ticker        AS ticker,
           p.fiscal_year   AS fiscal_year,
           {score}         AS score,
           row_number() OVER (
               PARTITION BY p.ticker, p.fiscal_year
               ORDER BY {order}, p.row_id
           )               AS rn
      FROM pool p {join}
     WHERE {where}
"""


async def grid_search(
    session: AsyncSession,
    embedding: list[float],
    query: str,
    *,
    tickers: Sequence[str],
    years: Sequence[int],
    depth: int,
    source_types: tuple[SourceType, ...] = ("chunk", "table"),
) -> dict[Cell, dict[str, list[Hit]]]:
    """Both arms, ranked *within* every (ticker, fiscal_year) cell, in one query.

    Returns each cell's two ranked lists in the shape `fuse` already consumes:
    {("AAPL", 2024): {"vector": [...], "text": [...]}}. The caller fuses per
    cell — see `retrieve_grid`.

    Why this exists rather than a bigger `limit`, measured on brief question 10:
    a single ranked search never surfaces Amazon at all, not at limit=10 and not
    at limit=100 with 64,889 tokens. Amazon genuinely does not rank in the top
    hundred for that question, so no value of `limit` reaches it. Coverage is
    not something you can buy with a larger sample of the wrong ordering.

        retrieve(limit=10)     3 of 5 companies    6,104 tokens
        retrieve(limit=100)    4 of 5 companies   64,889 tokens
        this, per_cell=1      25 of 25 cells      14,636 tokens

    The partition is what makes this different from `vector_search` under a
    filter: ranking happens *inside* each cell, so every company-year gets its
    own top-k and cannot be crowded out by a stronger one.

    `depth` is per cell per arm, and plays the role `_arm_limit` plays
    elsewhere: fusion can only reward agreement it was asked about.

    The AND-then-OR fallback of `text_search` is kept, but decided **per cell**
    rather than for the statement as a whole. That is the one place this
    function must differ, and the reason is the same measurement that motivates
    it: inside a single filing-year the pool is ~100 rows, so `plainto_tsquery`
    ANDing fifteen lexemes matches nothing far more often than it does
    corpus-wide. A statement-wide fallback would fire only when *every* cell
    came back empty — so one strong company matching under AND would suppress
    the fallback for the four that did not, and those cells would be reduced to
    a vector-only ranking by a decision made somewhere else entirely. Falling
    back cell by cell keeps AND's precision where AND works, which is the
    property `text_search` measured and this must not lose.

    A cell with no rows at all — a company that did not file that year — is
    absent from the result rather than present and empty, so the caller can tell
    "nothing matched" from "nothing exists". An arm that found nothing in a cell
    the other arm did reach is likewise absent rather than an empty list; `fuse`
    ranks over whichever arms are present, and `contributions` then records that
    only one saw it.
    """
    if depth < 1:
        raise ValueError(f"depth={depth} is not a retrieval")
    if not source_types:
        return {}

    # Normalised the way `Filters` normalises, so a caller cannot address the
    # same cell as both "aapl" and "AAPL" and get two half-filled grids.
    scope = list(dict.fromkeys(t.strip().upper() for t in tickers))
    span = list(dict.fromkeys(years))
    if not scope or not span:
        return {}

    pool = _union([
        _GRID_POOL_BRANCH.format(
            source_type=source_type, table=table, a=alias
        )
        for source_type, (table, alias) in (
            (st, _SOURCES[st]) for st in source_types
        )
    ])

    # A NULL embedding sorts last under ASC rather than being excluded, so an
    # unembedded row would take a slot in a thin cell and rank ahead of nothing.
    # The text arm needs no such guard: `@@` never matches a NULL tsvector.
    vector_arm = _GRID_ARM.format(
        arm="vector",
        score=f"1 - (p.embedding <=> {_QVEC})",
        order=f"p.embedding <=> {_QVEC}",
        join="",
        where="p.embedding IS NOT NULL",
    )
    text_arm = """
        SELECT * FROM text_all
        UNION ALL
        SELECT * FROM text_any a
         WHERE NOT EXISTS (
             SELECT 1 FROM text_all b
              WHERE b.ticker = a.ticker AND b.fiscal_year = a.fiscal_year
         )
    """

    def text_branch(column: str) -> str:
        rank = f"ts_rank_cd(p.search_vector, q.{column}, {TS_RANK_NORMALIZATION})"
        return _GRID_ARM.format(
            arm="text",
            score=rank,
            order=f"{rank} DESC",
            join="CROSS JOIN q",
            where=f"p.search_vector @@ q.{column}",
        )

    # Both tsqueries are built once in their own CTE and joined in, so the `@@`
    # filter and the `ts_rank_cd` ordering cannot drift into ranking a row that
    # does not match — the same guarantee `text_search` gets from binding
    # `:query` once, held across four more uses.
    stmt = text(f"""
        WITH q AS (
            SELECT {_TSQUERY_ALL} AS q_all, {_TSQUERY_ANY} AS q_any
        ),
        pool AS ({pool}),
        vector_arm AS ({vector_arm}),
        text_all AS ({text_branch("q_all")}),
        text_any AS ({text_branch("q_any")}),
        text_arm AS ({text_arm})
        SELECT arm, source_type, row_id, document_id, ticker, fiscal_year, score, rn
          FROM vector_arm
         WHERE rn <= :depth
        UNION ALL
        SELECT arm, source_type, row_id, document_id, ticker, fiscal_year, score, rn
          FROM text_arm
         WHERE rn <= :depth
         ORDER BY ticker, fiscal_year, arm, rn
    """).bindparams(
        bindparam("embedding", type_=Text()),
        bindparam("tickers", type_=ARRAY(Text())),
        bindparam("years", type_=ARRAY(Integer())),
    )

    result = await session.execute(
        stmt,
        {
            "embedding": _vector_literal(embedding),
            "query": query,
            "tickers": scope,
            "years": span,
            "depth": depth,
        },
    )

    # Rows arrive ordered by (ticker, fiscal_year, arm, rn), so appending in
    # order is enough — no per-cell sort afterwards.
    grid: dict[Cell, dict[str, list[Hit]]] = {}
    for row in result:
        arms = grid.setdefault((row.ticker, row.fiscal_year), {})
        arms.setdefault(row.arm, []).append(
            Hit(
                source_type=row.source_type,
                row_id=row.row_id,
                document_id=row.document_id,
                rank=row.rn,
                score=row.score,
            )
        )
    return grid


# How the two source types combine: one UNION per search, so each search returns
# a single ranked list with chunks and tables interleaved. Two lists reach
# fusion, not four.
#
# The per-branch LIMIT is an optimisation, not a fairness guarantee. It bounds
# what the union materialises; it cannot change which rows survive, because any
# row in the global top-k is also in its own branch's top-k. Dropping it returns
# exactly the same results, more slowly.
#
# So a search genuinely can come back all one type, and that is the ranking
# working: "supplier concentration risk" is ten chunks at limit=10, "Apple net
# sales by category" is ten tables. Neither is crowded out — the question simply
# has one kind of answer. A caller that needs both regardless has to ask for
# both, with `source_types`, or ask deeper.
#
# What this does *not* solve, and fusion must: a single top-10 over the whole
# corpus can legitimately be ten NVDA chunks. That is right for "what did NVIDIA
# say about supply" and useless for question 6 of the brief, "which of the five
# companies changed risk-factor language between 2021 and 2025". Spreading
# results across companies and years is a property of the caller's question, not
# of a SQL ranking — it belongs in the retriever, either as per-ticker searches
# or as a diversity pass after fusion.
