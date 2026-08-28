"""Question in, ranked passages out.

The whole retrieval path: embed the question, run both arms, fuse on rank,
hydrate the survivors, optionally widen each one with its neighbours. Nothing
above this module should know that there are two arms, or that tables are a
separate table.

Measured on the ingested corpus, and both findings change what this file has to
do rather than merely how it does it:

- **A single top-10 does not cover a comparative question.** Brief question 6
  asks which of *five* companies changed risk-factor language; the top-10 comes
  back NVDA x6, AAPL x4, with Alphabet, Microsoft and Amazon absent entirely.
  Question 8 names four companies and returns two. Question 1 asks only about
  Apple and correctly returns ten Apple passages — so this is not a ranking bug
  to fix in `fusion.py`. Ranking by relevance is right; the question simply
  needs several searches, not a bigger k.
- **Neighbouring chunks cross section boundaries 23.3% of the time** (536 of
  2,296 adjacent pairs). Expanding a hit to `chunk_index ± 1` therefore drags in
  another Item's prose one time in four, and the citation still reads as the
  original Item. Neighbours must be constrained to the same section, not merely
  the same document.

Neighbour expansion also costs about 3x the tokens — a mean chunk is 540 tokens
and its two neighbours add 1,056 — so it is a per-call choice, not a default.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embeddings import embed_texts
from app.retrieval.fusion import RRF_K, FusedHit, fuse
from app.retrieval.queries import (
    NO_FILTERS,
    Filters,
    Passage,
    SourceType,
    hydrate,
    text_search,
    vector_search,
)

# Each arm is asked for more than the caller wants, because fusion only rewards
# a passage for appearing in both — a passage ranked 18th by one arm and 2nd by
# the other is exactly the result hybrid retrieval exists to surface, and it is
# invisible if each arm only returns 10. Measured: for the NVIDIA question the
# 3rd fused result was {'text': 2, 'vector': 19}.
ARM_MULTIPLIER = 2
MIN_ARM_LIMIT = 20

# How many passages each company gets in a fan-out. Question 6 names five
# companies, and "which of them changed their risk-factor language" is answered
# by a couple of passages each, not ten — five at three is 15 passages and
# roughly 8k tokens, where five at ten would be 50 and would not fit a turn
# alongside the rest of the prompt.
DEFAULT_PER_TICKER = 3


@dataclass
class RetrievedPassage:
    """A fused hit with its content, its provenance, and optional context."""

    passage: Passage
    score: float
    rank: int
    # arm name -> rank in that arm. Kept this far up on purpose: when a brief
    # question fails, "never retrieved" and "retrieved at rank 19 and fusion
    # buried it" need different fixes, and this is what tells them apart.
    contributions: dict[str, int]
    # Neighbouring chunks in the same section, joined. None when not requested.
    context: str | None = None


async def retrieve(
    session: AsyncSession,
    question: str,
    *,
    limit: int = 10,
    filters: Filters = NO_FILTERS,
    with_context: bool = False,
    k: int = RRF_K,
) -> list[RetrievedPassage]:
    """Embed, search both arms, fuse, hydrate.

    TODO: implement.

    Steps, in order:
      1. `embed_texts([question])` — one call, one vector.
      2. `vector_search` and `text_search`, each at `_arm_limit(limit)`.
      3. `fuse({"vector": ..., "text": ...}, limit=limit, k=k)`.
      4. `hydrate` the survivors, once, for all of them.
      5. If `with_context`, widen each chunk with `neighbours`.

    Run the two arms concurrently with `asyncio.gather`. They are independent
    queries against the same session — check that works before relying on it,
    because an AsyncSession is not safe for concurrent use and may need two.

    A question with no lexical signal at all ("what about it") yields a NULL
    tsquery and an empty text arm. That is not an error: fusion over one arm is
    still a valid ranking, and the vector arm always returns something.
    """
    if limit <= 0:
        return []

    arm_limit = _arm_limit(limit)

    # The text arm doesn't need the embedding, so start the embed call and run
    # the lexical query underneath it. Only one DB operation is ever in flight,
    # so this is safe on a single session.
    embedding = asyncio.create_task(embed_texts([question]))
    try:
        text_hits = await text_search(
            session, question, limit=arm_limit, filters=filters
        )
        (question_vector,) = await embedding
    except BaseException:
        embedding.cancel()
        raise

    vector_hits = await vector_search(
        session, question_vector, limit=arm_limit, filters=filters
    )

    fused = fuse({"vector": vector_hits, "text": text_hits}, limit=limit, k=k)
    if not fused:
        return []

    passages = await hydrate(session, [f.key for f in fused])

    results: list[RetrievedPassage] = []
    for f in fused:
        passage = passages.get(f.key)
        if passage is None:
            continue  # row went away between search and hydrate
        results.append(
            RetrievedPassage(
                passage=passage,
                score=f.score,
                rank=len(results) + 1,
                contributions=dict(f.contributions),
            )
        )

    if with_context and results:
        context = await neighbours(session, [r.passage for r in results])
        for r in results:
            r.context = context.get((r.passage.source_type, r.passage.row_id))

    return results


def _arm_limit(limit: int) -> int:
    """How deep each arm searches before fusion.

    The floor matters more than the multiplier: at limit=3 a bare 2x gives each
    arm 6 results, and a passage the other arm ranked 15th can never be
    rewarded for agreement it was never asked about.
    """
    return max(limit * ARM_MULTIPLIER, MIN_ARM_LIMIT)


# Self-join on the anchor rather than a second round trip: the caller already
# holds `row_id`, but not the `chunk_index` or `section` the window is defined
# by — `Passage` carries what a citation needs, and neither of those is it.
# Joining the table to itself looks both up and expands in one statement.
#
# The join predicate is (document_id, chunk_index), which is exactly the
# UniqueConstraint on document_chunks, so the expansion is an index lookup per
# anchor rather than a scan.
#
# `n.section = a.section` is NULL, not true, when either side is NULL, so a
# chunk with no recorded section gets no neighbours at all. That is the
# fail-safe direction: an unknown section cannot be shown to be the same
# section, and a wrong window is worse than a narrow one here.
#
# The anchor matches its own predicate and is deliberately kept — `context` is
# the whole widened passage, in reading order, not a pair of fragments the
# caller would have to reassemble without knowing which came first.
#
# `a.section` comes back so the window can carry the heading once. Every chunk's
# text is prefixed with it by `ingest.chunk`, so joining the rows raw repeats it
# per neighbour — 2.5 times per window, measured — which is redundant tokens in
# a model's context and reads as three sections rather than one passage.
_NEIGHBOURS = text("""
    SELECT a.id      AS anchor_id,
           a.section AS section,
           n.text    AS text
      FROM document_chunks a
      JOIN document_chunks n
        ON n.document_id = a.document_id
       AND n.section     = a.section
       AND n.chunk_index BETWEEN a.chunk_index - :radius
                             AND a.chunk_index + :radius
     WHERE a.id = ANY(:anchor_ids)
     ORDER BY a.id, n.chunk_index
""").bindparams(bindparam("anchor_ids", type_=ARRAY(PgUUID(as_uuid=True))))


async def neighbours(
    session: AsyncSession,
    passages: list[Passage],
    *,
    radius: int = settings.retrieval_neighbor_radius,
) -> dict[tuple[SourceType, UUID], str]:
    """Adjacent chunks, same document *and* same section.

    Returns the widened text per passage, keyed the way every other identity in
    this package is keyed — (source_type, row_id). A passage with nothing to add
    is absent rather than present-and-empty, so the caller's `.get` leaves
    `context` as None and "not widened" stays distinguishable from "widened to
    nothing".

    The same-section constraint is the whole point. 23.3% of adjacent pairs sit
    either side of a section boundary, so `chunk_index ± 1` alone attaches
    another Item's text to this passage's citation one time in four — the exact
    failure the product exists to avoid, and one that looks like a good answer.

    Tables have no neighbours. They are whole by construction, and a table's
    `table_index` neighbour is a different table, not more of this one.

    The window keeps the section heading once, at the top, rather than once per
    neighbour. All three chunks carry the same heading by construction — the
    join requires it — so repeating it says nothing and costs tokens.
    """
    if radius < 1:
        return {}

    # dict.fromkeys, not a set: one passage can be fused in twice under
    # different keys, and a stable order keeps the same query text across calls.
    anchors = list(
        dict.fromkeys(p.row_id for p in passages if p.source_type == "chunk")
    )
    if not anchors:
        return {}

    result = await session.execute(
        _NEIGHBOURS, {"anchor_ids": anchors, "radius": radius}
    )

    windows: dict[tuple[SourceType, UUID], list[str]] = {}
    for row in result:
        key = ("chunk", row.anchor_id)
        if key not in windows:
            # The heading, once. Every row in this window shares it.
            windows[key] = [row.section]
        # removeprefix rather than a slice: it is a no-op if a future chunker
        # stops prefixing, which degrades to today's duplicated heading instead
        # of silently eating the first line of the body.
        body = row.text.removeprefix(f"{row.section}\n\n")
        if body and body != row.section:
            windows[key].append(body)

    return {key: "\n\n".join(parts) for key, parts in windows.items()}


async def retrieve_per_ticker(
    session: AsyncSession,
    question: str,
    tickers: Sequence[str],
    *,
    per_ticker: int = DEFAULT_PER_TICKER,
    filters: Filters = NO_FILTERS,
    with_context: bool = False,
    k: int = RRF_K,
) -> list[RetrievedPassage]:
    """One search per company, merged so every company is represented.

    For a comparative question — "which of these five changed their risk-factor
    language" — a single ranked list is the wrong shape. Ranking by relevance is
    working correctly when it returns NVDA x6 and AAPL x4; those companies
    genuinely have the most on-topic passages, and no value of `limit` or `k`
    changes that. The question needs several searches, not a deeper one.

    Each company is searched under its own `Filters`, so it is ranked against
    itself and cannot be crowded out. Alphabet's second-best passage scores
    0.01639 and is nowhere near a global top-20; only a search that excludes the
    other four companies will ever surface it.

    `filters` carries everything except tickers — a fiscal-year range or a form
    restriction applies to every arm of the fan-out.
    """
    if not tickers:
        raise ValueError(
            "retrieve_per_ticker() has nothing to fan out over. A question that "
            "names no companies wants retrieve()."
        )
    if filters.tickers:
        raise ValueError(
            f"filters.tickers={filters.tickers} would be overwritten per arm. "
            f"Pass the companies as the `tickers` argument instead."
        )
    if per_ticker <= 0:
        return []

    # Same normalisation Filters applies, so these keys and the scoped filter
    # cannot disagree about what "aapl" is.
    scope = list(dict.fromkeys(t.strip().upper() for t in tickers))

    # Embedded once, not once per company: fanning out on filters does not
    # change the question, so N identical embedding calls would be N-1 network
    # round trips and N-1 times the cost for the same vector.
    (question_vector,) = await embed_texts([question])

    # Sequential on purpose. An AsyncSession is not safe for concurrent use, and
    # each arm runs in single-digit milliseconds — five companies is ~30ms of
    # database time against an embedding call an order of magnitude longer.
    arm_limit = _arm_limit(per_ticker)
    by_ticker: dict[str, list[FusedHit]] = {}
    for ticker in scope:
        scoped = replace(filters, tickers=(ticker,))
        text_hits = await text_search(
            session, question, limit=arm_limit, filters=scoped
        )
        vector_hits = await vector_search(
            session, question_vector, limit=arm_limit, filters=scoped
        )
        by_ticker[ticker] = fuse(
            {"vector": vector_hits, "text": text_hits}, limit=per_ticker, k=k
        )

    merged = _interleave(by_ticker)
    if not merged:
        return []

    # One hydration for the whole fan-out, not one per company: the arms are
    # ranked separately but their survivors are just row ids by this point.
    passages = await hydrate(session, [f.key for f in merged])

    results: list[RetrievedPassage] = []
    for f in merged:
        passage = passages.get(f.key)
        if passage is None:
            continue  # row went away between search and hydrate
        results.append(
            RetrievedPassage(
                passage=passage,
                # The within-company RRF score, deliberately not what the merged
                # order sorts by. Two companies' scores are computed over
                # different populations and comparing them is what the fan-out
                # exists to avoid.
                score=f.score,
                rank=len(results) + 1,
                contributions=dict(f.contributions),
            )
        )

    if with_context and results:
        context = await neighbours(session, [r.passage for r in results])
        for r in results:
            r.context = context.get((r.passage.source_type, r.passage.row_id))

    return results


def _interleave(by_ticker: dict[str, list[FusedHit]]) -> list[FusedHit]:
    """Round-robin the per-company lists: every best, then every second.

    This is what keeps `rank` meaningful after the merge. Sorting the merged set
    by score would undo the fan-out at exactly the moment it matters — the
    caller truncating to fit a context window — and hand back the same
    NVDA-heavy top-10 the fan-out was built to avoid. Round-robin makes the
    guarantee survive truncation: cut the list anywhere and the companies are
    still within one passage of each other.

    So `rank` no longer means "most relevant in the corpus". It means position
    in a deliberately balanced ordering, and rank 1 is the best passage for the
    strongest company at that depth. A caller that wants pure relevance wants
    `retrieve`.

    Within a round, order by score and then by ticker: the score comparison is
    across populations and so is only a presentation choice, but it is a
    deterministic one, and determinism is what makes the output diffable
    between runs.

    No cross-company dedupe, because a document has exactly one ticker and the
    arms are disjoint by construction.
    """
    depth = max((len(hits) for hits in by_ticker.values()), default=0)

    merged: list[FusedHit] = []
    for i in range(depth):
        round_i = [
            (hits[i], ticker) for ticker, hits in by_ticker.items() if i < len(hits)
        ]
        round_i.sort(key=lambda pair: (-pair[0].score, pair[1]))
        merged.extend(hit for hit, _ in round_i)
    return merged


# Why fan out on filters rather than the two alternatives.
#
# A diversity pass — fuse once, then cap at N per ticker — cannot work here, and
# the numbers say so rather than the intuition. Alphabet's second-best passage
# for question 6 scores 0.01639. That is nowhere near a global top-20, so it is
# not in the list a diversity pass would be trimming. Capping can only ever
# redistribute what relevance already surfaced; it cannot conjure a company that
# never appeared. Amazon is in exactly that position.
#
# Leaving it to the agent stays available and is not exclusive with this: the
# agent still decides *which* companies a question is about, and passes them in.
# What it no longer has to do is decide retrieval strategy in the prompt, or
# reassemble N tool results into one balanced ranking. The merge rule is a
# property of retrieval, and it is testable here in a way a prompt is not.
