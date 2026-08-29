"""Tests for app.retrieval.fusion.

Pure functions over ranks. No database, no network, no corpus.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.retrieval.fusion import fuse


def test_agreement_beats_a_single_strong_rank(hit):
    """A passage ranked 10th by *both* arms outranks one ranked 1st by one.

    2/(60+10) = 0.0286 against 1/(60+1) = 0.0164. This is the entire point of
    RRF, and it is the property an implementation gets backwards while still
    looking plausible — summing scores instead of ranks, or overwriting instead
    of accumulating, both pass a casual read.
    """
    fused = fuse({"vector": [hit(2, 1), hit(1, 10)], "text": [hit(1, 10)]}, limit=10)

    assert [f.row_id for f in fused] == [UUID(int=1), UUID(int=2)]
    assert [f.rank for f in fused] == [1, 2]

    # The arithmetic, not only the order. Order alone does not separate this
    # from an implementation that sums the arms' own scores: those are 20 and 1
    # here, which sorts the same way and agrees with RRF by luck. Overwriting
    # instead of accumulating is what the doubled value catches — it would
    # leave the agreed passage at 1/70, behind the single strong rank.
    assert fused[0].score == pytest.approx(2 / 70)
    assert fused[1].score == pytest.approx(1 / 61)


def test_limit_truncates_after_ranking(hit):
    """`limit` keeps the best results, not the first ones seen.

    Every caller supplies a meaningful limit — `retrieve` its top-k,
    `retrieve_per_ticker` its per-company share, `retrieve_grid` its per-cell
    share — so a `limit` that did nothing would not fail loudly. It would
    quietly return every candidate: a 25-cell grid at per_cell=1 would hand
    back 500 passages instead of 25, and the caller's context window would find
    out before any assertion did.
    """
    # Row 5 is the vector arm's *last* result and the only one the text arm
    # also found, so agreement lifts it to the top. Truncating the list as it
    # arrives would drop it and keep row 1; truncating after ranking keeps it.
    vector = [hit(1, 1), hit(2, 2), hit(3, 3), hit(5, 4)]
    fused = fuse({"vector": vector, "text": [hit(5, 1)]}, limit=2)

    assert [f.row_id.int for f in fused] == [5, 1]
    # Ranks number the survivors, not their positions in the untruncated list.
    assert [f.rank for f in fused] == [1, 2]

    # A limit past the end is not an error, and does not pad.
    assert len(fuse({"vector": vector}, limit=99)) == 4


def test_output_is_independent_of_arm_order(hit):
    """The same arms in a differently-built dict fuse to the same list.

    Ties are the common case here, not a rare collision: where the arms do not
    overlap, every document scores 1/(k+rank) exactly once, so their rank-1
    entries tie, their rank-2 entries tie, and so on. Without a total order the
    ranking falls out of dict insertion order and changes with how the caller
    happened to build `arms`.
    """
    # Zero overlap, which is the real shape: "geographic revenue exposure by
    # country" overlapped in nothing across the measured corpus. Every score
    # here is reached exactly once, so all three pairs tie.
    vector = [hit(1, 1), hit(2, 2), hit(3, 3)]
    text = [hit(4, 1), hit(5, 2), hit(6, 3)]

    assert fuse({"vector": vector, "text": text}, limit=10) == fuse(
        {"text": text, "vector": vector}, limit=10
    )

    fused = fuse({"vector": vector, "text": text}, limit=10)

    # The test is worthless unless the input really does tie. Asserting the
    # duplicate scores keeps it honest: with six distinct scores, order
    # independence would hold under any sort and prove nothing.
    assert len({f.score for f in fused}) == 3

    # Ties fall to the row id, so the order is the one a reader can predict
    # from the integers passed in rather than whatever the dict happened to
    # iterate.
    assert [f.row_id.int for f in fused] == [1, 4, 2, 5, 3, 6]


def test_ties_break_on_best_single_rank(hit):
    """Equal scores are settled by the better placing in any single arm.

    Pin the rule, not just the determinism: sorting by score alone leaves these
    two in either order, and both orders are stable.
    """
    # 1/70 against 1/140 + 1/140. Equal to the last bit — halving and doubling
    # are exact in binary — so the score decides nothing and the next key runs.
    alone = hit(9, 10)
    agreed = hit(2, 80)
    fused = fuse({"vector": [alone, agreed], "text": [agreed]}, limit=10)

    assert fused[0].score == fused[1].score

    # The row ids are deliberately the wrong way round: 2 sorts before 9, so
    # the trailing id tie-break would put `agreed` first. Only the best-single-
    # rank rule puts the passage some arm ranked 10th ahead of the one both
    # arms ranked 80th.
    assert [f.row_id.int for f in fused] == [9, 2]


def test_contributions_record_every_arm(hit):
    """Each result carries the rank every arm gave it, and no others.

    This is the field that separates "not in the corpus" from "one arm found it
    and fusion buried it" — the two diagnoses the CLI exists to tell apart, and
    they need opposite fixes.
    """
    fused = fuse({"vector": [hit(2, 1), hit(1, 3)], "text": [hit(1, 11)]}, limit=10)
    by_row = {f.row_id.int: f for f in fused}

    assert by_row[1].contributions == {"vector": 3, "text": 11}

    # An exact dict, not a subset. A missing arm has to be missing — an entry
    # holding None or 0 would print as a rank in the CLI's header and read as
    # "found at rank 0", which is the opposite of the truth.
    assert by_row[2].contributions == {"vector": 1}


def test_a_chunk_and_a_table_are_never_merged(hit):
    """One row id in both tables stays two results.

    Fusion keys on (source_type, row_id). Keying on row_id alone would merge
    them, and both are UUIDs from different tables with no constraint between
    them.
    """
    chunk = hit(1, 1)
    table = hit(1, 2, source_type="table")

    fused = fuse({"vector": [chunk, table]}, limit=10)

    # Two results, and the same-arm duplicate check did not fire either — it
    # keys on the same pair, so a key collapsed to the row id alone would turn
    # a legitimate chunk-and-table pair into a raise.
    assert [(f.source_type, f.row_id.int) for f in fused] == [
        ("chunk", 1),
        ("table", 1),
    ]


def test_empty_and_missing_arms(hit):
    """No arms, an empty arm, and a lone arm are all ordinary inputs.

    The last is not hypothetical: a question with no lexical signal yields a
    NULL tsquery and an empty text arm, and `retrieve` treats that as a normal
    ranking rather than an error.
    """
    assert fuse({}, limit=10) == []

    # An arm that found nothing contributes nothing — it must not appear in
    # `contributions`, where it would claim a rank it never gave.
    with_empty = fuse({"vector": [hit(1, 1)], "text": []}, limit=10)
    assert [f.row_id.int for f in with_empty] == [1]
    assert with_empty[0].contributions == {"vector": 1}

    # One arm still produces a ranking, numbered from 1, not a degenerate list
    # that the caller has to special-case.
    alone = fuse({"vector": [hit(1, 1), hit(2, 2)]}, limit=10)
    assert [f.rank for f in alone] == [1, 2]
    assert [f.row_id.int for f in alone] == [1, 2]


def test_a_duplicate_within_one_arm_raises(hit):
    """One arm ranking the same row twice is a bug, not something to absorb.

    A ranked list that places one row at two positions would be double-counted
    by the accumulate, quietly inflating that row above everything else.
    """
    with pytest.raises(ValueError) as excinfo:
        fuse({"vector": [hit(1, 1), hit(1, 2)]}, limit=10)

    # Naming the arm, because the caller's next question is which query
    # produced it. The same row across two arms is the normal agreement case
    # and must not raise; `test_contributions_record_every_arm` covers it.
    message = str(excinfo.value)
    assert "vector" in message
    assert "two positions" in message
