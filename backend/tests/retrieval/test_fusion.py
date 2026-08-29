"""Tests for app.retrieval.fusion.

Pure functions over ranks. No database, no network, no corpus.
"""

from __future__ import annotations


def test_agreement_beats_a_single_strong_rank(hit):
    """TODO: a passage ranked 10th by *both* arms outranks one ranked 1st by one.

    2/(60+10) = 0.0286 against 1/(60+1) = 0.0164. This is the entire point of
    RRF, and it is the property an implementation gets backwards while still
    looking plausible — summing scores instead of ranks, or overwriting instead
    of accumulating, both pass a casual read.
    """
    raise NotImplementedError


def test_output_is_independent_of_arm_order(hit):
    """TODO: fuse the same arms with the dict built in both orders; assert the
    results are identical.

    Ties are the common case here, not a rare collision: where the arms do not
    overlap, every document scores 1/(k+rank) exactly once, so their rank-1
    entries tie, their rank-2 entries tie, and so on. Without a total order the
    ranking falls out of dict insertion order and changes with how the caller
    happened to build `arms`.
    """
    raise NotImplementedError


def test_ties_break_on_best_single_rank(hit):
    """TODO: two passages with equal RRF scores, one of which placed higher in
    some single arm — assert that one wins.

    Pin the rule, not just the determinism: sorting by score alone leaves these
    two in either order, and both orders are stable.
    """
    raise NotImplementedError


def test_contributions_record_every_arm(hit):
    """TODO: assert a passage found by both arms carries both ranks, and one
    found by a single arm carries only that one.

    This is the field that separates "not in the corpus" from "one arm found it
    and fusion buried it" — the two diagnoses the CLI exists to tell apart, and
    they need opposite fixes.
    """
    raise NotImplementedError


def test_a_chunk_and_a_table_are_never_merged(hit):
    """TODO: give a chunk and a table the same `row_id` and assert they stay
    two results.

    Fusion keys on (source_type, row_id). Keying on row_id alone would merge
    them, and both are UUIDs from different tables with no constraint between
    them.
    """
    raise NotImplementedError


def test_empty_and_missing_arms(hit):
    """TODO: assert `fuse({})` is empty, that an arm with no hits is harmless,
    and that a single arm still produces a valid ranking.

    The last is not hypothetical: a question with no lexical signal yields a
    NULL tsquery and an empty text arm, and `retrieve` treats that as a normal
    ranking rather than an error.
    """
    raise NotImplementedError


def test_a_duplicate_within_one_arm_raises(hit):
    """TODO: assert ValueError when one arm ranks the same row twice.

    A ranked list that places one row at two positions would be double-counted
    by the accumulate, quietly inflating that row above everything else.
    """
    raise NotImplementedError
