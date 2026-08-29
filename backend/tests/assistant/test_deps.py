"""Tests for app.assistant.deps — the per-turn passage ledger.

Pure. The ledger is what makes "the model cannot cite what was not retrieved"
structural rather than a prompt instruction, so these tests are about identity
and accounting, not retrieval.

Imports are left to the implementation: `DocumentAgentDeps`,
`MAX_PASSAGES_PER_TURN`, `PassageBudgetExceeded` and `ALREADY_SHOWN` from
app.assistant.deps, plus `pytest`.
"""

from __future__ import annotations


def test_handles_count_up_across_calls(deps, retrieved):
    """Two searches mint S1..S2 then S3, not S1..S2 then S1.

    A counter reset per call — or derived from `len(ledger)` — makes the second
    call's S1 overwrite the first's, and every citation to the original S1 then
    resolves to a passage the model never read. The quote check is the only
    thing that would notice, and only sometimes.

    TODO: implement.
    """
    raise NotImplementedError


def test_the_same_row_never_gets_two_handles(deps, retrieved):
    """A passage found by two searches keeps its first handle.

    Two handles on one row let the model corroborate a claim with a single piece
    of evidence cited twice, which is the shape of a much stronger answer than
    the corpus supports.

    TODO: implement. Cover the same-call case too — a fan-out and a grid can
     both return one row more than once — and assert the ledger grew by one.
    """
    raise NotImplementedError


def test_a_chunk_and_a_table_with_one_row_id_stay_separate(deps, retrieved):
    """Identity is (source_type, row_id), the key `fuse` and `hydrate` use.

    Both tables are UUID-keyed with no constraint between them, so keying the
    ledger on row_id alone would collapse a chunk into a table and hand the
    model one passage where two were retrieved.

    TODO: implement.
    """
    raise NotImplementedError


def test_a_table_caption_reaches_the_model(deps, retrieved):
    """`title` survives the RetrievedPassage -> SourcePassage translation.

    Regression, and the reason `retrieved` is not built on `passage`. hydrate
    fetches the caption and `offer` dropped it, so "Unconditional Purchase
    Obligations" arrived as a column of years and unlabelled figures — while
    brief question 8 asks about purchase commitments. It also carries the scale:
    of 1,131 tables with a recorded `units`, the string appears in the title or
    the markdown for all of them.

    TODO: implement.
    """
    raise NotImplementedError


def test_a_re_offered_passage_returns_without_its_body(deps, retrieved):
    """The second offer returns the handle and metadata, not the text again.

    Tool returns accumulate in the message history, so a grid overlapping twenty
    known passages would repeat ~10,800 tokens the model already holds. Assert
    both halves: the returned text is ALREADY_SHOWN, *and* the ledger still
    holds the real text — the elision is a view, and a version that wrote it
    into the ledger would break every quote check against that passage.

    TODO: implement.
    """
    raise NotImplementedError


def test_a_widened_passage_is_what_comes_back(deps, retrieved):
    """`read_surrounding_chunks` writes the ledger, and the ledger is the truth.

    Simulate the widen by writing to `deps.ledger[handle]` directly, the way the
    tool does, then assert `resolve` returns the widened text. This is what
    makes quote checking work against the text the model was actually shown.

    TODO: implement.
    """
    raise NotImplementedError


def test_the_budget_raises_and_changes_nothing(deps, retrieved):
    """Over MAX_PASSAGES_PER_TURN raises, and the ledger is untouched.

    Both halves matter. Truncating instead would produce a confident answer
    built on a silently shortened grid; a raise that had already written half
    the passages would leave the turn holding evidence the model was never told
    about, and the handles would be minted against nothing it can see.

    Measured against the real brief: this fires on 3 of the 10 questions, so it
    is a normal path, not an edge case.

    TODO: implement. Assert the message names what was asked for against what
     was left — `search_filings` turns it into the ModelRetry verbatim.
    """
    raise NotImplementedError


def test_a_re_offer_costs_nothing_against_the_budget(deps, retrieved):
    """Only genuinely new passages count. Re-offering a known one is free.

    Otherwise a model that searches twice over overlapping ground exhausts the
    turn without having been shown anything new, and the retry telling it to
    narrow is advice it cannot act on.

    TODO: implement — fill the ledger to the ceiling, then re-offer a passage
     already in it and assert it does not raise.
    """
    raise NotImplementedError


def test_resolve_is_exact(deps, retrieved):
    """"[S1]" and "s1" resolve to nothing; None rather than a raise.

    Any leniency belongs in the validator, applied once next to the rule it
    bends, rather than spread across every caller. None is what lets the
    validator name the handle in a message the model can act on.

    TODO: implement.
    """
    raise NotImplementedError
