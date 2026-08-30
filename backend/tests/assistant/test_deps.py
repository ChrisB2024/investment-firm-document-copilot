"""Tests for app.assistant.deps — the per-turn passage ledger.

Pure. The ledger is what makes "the model cannot cite what was not retrieved"
structural rather than a prompt instruction, so these tests are about identity
and accounting, not retrieval.

`deps` and `retrieved` come from tests/conftest.py. `retrieved(n)` puts
`UUID(int=n)` on the row, so a test names identity by writing back the integer
it passed in.
"""

from __future__ import annotations

import pytest

from app.assistant.deps import (
    ALREADY_SHOWN,
    MAX_PASSAGES_PER_TURN,
    PassageBudgetExceeded,
)


def test_handles_count_up_across_calls(deps, retrieved):
    """Two searches mint S1..S2 then S3, not S1..S2 then S1.

    A counter reset per call — or derived from `len(ledger)` — makes the second
    call's S1 overwrite the first's, and every citation to the original S1 then
    resolves to a passage the model never read. The quote check is the only
    thing that would notice, and only sometimes.
    """
    first = deps.offer([retrieved(1), retrieved(2)])
    assert [p.handle for p in first] == ["S1", "S2"]

    second = deps.offer([retrieved(3)])
    assert [p.handle for p in second] == ["S3"]

    assert list(deps.ledger) == ["S1", "S2", "S3"]
    # The first call's passages still resolve to the rows they were minted for.
    assert [deps.resolve(f"S{n}").row_id.int for n in (1, 2, 3)] == [1, 2, 3]


def test_the_same_row_never_gets_two_handles(deps, retrieved):
    """A passage found by two searches keeps its first handle.

    Two handles on one row let the model corroborate a claim with a single piece
    of evidence cited twice, which is the shape of a much stronger answer than
    the corpus supports.
    """
    deps.offer([retrieved(1), retrieved(2)])

    again = deps.offer([retrieved(1)])
    assert [p.handle for p in again] == ["S1"]
    assert list(deps.ledger) == ["S1", "S2"]

    # Within one call too: a fan-out and a grid can both return one row more
    # than once, and the duplicate must collapse rather than mint a second
    # handle for the same evidence.
    one_call = deps.offer([retrieved(3), retrieved(3)])
    assert [p.handle for p in one_call] == ["S3"]
    assert list(deps.ledger) == ["S1", "S2", "S3"]


def test_a_chunk_and_a_table_with_one_row_id_stay_separate(deps, retrieved):
    """Identity is (source_type, row_id), the key `fuse` and `hydrate` use.

    Both tables are UUID-keyed with no constraint between them, so keying the
    ledger on row_id alone would collapse a chunk into a table and hand the
    model one passage where two were retrieved.
    """
    offered = deps.offer([retrieved(1), retrieved(1, source_type="table")])

    assert [p.handle for p in offered] == ["S1", "S2"]
    assert [p.source_type for p in offered] == ["chunk", "table"]
    assert deps.resolve("S1").row_id == deps.resolve("S2").row_id


def test_a_table_caption_reaches_the_model(deps, retrieved):
    """`title` survives the RetrievedPassage -> SourcePassage translation.

    Regression, and the reason `retrieved` is not built on `passage`. hydrate
    fetches the caption and `offer` dropped it, so "Unconditional Purchase
    Obligations" arrived as a column of years and unlabelled figures — while
    brief question 8 asks about purchase commitments. It also carries the scale:
    of 1,131 tables with a recorded `units`, the string appears in the title or
    the markdown for all of them.
    """
    caption = "Unconditional Purchase Obligations (in millions)"
    [table] = deps.offer(
        [retrieved(1, source_type="table", title=caption, text="| 2025 | 1,000 |")]
    )

    assert table.title == caption
    assert deps.resolve("S1").title == caption
    # Serialised, because "reaches the model" is what pydantic-ai sends rather
    # than what the object holds.
    assert table.model_dump()["title"] == caption

    # Chunks carry their heading in the text; `_HYDRATE` selects NULL here.
    [chunk] = deps.offer([retrieved(2)])
    assert chunk.title is None


def test_a_re_offered_passage_returns_without_its_body(deps, retrieved):
    """The second offer returns the handle and metadata, not the text again.

    Tool returns accumulate in the message history, so a grid overlapping twenty
    known passages would repeat ~10,800 tokens the model already holds. Assert
    both halves: the returned text is ALREADY_SHOWN, *and* the ledger still
    holds the real text — the elision is a view, and a version that wrote it
    into the ledger would break every quote check against that passage.
    """
    body = "Services revenue grew to $96.2 billion."
    [first] = deps.offer([retrieved(1, text=body)])
    assert first.text == body

    [again] = deps.offer([retrieved(1, text=body)])
    assert again.text == ALREADY_SHOWN
    # Everything identifying it still comes back, or an almost-empty result
    # reads as "nothing more was found" and the model stops looking.
    assert again.handle == "S1"
    assert (again.ticker, again.fiscal_year, again.row_id) == (
        first.ticker,
        first.fiscal_year,
        first.row_id,
    )

    assert deps.resolve("S1").text == body


def test_a_widened_passage_is_what_comes_back(deps, retrieved):
    """`read_surrounding_chunks` writes the ledger, and the ledger is the truth.

    Simulate the widen by writing to `deps.ledger[handle]` directly, the way the
    tool does, then assert `resolve` returns the widened text. This is what
    makes quote checking work against the text the model was actually shown.
    """
    [offered] = deps.offer([retrieved(1, text="In addition, the Company relies.")])
    window = f"Item 1A. Risk Factors\n\nSupply is concentrated.\n\n{offered.text}"

    deps.ledger["S1"] = offered.model_copy(update={"text": window})

    widened = deps.resolve("S1")
    assert widened.text == window
    # The swap is a wider view of the same row, not a different passage.
    assert widened.handle == "S1"
    assert widened.row_id == offered.row_id
    assert list(deps.ledger) == ["S1"]


def test_the_budget_raises_and_changes_nothing(deps, retrieved):
    """Over MAX_PASSAGES_PER_TURN raises, and the ledger is untouched.

    Both halves matter. Truncating instead would produce a confident answer
    built on a silently shortened grid; a raise that had already written half
    the passages would leave the turn holding evidence the model was never told
    about, and the handles would be minted against nothing it can see.

    Measured against the real brief: this fires on 3 of the 10 questions, so it
    is a normal path, not an edge case.
    """
    deps.offer([retrieved(n) for n in range(1, MAX_PASSAGES_PER_TURN)])
    assert len(deps.ledger) == MAX_PASSAGES_PER_TURN - 1

    with pytest.raises(PassageBudgetExceeded) as exceeded:
        deps.offer([retrieved(MAX_PASSAGES_PER_TURN + n) for n in range(3)])

    # `search_filings` puts this in front of the model verbatim, so it has to
    # name what was asked for against what was left.
    message = str(exceeded.value)
    assert str(MAX_PASSAGES_PER_TURN) in message
    assert "1 left" in message
    assert "room for 3" in message

    assert len(deps.ledger) == MAX_PASSAGES_PER_TURN - 1
    # No handle was minted against the batch that failed: the next passage to
    # be offered takes the seat, not the number after the three that did not.
    [next_offered] = deps.offer([retrieved(MAX_PASSAGES_PER_TURN)])
    assert next_offered.handle == f"S{MAX_PASSAGES_PER_TURN}"


def test_a_re_offer_costs_nothing_against_the_budget(deps, retrieved):
    """Only genuinely new passages count. Re-offering a known one is free.

    Otherwise a model that searches twice over overlapping ground exhausts the
    turn without having been shown anything new, and the retry telling it to
    narrow is advice it cannot act on.
    """
    full = [retrieved(n) for n in range(1, MAX_PASSAGES_PER_TURN + 1)]
    deps.offer(full)
    assert len(deps.ledger) == MAX_PASSAGES_PER_TURN

    overlapping = deps.offer(full[:10])
    assert [p.handle for p in overlapping] == [f"S{n}" for n in range(1, 11)]
    assert all(p.text == ALREADY_SHOWN for p in overlapping)
    assert len(deps.ledger) == MAX_PASSAGES_PER_TURN

    # One genuinely new passage on a full ledger still raises, or the ceiling
    # is not a ceiling.
    with pytest.raises(PassageBudgetExceeded):
        deps.offer([retrieved(MAX_PASSAGES_PER_TURN + 1)])


def test_resolve_is_exact(deps, retrieved):
    """"[S1]" and "s1" resolve to nothing; None rather than a raise.

    Any leniency belongs in the validator, applied once next to the rule it
    bends, rather than spread across every caller. None is what lets the
    validator name the handle in a message the model can act on.
    """
    deps.offer([retrieved(1)])

    assert deps.resolve("S1").row_id.int == 1
    assert deps.resolve("[S1]") is None
    assert deps.resolve("s1") is None
    assert deps.resolve("S1 ") is None
    # Never minted, and never a KeyError: the validator turns None into a
    # message naming the handle.
    assert deps.resolve("S9") is None
