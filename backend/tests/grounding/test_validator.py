"""Tests for app.grounding.validator.

Pure functions over an answer and a dict. No database, no model, no corpus.

This is the module that decides whether an answer reaches an analyst, so the
tests that matter are the ones where a plausible-looking answer must be
*rejected*. An implementation that accepts everything passes any test suite
built only from good answers.

`passage`, `answer`, `retrieved` and `deps` come from tests/conftest.py. They
sit at root scope because pytest resolves fixtures per directory and this
package needs the same four as tests/assistant/.
"""

from __future__ import annotations

import pytest

from app.assistant.deps import DocumentAgentDeps
from app.grounding.validator import GroundingError, validate


def test_a_fabricated_handle_is_rejected(passage, answer):
    """A handle no tool minted resolves to nothing and fails.

    The architecture's headline guarantee, and the cheapest to get wrong: an
    implementation that skips unresolvable handles instead of failing on them
    produces an answer whose citations "all resolve" because the bad ones were
    quietly dropped.
    """
    ledger = {"S1": passage("S1", "Services revenue grew to $96.2 billion.")}

    invented = answer(
        ("S1", "Services revenue grew to $96.2 billion."),
        ("S7", "The Company relies on a single supplier."),
    )

    with pytest.raises(GroundingError) as rejected:
        validate(invented, ledger)

    # The good citation is not the complaint; the invented one is, by name.
    # The ModelRetry built from this is the only thing telling the model which
    # of its citations to fix.
    [violation] = rejected.value.violations
    assert violation.handle == "S7"
    assert "S7" in str(rejected.value)


def test_a_quote_that_is_not_in_the_passage_is_rejected(passage, answer):
    """A real handle with words the passage does not contain fails.

    The check that separates this module from a lookup table. "Cited a real
    passage that does not say this" is what a fluent model actually produces,
    and it is invisible to check 1 — the handle resolves perfectly.
    """
    ledger = {"S1": passage("S1", "Services revenue grew to $96.2 billion in 2024.")}

    # The same handle, resolving perfectly, with one figure changed.
    validate(answer(("S1", "Services revenue grew to $96.2 billion")), ledger)

    with pytest.raises(GroundingError) as rejected:
        validate(answer(("S1", "Services revenue grew to $97.4 billion")), ledger)

    [violation] = rejected.value.violations
    assert violation.handle == "S1"
    # The quote is echoed back, or two citations on one handle are
    # indistinguishable in the retry.
    assert "$97.4 billion" in violation.problem


def test_a_quote_from_another_passage_is_rejected(passage, answer):
    """A real quote under the wrong handle fails, though the words are in the ledger.

    The highest-consequence error this module catches, and the one a plausible
    refactor quietly removes: checking a quote against everything retrieved
    rather than against the passage cited. That refactor passes every other test
    in this file, because each of them either holds a single passage or uses a
    quote that appears in none — so nothing else pins *which* passage the words
    have to come from.

    In a five-company comparison, Microsoft's sentence under Apple's handle is a
    sourced-looking claim about Apple, and an analyst clicking through lands on a
    passage that really does say those words. Just not about that company.
    """
    ledger = {
        "S1": passage("S1", "The Company depends on single sources.", ticker="AAPL"),
        "S2": passage("S2", "We are scaling AI infrastructure.", n=2, ticker="MSFT"),
    }

    # Each quote is verbatim, and validates under its own handle.
    assert validate(answer(("S1", "The Company depends on single sources.")), ledger)
    assert validate(answer(("S2", "We are scaling AI infrastructure.")), ledger)

    # Swapped, it must not: the words are in the ledger, in the other passage.
    with pytest.raises(GroundingError) as rejected:
        validate(answer(("S1", "We are scaling AI infrastructure.")), ledger)
    [violation] = rejected.value.violations
    assert violation.handle == "S1"


def test_a_quote_may_come_from_the_title_or_the_section(passage, answer):
    """The checkable surface is every field the model was shown, not `text`.

    Regression: `_supports` read `passage.text` alone while the tool handed the
    model `title` and `section` too. A table's markdown is the grid by itself,
    so "Unconditional Purchase Obligations" — the natural way to cite a
    schedule of figures — was quoting the passage and failing the check.
    """
    table = passage(
        "S1",
        "| 2026 | 900 |\n| 2027 | 750 |",
        title="Unconditional Purchase Obligations (in millions)",
        source_type="table",
    )
    assert validate(answer(("S1", "Unconditional Purchase Obligations")), {"S1": table})
    assert validate(answer(("S1", "| 2027 | 750 |")), {"S1": table})

    # The same rule for `section`, which a citation UI renders as the Item the
    # claim came from.
    risk = passage("S2", "Supply is concentrated.", section="Item 1A. Risk Factors", n=2)
    assert validate(answer(("S2", "Item 1A. Risk Factors")), {"S2": risk})


def test_a_quote_cannot_span_two_fields(passage, answer):
    """Fields are checked whole, so a match cannot cross a seam that was prose.

    The counterweight to the test above. Joining title and text before
    searching would accept "Obligations (in millions) | 2026" — a span the
    passage never contained, assembled out of two fields that are adjacent only
    because we put them next to each other.
    """
    table = passage(
        "S1",
        "| 2026 | 900 |",
        title="Unconditional Purchase Obligations (in millions)",
        source_type="table",
    )

    with pytest.raises(GroundingError) as rejected:
        validate(answer(("S1", "Obligations (in millions) | 2026")), {"S1": table})

    [violation] = rejected.value.violations
    assert violation.handle == "S1"


def test_curly_quotes_fold_but_case_does_not(passage, answer):
    """The one normalisation, and its boundary.

    54% of chunks contain U+2019 and 34% contain U+201C/D, so a model that
    straightens a quotation mark while copying correctly must not be rejected.
    Case, spelling and word choice must still be, because those are what the
    citation attests to — and a test that only pins the folding would pass an
    implementation that lowercased everything.
    """
    curly = {"S1": passage("S1", "The Company’s “single source” suppliers.")}
    straight = {"S1": passage("S1", "The Company's \"single source\" suppliers.")}

    # Both directions of the fold, because the corpus holds both shapes and the
    # model writes whichever it writes.
    assert validate(answer(("S1", "The Company's \"single source\"")), curly)
    assert validate(answer(("S1", "The Company’s “single source”")), straight)

    # The whitespace fold this sits on top of: a line break in the passage is an
    # artefact of chunking, not a word.
    wrapped = {"S1": passage("S1", "The Company's\nsingle source\nsuppliers.")}
    assert validate(answer(("S1", "The Company's single source suppliers.")), wrapped)

    # And the boundary. Case is not a shape of an apostrophe.
    with pytest.raises(GroundingError):
        validate(answer(("S1", "the company's \"single source\"")), straight)
    # Nor is a word the passage does not use.
    with pytest.raises(GroundingError):
        validate(answer(("S1", "The Company's \"sole source\"")), straight)


def test_an_empty_quote_never_matches(passage, answer):
    """"" is a substring of every passage, so it has to be special-cased.

    Whitespace-only too, since it folds to "". This is the one degenerate quote
    that would otherwise validate against the entire corpus.
    """
    ledger = {"S1": passage("S1", "Services revenue grew to $96.2 billion.")}

    for quote in ("", "   ", "\n\t"):
        with pytest.raises(GroundingError) as rejected:
            validate(answer(("S1", quote)), ledger)
        [violation] = rejected.value.violations
        assert violation.handle == "S1"


def test_an_answer_with_no_citations_is_rejected(answer):
    """Asserting something with an empty citation list is the core failure.

    `InsufficientEvidence` was available and the model asserted instead. Worth
    its own test because it is the case an implementation gets wrong by
    omission: a loop over `citations` finds nothing to complain about and the
    answer sails through uncited.
    """
    uncited = answer(prose="Services revenue grew to $96.2 billion.")

    with pytest.raises(GroundingError) as rejected:
        validate(uncited, {})

    # One violation about the answer, not a per-handle one: the model is told
    # what it did rather than handed a list of dangling markers saying the same
    # thing one at a time.
    [violation] = rejected.value.violations
    assert violation.handle == "answer"
    assert "InsufficientEvidence" in violation.problem

    # Still one violation when the prose marks handles, and it names them —
    # "you marked S3 and cited nothing" is a different mistake to hear about
    # than "you cited nothing".
    marked = answer(prose="Services revenue grew [S3].")
    with pytest.raises(GroundingError) as rejected:
        validate(marked, {})
    [violation] = rejected.value.violations
    assert violation.handle == "answer"
    assert "S3" in violation.problem


def test_markers_and_citations_must_agree_in_both_directions(passage, answer):
    """A marker with no citation, and a citation with no marker.

    Different failures needing different messages: the first is a reference an
    analyst clicks and nothing happens, the second is evidence attached to no
    claim, which reads as corroboration and is not.
    """
    ledger = {
        "S1": passage("S1", "Services revenue grew."),
        "S2": passage("S2", "Operating costs fell.", n=2),
    }

    marked_but_uncited = answer(
        ("S1", "Services revenue grew."),
        prose="Revenue grew [S1] while costs fell [S2].",
    )
    with pytest.raises(GroundingError) as rejected:
        validate(marked_but_uncited, ledger)
    [dangling] = rejected.value.violations
    assert dangling.handle == "S2"

    cited_but_unmarked = answer(
        ("S1", "Services revenue grew."),
        ("S2", "Operating costs fell."),
        prose="Revenue grew [S1].",
    )
    with pytest.raises(GroundingError) as rejected:
        validate(cited_but_unmarked, ledger)
    [unmarked] = rejected.value.violations
    assert unmarked.handle == "S2"

    # Same handle, opposite mistakes. One shared "markers and citations
    # disagree" message would pass everything above and tell the model nothing
    # about which way to fix it.
    assert dangling.problem != unmarked.problem


def test_a_grouped_marker_counts_as_several(passage, answer):
    """[S3, S4] is two marked handles, not zero.

    Regression, and a nasty one: matching only [S3] reported *every* citation in
    the answer as unmarked while the prose visibly marked them all — the most
    confusing retry this module can send. Pin the negative too: "[Table 3]" is
    not a citation, or ordinary bracketed prose starts failing answers.
    """
    ledger = {
        "S3": passage("S3", "Apple said so.", n=3),
        "S4": passage("S4", "Microsoft said so.", n=4),
    }

    grouped = answer(
        ("S3", "Apple said so."),
        ("S4", "Microsoft said so."),
        prose="Both filings describe it the same way [S3, S4].",
    )
    validated = validate(grouped, ledger)
    assert [p.handle for p in validated.cited_passages] == ["S3", "S4"]

    # Every handle in the group is read, not just the first: S4 marked and
    # never cited is still a dangling reference.
    half_cited = answer(
        ("S3", "Apple said so."),
        prose="Both filings describe it the same way [S3, S4].",
    )
    with pytest.raises(GroundingError) as rejected:
        validate(half_cited, ledger)
    [violation] = rejected.value.violations
    assert violation.handle == "S4"

    # The negative. A bracketed aside is prose, not a marker.
    aside = answer(
        ("S3", "Apple said so."),
        prose="See the schedule [Table 3] for the figures [S3].",
    )
    assert validate(aside, ledger)


def test_every_violation_is_reported_at_once(passage, answer):
    """Three bad citations produce three violations, not the first one.

    Each round trip is a full turn's tokens against a retry budget of 1, so an
    implementation that raises on the first violation gives the model one
    correction for three problems and then fails the run.
    """
    ledger = {
        "S1": passage("S1", "Services revenue grew to $96.2 billion."),
        "S2": passage("S2", "Operating costs fell.", n=2),
    }

    # One of each rule, so this fails an implementation that collects within a
    # loop but raises between them.
    bad = answer(
        ("S1", "Services revenue grew to $97.4 billion."),
        ("S7", "The Company relies on a single supplier."),
        ("S2", "Operating costs fell."),
        prose="Revenue grew [S1] because of a supplier [S7].",
    )

    with pytest.raises(GroundingError) as rejected:
        validate(bad, ledger)

    violations = rejected.value.violations
    assert [v.handle for v in violations] == ["S1", "S7", "S2"]
    # The string form is what the API logs, and it carries all three too.
    assert len(str(rejected.value).splitlines()) == 3


def test_a_valid_answer_returns_its_passages_in_ledger_order(passage, answer):
    """The success path: deduplicated, ledger order, not citation order.

    Ledger order is retrieval order, which for a fan-out or a grid is
    deliberately balanced across companies; citation order is whatever the model
    happened to write. Build the answer citing handles out of order so the two
    differ, or the test cannot tell them apart.

    One passage cited twice — legitimate, two claims on one passage — must
    appear once here.
    """
    ledger = {
        "S1": passage("S1", "Apple describes supplier concentration.", ticker="AAPL"),
        "S2": passage("S2", "Microsoft describes the same risk.", n=2, ticker="MSFT"),
        "S3": passage("S3", "Nvidia describes it differently.", n=3, ticker="NVDA"),
    }

    good = answer(
        ("S3", "Nvidia describes it differently."),
        ("S1", "Apple describes supplier concentration."),
        ("S1", "supplier concentration"),
    )

    validated = validate(good, ledger)

    assert validated.answer == good
    # Ledger order, deduplicated, and S2 — offered but never cited — is absent.
    assert [p.handle for p in validated.cited_passages] == ["S1", "S3"]
    assert [p.ticker for p in validated.cited_passages] == ["AAPL", "NVDA"]


def test_a_stale_handle_from_a_previous_turn_is_rejected(deps, retrieved, answer):
    """Turn-scoping, tested where it is actually enforced.

    There is no "previous turn" inside `validate`; the guarantee comes from
    `DocumentAgentDeps` being built per turn, so a stale handle is simply one
    the ledger does not hold. That makes this test identical in mechanism to the
    fabricated-handle case — which is the point worth recording, so nobody
    later adds a timestamp field to make it feel more thorough.
    """
    first_turn = DocumentAgentDeps(
        session=None, user_id=deps.user_id, thread_id=deps.thread_id
    )
    first_turn.offer(
        [
            retrieved(1, text="Supply is concentrated in a single region."),
            retrieved(2, text="Costs rose in fiscal 2024."),
        ]
    )

    # The next turn on the same thread. Handles restart from S1, so the ledger
    # simply does not hold S2 any more.
    [current] = deps.offer([retrieved(3, text="Revenue grew in fiscal 2025.")])
    assert current.handle == "S1"

    stale = answer(("S2", "Costs rose in fiscal 2024."))
    with pytest.raises(GroundingError) as rejected:
        validate(stale, deps.ledger)
    [violation] = rejected.value.violations
    assert violation.handle == "S2"

    # And the sharper half: a stale handle that *does* exist this turn resolves
    # to a passage the model never read, and only the quote check catches it.
    reused = answer(("S1", "Supply is concentrated in a single region."))
    with pytest.raises(GroundingError) as rejected:
        validate(reused, deps.ledger)
    [violation] = rejected.value.violations
    assert violation.handle == "S1"
