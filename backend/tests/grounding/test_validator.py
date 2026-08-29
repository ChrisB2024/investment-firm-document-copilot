"""Tests for app.grounding.validator.

Pure functions over an answer and a dict. No database, no model, no corpus.

This is the module that decides whether an answer reaches an analyst, so the
tests that matter are the ones where a plausible-looking answer must be
*rejected*. An implementation that accepts everything passes any test suite
built only from good answers.
"""

from __future__ import annotations

# Every test below drives `validate` and asserts on `GroundingError`, and most
# want `pytest.raises`. Imports are left to the implementation so the scaffold
# does not ship a file ruff rejects for names nothing uses yet.
#
# `passage`, `answer`, `retrieved` and `deps` come from tests/conftest.py. They
# sit at root scope because pytest resolves fixtures per directory and this
# package needs the same four as tests/assistant/.


def test_a_fabricated_handle_is_rejected(passage, answer):
    """A handle no tool minted resolves to nothing and fails.

    The architecture's headline guarantee, and the cheapest to get wrong: an
    implementation that skips unresolvable handles instead of failing on them
    produces an answer whose citations "all resolve" because the bad ones were
    quietly dropped.

    TODO: implement. Assert the raise, and assert the message names the handle
     — the ModelRetry built from it is the only thing telling the model which
     citation to fix.
    """
    raise NotImplementedError


def test_a_quote_that_is_not_in_the_passage_is_rejected(passage, answer):
    """A real handle with words the passage does not contain fails.

    The check that separates this module from a lookup table. "Cited a real
    passage that does not say this" is what a fluent model actually produces,
    and it is invisible to check 1 — the handle resolves perfectly.

    TODO: implement.
    """
    raise NotImplementedError


def test_a_quote_may_come_from_the_title_or_the_section(passage, answer):
    """The checkable surface is every field the model was shown, not `text`.

    Regression: `_supports` read `passage.text` alone while the tool handed the
    model `title` and `section` too. A table's markdown is the grid by itself,
    so "Unconditional Purchase Obligations" — the natural way to cite a
    schedule of figures — was quoting the passage and failing the check.

    TODO: implement. Build a table passage whose `title` holds the caption and
     whose `text` holds only the grid, and assert a quote of each validates.
    """
    raise NotImplementedError


def test_a_quote_cannot_span_two_fields(passage, answer):
    """Fields are checked whole, so a match cannot cross a seam that was prose.

    The counterweight to the test above. Joining title and text before
    searching would accept "Obligations (in millions) | 2026" — a span the
    passage never contained, assembled out of two fields that are adjacent only
    because we put them next to each other.

    TODO: implement.
    """
    raise NotImplementedError


def test_curly_quotes_fold_but_case_does_not(passage, answer):
    """The one normalisation, and its boundary.

    54% of chunks contain U+2019 and 34% contain U+201C/D, so a model that
    straightens a quotation mark while copying correctly must not be rejected.
    Case, spelling and word choice must still be, because those are what the
    citation attests to — and a test that only pins the folding would pass an
    implementation that lowercased everything.

    TODO: implement both directions in one test; they are one decision.
    """
    raise NotImplementedError


def test_an_empty_quote_never_matches(passage, answer):
    """"" is a substring of every passage, so it has to be special-cased.

    Whitespace-only too, since it folds to "". This is the one degenerate quote
    that would otherwise validate against the entire corpus.

    TODO: implement.
    """
    raise NotImplementedError


def test_an_answer_with_no_citations_is_rejected(answer):
    """Asserting something with an empty citation list is the core failure.

    `InsufficientEvidence` was available and the model asserted instead. Worth
    its own test because it is the case an implementation gets wrong by
    omission: a loop over `citations` finds nothing to complain about and the
    answer sails through uncited.

    TODO: implement. Assert the violation is the whole-answer one rather than a
     per-handle one, so the model is told what it did rather than handed a list
     of dangling markers saying the same thing one at a time.
    """
    raise NotImplementedError


def test_markers_and_citations_must_agree_in_both_directions(passage, answer):
    """A marker with no citation, and a citation with no marker.

    Different failures needing different messages: the first is a reference an
    analyst clicks and nothing happens, the second is evidence attached to no
    claim, which reads as corroboration and is not.

    TODO: implement both, and assert the two messages differ. A single shared
     "markers and citations disagree" message would pass a weaker test and tell
     the model nothing about which way to fix it.
    """
    raise NotImplementedError


def test_a_grouped_marker_counts_as_several(passage, answer):
    """[S3, S4] is two marked handles, not zero.

    Regression, and a nasty one: matching only [S3] reported *every* citation in
    the answer as unmarked while the prose visibly marked them all — the most
    confusing retry this module can send. Pin the negative too: "[Table 3]" is
    not a citation, or ordinary bracketed prose starts failing answers.

    TODO: implement.
    """
    raise NotImplementedError


def test_every_violation_is_reported_at_once(passage, answer):
    """Three bad citations produce three violations, not the first one.

    Each round trip is a full turn's tokens against a retry budget of 1, so an
    implementation that raises on the first violation gives the model one
    correction for three problems and then fails the run.

    TODO: implement.
    """
    raise NotImplementedError


def test_a_valid_answer_returns_its_passages_in_ledger_order(passage, answer):
    """The success path: deduplicated, ledger order, not citation order.

    Ledger order is retrieval order, which for a fan-out or a grid is
    deliberately balanced across companies; citation order is whatever the model
    happened to write. Build the answer citing handles out of order so the two
    differ, or the test cannot tell them apart.

    One passage cited twice — legitimate, two claims on one passage — must
    appear once here.

    TODO: implement.
    """
    raise NotImplementedError


def test_a_stale_handle_from_a_previous_turn_is_rejected(passage, answer):
    """Turn-scoping, tested where it is actually enforced.

    There is no "previous turn" inside `validate`; the guarantee comes from
    `DocumentAgentDeps` being built per turn, so a stale handle is simply one
    the ledger does not hold. That makes this test identical in mechanism to the
    fabricated-handle case — which is the point worth recording, so nobody
    later adds a timestamp field to make it feel more thorough.

    TODO: implement as a ledger from "turn 1" and an answer citing a handle
     minted against a different deps instance, asserting it fails.
    """
    raise NotImplementedError
