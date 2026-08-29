"""Tests for app.assistant.agent — dispatch, guards, and the grounding gate.

Two layers, and the split is deliberate.

The fast tests drive the tools directly with a stubbed session and monkeypatched
retrievers. What they pin is this module's own logic: which retriever a set of
arguments selects, what depth a grid gets, which mistakes become a ModelRetry,
and whether the gate lets an answer through. None of that needs a model, and a
test that ran one would be slower and would fail for reasons unrelated to the
code under test.

The integration tests need the thing nothing can fake: what gpt-5.5 actually
does. Both are questions the ten brief questions left open — see the module
docstring on each.

Imports are left to the implementation: `pytest`, `ModelRetry` and `RunContext`
from pydantic_ai, the tool functions and `_per_cell` from app.assistant.agent,
and the output models.
"""

from __future__ import annotations


def test_the_arguments_choose_the_retriever(monkeypatch, deps, retrieved):
    """tickers+years -> grid, tickers -> fan-out, years -> filtered, neither -> plain.

    The one piece of retrieval policy this module owns. Getting it wrong is
    silent: every branch returns passages, and a comparative question answered
    from a flat top-10 reads fine while missing three of the five companies it
    names — the exact failure Phase 3 built the grid to fix.

    TODO: implement by monkeypatching the three retrievers to record their call
     and return a passage each. Assert on which was called *and* on what it was
     handed: the years-only branch widens a discrete list into a closed range,
     so [2021, 2025] must arrive as Filters(2021, 2025).
    """
    raise NotImplementedError


def test_grid_depth_scales_with_the_grid(deps):
    """`_per_cell` keeps a small grid from inheriting a big grid's budget.

    Regression, found by running the CLI rather than by reading: with a fixed
    per_cell of 1, "how did Apple describe supplier risk in its latest 10-K"
    routed to a 1x1 grid and came back with a single passage, where `retrieve`
    would have given ten. Pin the shape, not one value — 1 cell caps at
    DEFAULT_SEARCH_LIMIT, 25 cells floors at 1, and the product stays near
    GRID_TARGET_PASSAGES in between.

    TODO: implement as a table over cell counts.
    """
    raise NotImplementedError


def test_a_company_outside_the_corpus_is_refused_by_name(deps):
    """"TSLA" raises ModelRetry naming the five, rather than returning nothing.

    The subtle one. An unknown ticker otherwise retrieves zero rows, and zero
    rows is indistinguishable from a company the corpus does not discuss — the
    model reports "not covered", which is true, phrased as a refusal that reads
    exactly like an honest one. Assert the message names the five, so a mixed
    question comes back answered for Apple and declined for Tesla.

    TODO: implement. Cover years outside 2021-2025 the same way, and assert
     lowercase "aapl" is accepted rather than refused — it normalises.
    """
    raise NotImplementedError


def test_an_exhausted_budget_becomes_a_retry_not_a_crash(monkeypatch, deps, retrieved):
    """PassageBudgetExceeded reaches the model as advice it can act on.

    This fires on 3 of the 10 brief questions, so the handler is a normal path.
    Assert the ModelRetry says both things: narrow the search, *or* answer from
    what you already have. Measured, the model needed three attempts to take the
    second option on question 6.

    TODO: implement.
    """
    raise NotImplementedError


def test_widening_an_unknown_handle_is_a_retry(deps):
    """A handle the turn never minted must not come back silently absent.

    `read_surrounding_chunks` omits a handle with nothing to add, so a silent
    omission for an unknown handle is indistinguishable from "that passage has
    no context" — and the model would cite it as-is, having been told nothing.

    TODO: implement.
    """
    raise NotImplementedError


def test_widening_replaces_the_ledger_text(monkeypatch, deps, passage):
    """The window becomes the passage, so a quote from it validates.

    If the ledger kept the narrow text, a model quoting the widened window would
    fail validation on every such citation — rejected for being more careful,
    the worst way to lose an answer. Assert a table handle comes back omitted
    rather than widened, too.

    TODO: implement by monkeypatching `neighbours`.
    """
    raise NotImplementedError


def test_the_gate_passes_a_refusal_untouched(deps):
    """`InsufficientEvidence` has nothing to ground.

    It makes no claim and cites nothing, so running the citation rules over it
    could only invent a violation.

    TODO: implement — call `enforce_grounding` with a RunContext carrying deps.
    """
    raise NotImplementedError


def test_the_gate_turns_a_violation_into_a_retry(deps, passage, answer):
    """A GroundingError reaches the model as a ModelRetry, not an exception.

    Assert the message carries the per-handle violations *and* the sentence that
    does the work: remove the claim, not the citation. A model told its quote
    does not match will otherwise take the cheapest path to a passing answer —
    delete the citation, leave the sentence — which is an uncited claim waved
    through by the check meant to stop it.

    TODO: implement.
    """
    raise NotImplementedError


# --- integration: what only a real model can answer -------------------------


def test_a_question_the_corpus_cannot_answer_is_refused():
    """The InsufficientEvidence path, which the brief never exercises.

    Measured: all ten brief questions returned GroundedAnswer, so nothing in the
    exit criterion covers the refusal branch. Question 10 was expected to refuse
    and correctly did not — it asks for the evidence *and* the boundary, and the
    model answered with citations plus `limitations`.

    So the refusal needs a question built for it. Ask about a company outside
    the corpus in a way that survives the ticker guard — a company name in the
    prose rather than a ticker argument, e.g. "what does Tesla's 10-K say about
    battery supply" — and assert the output is InsufficientEvidence with a
    `reason` that names what is missing.

    TODO: implement. Mark @pytest.mark.integration; needs a corpus session and
     OpenAI credentials.
    """
    raise NotImplementedError


def test_every_quote_the_model_writes_clears_the_supports_floor():
    """Guards the floor `_supports` does not have yet.

    Measured across the ten brief questions: 117 citations, shortest quote 19
    characters, median 208. `_supports` has no minimum length because every
    floor also rejects a legitimate short quote — "$96.2 billion" is two words —
    and 19 is the number that says where one could safely go.

    Assert the property rather than the number: every quote in a real run
    validates. That keeps this honest if a floor is added later, and it fails if
    a floor is set above what the model actually writes.

    TODO: implement. Mark @pytest.mark.integration; one question, not ten.
    """
    raise NotImplementedError
