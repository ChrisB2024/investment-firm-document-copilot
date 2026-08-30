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

A tool decorated with `@agent.tool` is still the plain function, so the fast
tests call it with a `RunContext` they build themselves. `_context` supplies the
one field any of this reads — `deps` — and a `TestModel` that is never asked for
a completion.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic_ai import ModelRetry, RunContext, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from sqlalchemy import select

from app.assistant import agent as agent_module
from app.assistant.agent import (
    CORPUS_TICKERS,
    CORPUS_YEARS,
    DEFAULT_SEARCH_LIMIT,
    GRID_TARGET_PASSAGES,
    _per_cell,
    agent,
    enforce_grounding,
    read_surrounding_chunks,
    search_filings,
)
from app.assistant.deps import MAX_PASSAGES_PER_TURN, DocumentAgentDeps
from app.assistant.outputs import (
    GroundedAnswer,
    InsufficientEvidence,
    SourcePassage,
)
from app.config import settings
from app.database.models import SourceDocument
from app.grounding.validator import validate
from app.retrieval.queries import NO_FILTERS, Filters

QUESTION = "how did supplier concentration risk change"

# The floor `_supports` could safely adopt. Measured across the ten brief
# questions: 117 citations, shortest 19 characters, median 208. Sixteen leaves
# a little room under that without admitting a two-word quote, which attests to
# almost nothing and matches almost any passage.
SAFE_QUOTE_FLOOR = 16


def _context(deps: DocumentAgentDeps) -> RunContext[DocumentAgentDeps]:
    """A RunContext carrying `deps`, which is all a tool or the gate reads.

    The model is a `TestModel` rather than the module's real one: nothing here
    completes anything, and a context built around the live client would make a
    mistake in these tests cost an API call instead of failing.
    """
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


async def test_the_arguments_choose_the_retriever(monkeypatch, deps, retrieved):
    """tickers+years -> grid, tickers -> fan-out, years -> filtered, neither -> plain.

    The one piece of retrieval policy this module owns. Getting it wrong is
    silent: every branch returns passages, and a comparative question answered
    from a flat top-10 reads fine while missing three of the five companies it
    names — the exact failure Phase 3 built the grid to fix.
    """
    calls: list[tuple[str, tuple, dict]] = []

    def _stub(name: str):
        async def _retriever(*args, **kwargs):
            calls.append((name, args, kwargs))
            # A distinct row per call, so the ledger records four passages
            # rather than re-offering one under the handle it already has.
            return [retrieved(len(calls))]

        return _retriever

    monkeypatch.setattr(agent_module, "retrieve_grid", _stub("grid"))
    monkeypatch.setattr(agent_module, "retrieve_per_ticker", _stub("fan-out"))
    monkeypatch.setattr(agent_module, "retrieve", _stub("retrieve"))

    ctx = _context(deps)

    await search_filings(ctx, QUESTION, tickers=["AAPL", "MSFT"], years=[2021, 2022])
    name, _, kwargs = calls[-1]
    assert name == "grid"
    assert kwargs["tickers"] == ["AAPL", "MSFT"]
    assert kwargs["years"] == [2021, 2022]
    # Four cells share the grid's budget rather than inheriting a 25-cell one.
    assert kwargs["per_cell"] == _per_cell(4)

    await search_filings(ctx, QUESTION, tickers=["AAPL"])
    name, args, _ = calls[-1]
    assert name == "fan-out"
    assert args[2] == ["AAPL"]

    await search_filings(ctx, QUESTION, years=[2021, 2025])
    name, _, kwargs = calls[-1]
    assert name == "retrieve"
    # A discrete list widens to the closed range `Filters` expresses, so the
    # three years nobody named are searched too.
    assert kwargs["filters"] == Filters(fiscal_year_from=2021, fiscal_year_to=2025)
    assert kwargs["limit"] == DEFAULT_SEARCH_LIMIT

    await search_filings(ctx, QUESTION)
    name, _, kwargs = calls[-1]
    assert name == "retrieve"
    assert kwargs.get("filters", NO_FILTERS).matches_everything

    # Every branch's results reached the ledger, under handles that count up.
    assert list(deps.ledger) == ["S1", "S2", "S3", "S4"]


def test_grid_depth_scales_with_the_grid():
    """`_per_cell` keeps a small grid from inheriting a big grid's budget.

    Regression, found by running the CLI rather than by reading: with a fixed
    per_cell of 1, "how did Apple describe supplier risk in its latest 10-K"
    routed to a 1x1 grid and came back with a single passage, where `retrieve`
    would have given ten. Pin the shape, not one value — 1 cell caps at
    DEFAULT_SEARCH_LIMIT, 25 cells floors at 1, and the product stays near
    GRID_TARGET_PASSAGES in between.
    """
    # A one-cell grid is `retrieve` with a filter and should return what
    # `retrieve` returns.
    assert _per_cell(1) == DEFAULT_SEARCH_LIMIT
    # The 5x5 the brief's comparative questions have: a passage per cell, which
    # is the coverage the grid exists for.
    assert _per_cell(GRID_TARGET_PASSAGES) == 1
    # And a grid past the budget still gets one, rather than none.
    assert _per_cell(GRID_TARGET_PASSAGES * 2) == 1

    for cells in (1, 2, 3, 4, 5, 6, 9, 13, 20, 25, 50):
        depth = _per_cell(cells)
        assert 1 <= depth <= DEFAULT_SEARCH_LIMIT
        if cells <= GRID_TARGET_PASSAGES:
            assert cells * depth <= GRID_TARGET_PASSAGES
        if 3 <= cells <= GRID_TARGET_PASSAGES:
            # Near the budget, not a fraction of it: a 13-cell grid returns 13
            # passages, not the 2 a fixed depth would give it.
            assert cells * depth >= GRID_TARGET_PASSAGES // 2


async def test_a_company_outside_the_corpus_is_refused_by_name(monkeypatch, deps):
    """"TSLA" raises ModelRetry naming the five, rather than returning nothing.

    The subtle one. An unknown ticker otherwise retrieves zero rows, and zero
    rows is indistinguishable from a company the corpus does not discuss — the
    model reports "not covered", which is true, phrased as a refusal that reads
    exactly like an honest one. Assert the message names the five, so a mixed
    question comes back answered for Apple and declined for Tesla.
    """
    ctx = _context(deps)

    with pytest.raises(ModelRetry) as refused:
        await search_filings(ctx, QUESTION, tickers=["AAPL", "TSLA"])
    message = str(refused.value)
    assert "TSLA" in message
    assert all(ticker in message for ticker in CORPUS_TICKERS)

    with pytest.raises(ModelRetry) as refused:
        await search_filings(ctx, QUESTION, years=[2019, 2024])
    message = str(refused.value)
    assert "2019" in message
    assert "2021" in message and "2025" in message

    # Normalised, not refused. Both spellings are one arm, or a model writing
    # "aapl" opens a second, half-filled search of the same company.
    asked: list[list[str]] = []

    async def _fan_out(session, question, tickers, **kwargs):
        asked.append(list(tickers))
        return []

    monkeypatch.setattr(agent_module, "retrieve_per_ticker", _fan_out)
    assert await search_filings(ctx, QUESTION, tickers=["aapl", " AAPL "]) == []
    assert asked == [["AAPL"]]


async def test_an_exhausted_budget_becomes_a_retry_not_a_crash(
    monkeypatch, deps, retrieved
):
    """PassageBudgetExceeded reaches the model as advice it can act on.

    This fires on 3 of the 10 brief questions, so the handler is a normal path.
    Assert the ModelRetry says both things: narrow the search, *or* answer from
    what you already have. Measured, the model needed three attempts to take the
    second option on question 6.
    """
    deps.offer([retrieved(n) for n in range(1, MAX_PASSAGES_PER_TURN)])

    async def _two_more(*args, **kwargs):
        return [retrieved(MAX_PASSAGES_PER_TURN + n) for n in range(2)]

    monkeypatch.setattr(agent_module, "retrieve", _two_more)

    with pytest.raises(ModelRetry) as retry:
        await search_filings(_context(deps), QUESTION)

    message = str(retry.value)
    assert str(MAX_PASSAGES_PER_TURN) in message
    # Both exits. Narrowing is not always available — question 6 needs all five
    # companies — and a retry offering only that leaves the model stuck.
    assert "fewer" in message
    assert "answer from the passages you already have" in message

    # Raised rather than truncated, and nothing half-written: a silently
    # shortened grid is a confident answer over evidence nobody was told about.
    assert len(deps.ledger) == MAX_PASSAGES_PER_TURN - 1


async def test_widening_an_unknown_handle_is_a_retry(deps):
    """A handle the turn never minted must not come back silently absent.

    `read_surrounding_chunks` omits a handle with nothing to add, so a silent
    omission for an unknown handle is indistinguishable from "that passage has
    no context" — and the model would cite it as-is, having been told nothing.
    """
    # An empty ledger: nothing in this turn minted S7.
    with pytest.raises(ModelRetry) as retry:
        await read_surrounding_chunks(_context(deps), ["S7"])

    message = str(retry.value)
    assert "S7" in message
    assert "search again" in message.lower()


async def test_widening_replaces_the_ledger_text(monkeypatch, deps, passage):
    """The window becomes the passage, so a quote from it validates.

    If the ledger kept the narrow text, a model quoting the widened window would
    fail validation on every such citation — rejected for being more careful,
    the worst way to lose an answer. Assert a table handle comes back omitted
    rather than widened, too.
    """
    narrow = passage("S1", "In addition, the Company relies on single sources.", n=1)
    table = passage("S2", "| year | amount |", source_type="table", n=2)
    alone = passage("S3", "A passage with no neighbour in its section.", n=3)
    deps.ledger.update({"S1": narrow, "S2": table, "S3": alone})

    window = f"Item 1A. Risk Factors\n\nSupply is concentrated.\n\n{narrow.text}"
    asked: list[SourcePassage] = []

    async def _neighbours(session, passages, **kwargs):
        asked.extend(passages)
        # Tables are never keyed — they are whole by construction — and a lone
        # anchor rebuilds its own text exactly.
        return {
            ("chunk", narrow.row_id): window,
            ("chunk", alone.row_id): alone.text,
        }

    monkeypatch.setattr(agent_module, "neighbours", _neighbours)

    widened = await read_surrounding_chunks(_context(deps), ["S1", "S2", "S3"])

    # Only the handle that gained something. Returning the other two spends
    # tokens telling the model what it already holds.
    assert widened == {"S1": window}
    assert deps.ledger["S1"].text == window
    # Identity survives the swap, or the citation points at a different row.
    assert deps.ledger["S1"].handle == "S1"
    assert deps.ledger["S1"].row_id == narrow.row_id
    assert deps.ledger["S2"].text == table.text
    assert deps.ledger["S3"].text == alone.text

    assert [p.handle for p in asked] == ["S1", "S2", "S3"]


async def test_the_gate_passes_a_refusal_untouched(deps):
    """`InsufficientEvidence` has nothing to ground.

    It makes no claim and cites nothing, so running the citation rules over it
    could only invent a violation.
    """
    refusal = InsufficientEvidence(
        reason="The corpus holds no filings for the company asked about.",
        searched=["battery cell supply agreements"],
    )

    # Against an empty ledger, where a GroundedAnswer could not pass.
    assert await enforce_grounding(_context(deps), refusal) is refusal


async def test_the_gate_turns_a_violation_into_a_retry(deps, passage, answer):
    """A GroundingError reaches the model as a ModelRetry, not an exception.

    Assert the message carries the per-handle violations *and* the sentence that
    does the work: remove the claim, not the citation. A model told its quote
    does not match will otherwise take the cheapest path to a passing answer —
    delete the citation, leave the sentence — which is an uncited claim waved
    through by the check meant to stop it.
    """
    deps.ledger["S1"] = passage("S1", "Services revenue grew to $96.2 billion.")

    bad = answer(
        ("S1", "Services revenue grew to $97.4 billion."),
        ("S9", "The Company relies on a single supplier."),
    )

    with pytest.raises(ModelRetry) as retry:
        await enforce_grounding(_context(deps), bad)

    message = str(retry.value)
    # One line per violation, each naming its handle: a model told only that
    # the answer failed has nothing specific to fix.
    assert "- S1:" in message
    assert "- S9:" in message
    # The load-bearing sentence, and the honest alternative to it.
    assert "remove the claim" in message
    assert "InsufficientEvidence" in message


async def test_the_output_retry_budget_is_one_correction_then_failure(deps):
    """A rejected answer gets exactly one more attempt, then the run fails.

    `retries={"output": 1}` carries the longest comment in agent.py and nothing
    else pins it. The argument is that a wider budget mostly buys more attempts
    at the *worst* outcome: a model told its quote does not match sometimes
    fixes it, and sometimes deletes the citation and keeps the claim, which
    passes the validator and ships an uncited assertion. Raising this to 3
    silently triples the chances of that, and no other test would notice.

    Driven through the real agent rather than a stand-in, so it pins the
    configured value and not a copy of it. The model answers identically every
    time and cites a handle no tool minted, so the gate rejects it on both
    passes and the run has to give up.
    """
    calls: list[int] = []

    def _cite_a_fabricated_handle(messages, info: AgentInfo) -> ModelResponse:
        calls.append(len(calls))
        tool = next(t for t in info.output_tools if "Grounded" in t.name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool.name,
                    {
                        "answer": "A claim [S9].",
                        "citations": [{"handle": "S9", "quote": "never retrieved"}],
                    },
                )
            ]
        )

    with (
        agent.override(model=FunctionModel(_cite_a_fabricated_handle)),
        pytest.raises(UnexpectedModelBehavior) as failure,
    ):
        await agent.run(QUESTION, deps=deps)

    # The initial answer plus one correction. Three would mean the budget moved.
    assert len(calls) == 2
    # The controlled failure Phase 5 has to render rather than 500 on.
    assert "retries" in str(failure.value)


# --- integration: what only a real model can answer -------------------------


@pytest.mark.integration
async def test_the_hardcoded_corpus_matches_the_ingested_one(corpus):
    """CORPUS_TICKERS and CORPUS_YEARS are a copy, so check them against reality.

    agent.py names the five and the year span rather than querying per tool
    call, and its own comment says where that lies: a half-ingested corpus makes
    the guard permissive, so NVDA is accepted, returns nothing, and the model
    reports "not covered" — the failure the guard exists to prevent, wearing its
    costume. Only a query can catch that, and it is one query.

    Also the direction nobody thinks about: ingesting a sixth company without
    editing the constant makes it *restrictive*, and every question about that
    company is refused by name while its filings sit in the corpus.
    """
    rows = (
        await corpus.execute(
            select(SourceDocument.ticker, SourceDocument.fiscal_year).distinct()
        )
    ).all()

    assert {row.ticker for row in rows} == set(CORPUS_TICKERS)
    assert {row.fiscal_year for row in rows} == set(CORPUS_YEARS)


@pytest.mark.integration
async def test_a_question_the_corpus_cannot_answer_is_refused(corpus):
    """The InsufficientEvidence path, which the brief never exercises.

    Measured: all ten brief questions returned GroundedAnswer, so nothing in the
    exit criterion covers the refusal branch. Question 10 was expected to refuse
    and correctly did not — it asks for the evidence *and* the boundary, and the
    model answered with citations plus `limitations`.

    So the refusal needs a question built for it: a company outside the corpus,
    named in the prose rather than as a ticker argument, so it reaches the model
    as a retrieval result of nothing rather than as the ticker guard's retry.
    """
    deps = DocumentAgentDeps(session=corpus, user_id=uuid4(), thread_id=uuid4())

    result = await agent.run(
        "What does Tesla's 10-K say about battery cell supply agreements?",
        deps=deps,
        usage_limits=UsageLimits(request_limit=settings.openai_agent_request_limit),
    )

    assert isinstance(result.output, InsufficientEvidence)
    # Names what is missing, rather than "I could not find anything" — the
    # difference between a refusal an analyst can act on and one they cannot.
    assert "tesla" in result.output.reason.lower()
    assert result.output.searched


@pytest.mark.integration
async def test_every_quote_the_model_writes_clears_the_supports_floor(corpus):
    """Guards the floor `_supports` does not have yet.

    Measured across the ten brief questions: 117 citations, shortest quote 19
    characters, median 208. `_supports` has no minimum length because every
    floor also rejects a legitimate short quote — "$96.2 billion" is two words —
    and 19 is the number that says where one could safely go.

    Asserts the property rather than the number: every quote in a real run
    validates. A floor set above what the model actually writes fails inside the
    gate, so the run raises before this reaches its assertions.
    """
    deps = DocumentAgentDeps(session=corpus, user_id=uuid4(), thread_id=uuid4())

    result = await agent.run(
        "How did Apple describe supplier concentration risk in its latest 10-K?",
        deps=deps,
        usage_limits=UsageLimits(request_limit=settings.openai_agent_request_limit),
    )

    assert isinstance(result.output, GroundedAnswer)
    assert result.output.citations

    # The same function that admitted the answer, run again over the same
    # ledger: every quote resolves to a passage and appears in it verbatim.
    validated = validate(result.output, deps.ledger)
    assert {p.handle for p in validated.cited_passages} == {
        citation.handle for citation in result.output.citations
    }
    assert all(citation.quote.strip() for citation in result.output.citations)

    # The tripwire. Everything above holds by construction once the gate has
    # passed — it ran `validate` over this same ledger before returning — so it
    # can only catch a floor that is already too high, via the run raising.
    # What it cannot catch is drift *before* a floor exists: if the model starts
    # writing 12-character quotes this stays green, and adding the floor later
    # breaks production instead of this test.
    shortest = min(len(citation.quote) for citation in result.output.citations)
    assert shortest >= SAFE_QUOTE_FLOOR, (
        f"shortest quote is {shortest} chars; a floor at {SAFE_QUOTE_FLOOR} "
        f"would now reject an answer this model actually writes"
    )
