"""The LLM boundary: instructions, two tools, one typed output.

Below this module everything is deterministic and testable without a model —
retrieval, fusion, grounding. Above it is a chat turn. The seam owns no
retrieval policy of its own; it decides what the model may call and may return.

Two tools, where architecture.md names three; `read_chunk` is rejected at the
foot of the file. Passages are addressed by handle: the model is shown no row id
(outputs.py excludes them from serialisation) and no tool accepts one, so a
handle can only ever become a citation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.assistant.deps import DocumentAgentDeps, PassageBudgetExceeded
from app.assistant.outputs import (
    GroundedAnswer,
    InsufficientEvidence,
    SourcePassage,
)
from app.config import settings
from app.grounding.validator import GroundingError, validate
from app.retrieval.queries import Filters
from app.retrieval.retriever import (
    neighbours,
    retrieve,
    retrieve_grid,
    retrieve_per_ticker,
)

# Phase 3 measured 10 as the right depth for a one-company question: brief
# question 1 asks only about Apple and its top-10 is all Apple, all on topic.
DEFAULT_SEARCH_LIMIT = settings.retrieval_top_k

# Duplicated from instructions.md so a rejection and the prompt cannot disagree
# about what exists; querying it per tool call would be a request-path round trip
# for something that changes between ingestions, not between turns. A
# half-ingested corpus makes this lie permissively — NVDA accepted, nothing
# returned, "not covered" reported — which is the failure the check prevents.
CORPUS_TICKERS = frozenset({"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"})
CORPUS_YEARS = range(2021, 2026)

# Explicit, not the string "openai:gpt-5.5": that form has pydantic-ai read
# OPENAI_API_KEY from os.environ, and every env var here is declared on
# app.config.settings (../CLAUDE.md). Built at import like app/embeddings.py's
# client, so a missing key fails at startup, not on the first turn.
_model = OpenAIChatModel(
    settings.openai_chat_model,
    provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
)

# No temperature, deliberately: gpt-5.5 reasons by default, OpenAI rejects
# sampling parameters while it does, and pydantic-ai strips `temperature` with a
# UserWarning — so 0.0 would be no setting at all. `openai_agent_temperature`
# was deleted rather than left unwired. Determinism comes from the instructions
# and the grounding gate.

# Read once: the file cannot change under a running process, so a per-run read
# buys a blocking stat+read on the request path. A missing instructions.md is a
# broken deploy, and raising here says so at startup.
_INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")

agent = Agent(
    _model,
    deps_type=DocumentAgentDeps,
    # `instructions=`, not `system_prompt=`: a system prompt joins the message
    # history, so by turn nine the contract is eight exchanges back behind
    # everything the model has since said. Instructions belong to the run.
    # Checked at turn 3: one leading system message, no duplicates.
    instructions=_INSTRUCTIONS,
    # A union, so refusal is a shape the model selects rather than prose the
    # backend has to recognise. See outputs.py.
    output_type=[GroundedAnswer, InsufficientEvidence],
    # Two budgets, because the two failures differ. `output` is the grounding
    # gate: one correction, then fail. A model told its quote does not match
    # sometimes fixes it and sometimes drops the citation and keeps the claim,
    # which passes the validator and is the worst outcome of the three — so a
    # wider budget mostly buys more attempts at that. Exhausting it raises
    # UnexpectedModelBehavior, the controlled failure Phase 5 must show rather
    # than 500 on. `tools` is the mechanical kind, where the retry names the
    # five tickers and the fix is unambiguous: cheap, and per-tool.
    retries={"tools": 2, "output": 1},
)


@agent.tool
async def search_filings(
    ctx: RunContext[DocumentAgentDeps],
    question: str,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
) -> list[SourcePassage]:
    """Search the 10-K corpus for passages relevant to a question.

    Args:
        question: What to search for, in the filings' own vocabulary where you
            know it. This is matched semantically and lexically, so a phrase
            from a 10-K ("concentration of credit risk") retrieves better than a
            paraphrase of one.
        tickers: The companies the question is about, e.g. ["AAPL", "NVDA"].
            Each is searched separately and gets its own results, so a company
            with less on-topic language is still represented. Omit when the
            question is not about particular companies.
        years: The fiscal years the question is about, e.g. [2021, 2022]. Pass
            these together with `tickers` when the question is about how
            something changed over time; every company-year then gets its own
            results. Omit for a question about a single point in time. Given
            without `tickers`, the list is read as a span from its earliest year
            to its latest, so [2021, 2025] searches all five years.

    Returns:
        Passages, each with a handle to cite it by.

    ---
    Above the line is prompt: pydantic-ai builds the tool schema from the
    summary and the `Args:` block and drops everything after `Returns:`.
    Checked, not assumed — which is what makes it safe to argue down here.

    Strategy is derived from the arguments, not chosen by the model. Which
    companies and years a question is about is reading comprehension; which of
    the three retrievers serves that shape was measured in Phase 3 and is
    testable here in a way a prompt is not.

    Years without tickers widen to a closed range because that is what `Filters`
    expresses. Handing the grid those years instead would need companies, and
    the only ones to invent are all five — a 25-cell ~15,000-token search nobody
    asked for. The superset costs the model nothing, since every passage carries
    its own fiscal year.

    Rejecting an unknown ticker matters more than it looks: "TSLA" otherwise
    retrieves nothing, which is indistinguishable from a company the corpus does
    not discuss. The model reports "not covered" — true, and phrased as a
    refusal that reads exactly like an honest one.
    """
    scope = _validate_tickers(tickers)
    span = _validate_years(years)
    session = ctx.deps.session

    if scope and span:
        retrieved = await retrieve_grid(session, question, tickers=scope, years=span)
    elif scope:
        retrieved = await retrieve_per_ticker(session, question, scope)
    elif span:
        retrieved = await retrieve(
            session,
            question,
            limit=DEFAULT_SEARCH_LIMIT,
            filters=Filters(fiscal_year_from=span[0], fiscal_year_to=span[-1]),
        )
    else:
        retrieved = await retrieve(session, question, limit=DEFAULT_SEARCH_LIMIT)

    # Empty is a finding, not a failure: the model needs "nothing matched" to
    # reach InsufficientEvidence honestly.
    if not retrieved:
        return []

    try:
        return ctx.deps.offer(retrieved)
    except PassageBudgetExceeded as exceeded:
        raise ModelRetry(
            f"{exceeded} Search again for fewer companies or fewer years, or "
            f"answer from the passages you already have."
        ) from exceeded


def _validate_tickers(tickers: list[str] | None) -> list[str]:
    """The requested companies, normalised, or a ModelRetry naming the corpus."""
    if not tickers:
        return []

    # Same normalisation Filters and grid_search apply, so a model writing
    # "aapl" cannot produce a scope that disagrees with the query it drives.
    scope = list(dict.fromkeys(t.strip().upper() for t in tickers))
    unknown = [t for t in scope if t not in CORPUS_TICKERS]
    if unknown:
        raise ModelRetry(
            f"Not in the corpus: {', '.join(unknown)}. It holds only "
            f"{', '.join(sorted(CORPUS_TICKERS))}. Search again for those, and "
            f"say in your answer that the corpus does not cover the rest."
        )
    return scope


def _validate_years(years: list[int] | None) -> list[int]:
    """The requested fiscal years, sorted and deduplicated, or a ModelRetry.

    Sorted because the years-only path reads first and last as a span, and
    [2025, 2021] means the range [2021, 2025] does.
    """
    if not years:
        return []

    span = sorted(dict.fromkeys(years))
    outside = [y for y in span if y not in CORPUS_YEARS]
    if outside:
        raise ModelRetry(
            f"Not in the corpus: {', '.join(str(y) for y in outside)}. It holds "
            f"fiscal years {CORPUS_YEARS.start}-{CORPUS_YEARS[-1]}. Search again "
            f"within those, and say in your answer which years are missing."
        )
    return span


@agent.tool
async def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps],
    handles: list[str],
) -> dict[str, str]:
    """Read the passages either side of a search result, from the same Item.

    Use this when a passage you intend to cite is clearly mid-thought — it opens
    with "In addition," or refers to a list that is not shown. It costs roughly
    three times the tokens of the passage itself, so it is not worth running
    over every result. Tables are whole as retrieved and have no surrounding
    prose, so a table handle comes back omitted rather than widened.

    Args:
        handles: Handles of passages to widen, from an earlier search.

    Returns:
        Handle to widened text. A handle with nothing to add is omitted.

    ---
    Widening *replaces* the ledger text rather than sitting beside it. Quote
    checking reads the ledger, so a model quoting the widened window against a
    narrow entry would fail every such citation — rejected for being more
    careful, the worst way to lose an answer.

    What survives verbatim is the anchor's *body*, not `passage.text`:
    `neighbours` emits the heading once and then each body in chunk_index order,
    so the preceding chunk sits between the two. A quote spanning heading into
    body therefore validates before this call and not after — accepted, because
    the instructions ask for a sentence or clause, and not replacing fails far
    more often.

    The write goes through `ctx.deps.ledger[handle]` so it does not depend on
    `resolve` returning the stored object rather than a copy. Tables are
    explained in the prompt above rather than in the return value: a sentence
    there would share a channel with passage text, quotable and with nothing in
    the ledger to check it against.
    """
    resolved: dict[str, SourcePassage] = {}
    unknown: list[str] = []
    # dict.fromkeys: a handle listed twice costs one lookup, and the order asked
    # in is worth keeping.
    for handle in dict.fromkeys(handles):
        passage = ctx.deps.resolve(handle)
        if passage is None:
            unknown.append(handle)
        else:
            resolved[handle] = passage

    # Not a silent omission: an absent handle reads as "that passage has no
    # surrounding context", and the model would cite it as-is having been told
    # nothing.
    if unknown:
        raise ModelRetry(
            f"No passage in this turn has the handle {', '.join(unknown)}. Use "
            f"only handles that search_filings returned in this conversation, "
            f"and search again if you need a passage you have not retrieved."
        )
    if not resolved:
        return {}

    # `neighbours` reads identity alone, which SourcePassage carries under the
    # same names as the Passage its signature names. It also constrains the
    # window to the same *section*: 23.3% of adjacent pairs straddle a section
    # boundary, so a raw chunk_index window attaches another Item's prose to
    # this citation one time in four.
    windows = await neighbours(ctx.deps.session, list(resolved.values()))

    widened: dict[str, str] = {}
    for handle, passage in resolved.items():
        window = windows.get((passage.source_type, passage.row_id))
        # A lone anchor rebuilds its own text exactly. Returning it spends
        # tokens telling the model what it already holds.
        if window is None or window == passage.text:
            continue
        ctx.deps.ledger[handle] = passage.model_copy(update={"text": window})
        widened[handle] = window

    return widened


@agent.output_validator
async def enforce_grounding(
    ctx: RunContext[DocumentAgentDeps],
    output: GroundedAnswer | InsufficientEvidence,
) -> GroundedAnswer | InsufficientEvidence:
    """No answer leaves this agent without its citations checked.

    The gate the rest of the module is shaped around: the ledger exists so this
    has something to check against, handles so it can be mechanical. It runs on
    every candidate output, retries included.

    `validate` returns a ValidatedAnswer and this must return a GroundedAnswer,
    so the resolved passages cannot leave through here. Phase 5 re-runs
    `validate` on the final output for `cited_passages` — pure, no I/O,
    microseconds — which guarantees what the analyst sees came from the same
    function that admitted the answer. Stashing it on `ctx.deps` trades that for
    per-turn mutable state; making ValidatedAnswer the output type would have
    the model building `cited_passages` itself, the path outputs.py closes.

    `async def` because pydantic-ai runs a sync validator in an executor.
    """
    # A refusal makes no claim and cites nothing; running the rules over it
    # could only invent a violation.
    if isinstance(output, InsufficientEvidence):
        return output

    try:
        validate(output, ctx.deps.ledger)
    except GroundingError as error:
        # From `error.violations`, not `str(error)`: the exception carries them
        # structurally so each caller phrases them for its own audience.
        #
        # "Remove the claim, not the citation" is the load-bearing sentence. A
        # model told its quote does not match will otherwise take the cheapest
        # path to a passing answer — delete the citation, leave the sentence —
        # which is an uncited claim waved through by the check meant to stop it.
        # Naming the honest exits makes the cheap one the least attractive.
        raise ModelRetry(
            "This answer did not pass the citation check:\n"
            + "\n".join(f"- {v.handle}: {v.problem}" for v in error.violations)
            + "\n\nFix each one against the passages you were given. A quote must "
            "be copied character for character from the passage it cites. Where "
            "a claim has no passage from this turn behind it, remove the claim "
            "itself rather than its citation, or return InsufficientEvidence."
        ) from error

    return output


# Rejected, not deferred: `read_chunk`, the third tool architecture.md names.
# `search_filings` returns full passage text, so it could only re-return what is
# already in the context window. Snippets would change that (a 25-cell grid at
# 220 characters is ~2,000 tokens against ~15,000) — but a comparative question
# would then need 25 follow-up reads against a request limit of 20, so snippets
# imply a *batched* read over many handles, which is the shape
# read_surrounding_chunks already has. Two tools changed, not a third added.
# Reopen when a turn actually overflows the context window.
