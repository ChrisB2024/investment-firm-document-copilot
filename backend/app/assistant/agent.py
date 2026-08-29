"""The LLM boundary: instructions, two tools, one typed output.

Everything below this line is deterministic and tested without a model —
retrieval, fusion, grounding. Everything above it is a chat turn. This module is
the seam, and it is deliberately thin: it decides what the model may call and
what it may return, and it owns no retrieval policy of its own.

Two tools, where architecture.md names three. `read_chunk` is dropped because
`search_filings` already returns each passage's full text, so a tool to fetch
the text of a passage the model is already holding has nothing to do — see the
TODO below if that changes.

The model never writes SQL and never sees a row id it could query by. It sees
handles, and the only thing it can do with one is cite it.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import (
    GroundedAnswer,
    InsufficientEvidence,
    SourcePassage,
)
from app.config import settings

# How many passages a single search returns when the question is not
# comparative. `retrieval_top_k` is 10, which Phase 3 measured as the right
# depth for a one-company question — brief question 1 asks only about Apple and
# its top-10 is all Apple and all on topic.
DEFAULT_SEARCH_LIMIT = settings.retrieval_top_k


# TODO: build the model explicitly rather than passing the string "openai:gpt-5.5".
#  The string form makes pydantic-ai read OPENAI_API_KEY out of os.environ, and
#  ../CLAUDE.md puts every environment variable behind app.config. The key is a
#  SecretStr there, so:
#
#      from pydantic_ai.models.openai import OpenAIChatModel
#      from pydantic_ai.providers.openai import OpenAIProvider
#
#      _model = OpenAIChatModel(
#          settings.openai_chat_model,
#          provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
#      )
#
#  app/embeddings.py already builds its OpenAI client this way; match it.
#
# TODO: settings.openai_agent_temperature is 0.0 and gpt-5.5 may reject a
#  temperature at all (the reasoning models do). Find out before wiring it into
#  model_settings, and if it is rejected, delete the setting rather than leaving
#  a config knob that silently does nothing.
#
# TODO: load instructions.md at import, not per run — it is a file read on the
#  request path otherwise. `(Path(__file__).parent / "instructions.md").read_text()`.
#  Passing it as `instructions=` rather than `system_prompt=`: instructions are
#  re-sent with every request in a multi-turn conversation, system prompts from
#  earlier turns are not, and the grounding contract has to hold on turn nine as
#  firmly as on turn one.
agent = Agent(
    deps_type=DocumentAgentDeps,
    # A union, so refusal is a shape the model selects rather than prose the
    # backend has to recognise. See outputs.py.
    output_type=[GroundedAnswer, InsufficientEvidence],
    # TODO: retries. This is the budget for output_validator raising ModelRetry,
    #  and it wants to be small — 1, maybe 2. Each retry is a full turn's tokens,
    #  and a model that cited a passage it did not read will not usually fix that
    #  on the third attempt; it will drop the citation and keep the claim, which
    #  is worse. Cap it low and let the controlled failure happen.
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
            results. Omit for a question about a single point in time.

    Returns:
        Passages, each with a handle to cite it by.

    ---
    Everything above the line is prompt: pydantic-ai turns the summary and the
    `Args:` block into the tool description and the per-argument schema the
    model reads, and drops everything after `Returns:`. Checked, not assumed —
    the notes below are invisible to the model, which is what makes it safe to
    argue with ourselves down here.

    The strategy is derived from the arguments rather than chosen by the model,
    which is the split retriever.py argues for: the agent knows which companies
    and years a question is about, and that is a reading-comprehension task it
    is good at. Which of `retrieve` / `retrieve_per_ticker` / `retrieve_grid`
    serves that shape is a property of retrieval, measured in Phase 3, and
    testable here in a way a paragraph of prompt is not.

    TODO: implement.
      - tickers and years   -> retrieve_grid(question, tickers=..., years=...)
        tickers only        -> retrieve_per_ticker(question, tickers)
        neither             -> retrieve(question, limit=DEFAULT_SEARCH_LIMIT)
        years only          -> retrieve(question, filters=Filters(fiscal_year_from=min,
                               fiscal_year_to=max)). Note this widens a discrete
                               list into a closed range, because that is what
                               `Filters` expresses; [2021, 2025] therefore means
                               2021-2025, not those two years. Decide whether
                               that is acceptable or whether the grid should
                               take over as soon as `years` is present at all.
      - Offer the results through ctx.deps.offer() and return what it gives
        back. Never build a SourcePassage here — a passage the model can see
        must be a passage the ledger holds, and one function doing both is what
        keeps that true.
      - Turn an empty result into a plain empty list, not a raise. "Nothing
        matched" is information the model needs in order to reach
        InsufficientEvidence honestly, and it is different from a broken search.
      - Validate `tickers` against the corpus and raise ModelRetry naming the
        five that exist. A typo or a company we do not hold ("TSLA") otherwise
        comes back as zero results, which the model will report as "the corpus
        does not cover this" — a refusal that reads exactly like a true one.
        Same for years outside 2021-2025.

    TODO: budget. A grid call is ~15,000 tokens; nothing here stops the model
     issuing three. MAX_PASSAGES_PER_TURN in deps.py is the intended guard —
     decide there whether it truncates or raises, and turn a raise into a
     ModelRetry here telling the model to narrow.
    """
    raise NotImplementedError


@agent.tool
async def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps],
    handles: list[str],
) -> dict[str, str]:
    """Read the passages either side of a search result, from the same Item.

    Use this when a passage you intend to cite is clearly mid-thought — it opens
    with "In addition," or refers to a list that is not shown. It costs roughly
    three times the tokens of the passage itself, so it is not worth running
    over every result.

    Args:
        handles: Handles of passages to widen, from an earlier search.

    Returns:
        Handle to widened text. A handle with nothing to add is omitted.

    ---
    TODO: implement.
      - Resolve each handle through ctx.deps.resolve(). An unresolvable one is a
        ModelRetry naming it, not a silent omission: the model would otherwise
        read the gap as "that passage has no context" and cite it as-is.
      - Call retriever.neighbours() with the resolved passages. It already does
        the hard part — the window is constrained to the same *section*, because
        23.3% of adjacent chunk pairs straddle a section boundary and
        `chunk_index ± 1` alone attaches another Item's prose to this citation
        one time in four.
      - Tables have no neighbours and `neighbours` already drops them. Returning
        them absent is right; consider whether the model should be told why,
        since a table is exactly the kind of result it might try to widen.

    TODO: decide whether widening replaces the passage text in the ledger or
     sits beside it. It matters for quote checking: if the model quotes from the
     widened window and the ledger still holds the narrow passage, every such
     citation fails validation and the answer is rejected for being *more*
     careful. Replacing is the fix, and it means `offer` and this tool write to
     the same field.
    """
    raise NotImplementedError


# TODO: @agent.output_validator — the grounding gate.
#
#   @agent.output_validator
#   async def enforce_grounding(
#       ctx: RunContext[DocumentAgentDeps],
#       output: GroundedAnswer | InsufficientEvidence,
#   ) -> GroundedAnswer | InsufficientEvidence:
#
#  InsufficientEvidence passes through untouched — there is nothing to ground.
#  A GroundedAnswer goes to app.grounding.validator, and a GroundingError
#  becomes a ModelRetry carrying the violations, which is the one message the
#  model can actually act on.
#
#  The validator returns a ValidatedAnswer and the agent's output_type is
#  GroundedAnswer, so the resolved passages cannot come back through here.
#  Decide where they attach: probably the orchestrator in Phase 5 re-runs the
#  validate call on the final output, which is cheap and pure. The alternative —
#  making ValidatedAnswer the output type — would have the model constructing
#  `cited_passages` itself, which outputs.py rejects for good reason.
#
# TODO: read_chunk, if it earns its place. It does not today: search returns
#  full passage text, so the model is never holding a handle whose text it has
#  not seen. It would earn it if search started returning snippets — a 25-cell
#  grid at 220 characters each is ~2,000 tokens against ~15,000, which is a real
#  saving — but then a comparative question needs 25 follow-up reads against a
#  request limit of 20, and the arithmetic stops working. Leave it out until
#  something measured asks for it.
