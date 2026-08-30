"""One chat turn, end to end.

Load the thread, build the turn's deps, run the agent, validate, persist. The
route knows about HTTP; the agent knows about the model; this knows the order.

**Streaming and the grounding contract are in tension, and the contract wins.**
The obvious design streams the answer to the browser as the model writes it.
That cannot work here: `enforce_grounding` runs on the *final* output, so
anything streamed before it is text no one has checked — and if the gate then
rejects, the analyst has already read a confident unsourced claim and the
retraction arrives second. That is precisely the failure this product exists to
prevent, reintroduced by the transport.

Two facts make the decision easier than it sounds. `output_type` is a union, so
the model answers by calling an output tool, not by emitting prose — there is no
text stream to forward, only partial JSON. And the measured runs take 2-5 model
requests, most of it retrieval, so the wait is dominated by tool calls rather
than by token generation.

So: stream *progress*, never unvalidated prose. The tool trace is genuinely
useful to an analyst — "searching AAPL, MSFT 2021-2025", "reading 25 passages" —
and it is honest, because it reports what happened rather than guessing what the
answer will say. The answer arrives once, whole, already grounded.

TODO: revisit only with a measurement. If a turn's generation phase turns out to
 dominate its wall clock, the option that keeps the contract is streaming the
 answer *after* the gate passes, chunked for feel rather than for latency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from app.assistant.outputs import ValidatedAnswer
from app.auth.dependencies import CurrentUser


async def run_turn(
    user: CurrentUser,
    thread_id: UUID,
    question: str,
) -> AsyncIterator[object]:
    """Run one turn, yielding progress events, then persist and yield the answer.

    An async iterator rather than a coroutine returning a result: the route
    needs to emit events while this is still working, and the alternative is a
    callback threaded through every layer.

    TODO: implement.
      - Own the session for the whole turn. `DocumentAgentDeps` holds it, the
        agent's tools query through it, and `record_turn` writes through it.
      - Load the thread scoped to the user *before* spending a model call on a
        thread they do not own.
      - Build a fresh `DocumentAgentDeps` per turn. Handles are turn-scoped, and
        reusing deps across turns would let this turn cite a passage retrieved
        for the last one — the ledger would resolve it perfectly.
      - `agent.run(..., usage_limits=UsageLimits(request_limit=...))`. The CLI
        already does this; `settings.openai_agent_request_limit` is otherwise a
        number nothing reads.
      - Re-run `validate` on the final output to get `cited_passages`. Pure, no
        I/O, and it means the passages the analyst is shown were resolved by the
        same function that admitted the answer.
      - Persist only on success, in one transaction.

    TODO: the failure paths, which are the product rather than an afterthought.
      - `UnexpectedModelBehavior` from an exhausted grounding budget is the
        controlled failure: the answer could not be grounded. It must reach the
        analyst as *that*, not as a 500. Measured, it did not fire once across
        the ten brief questions, which makes it the path most likely to be
        wrong when it does.
      - `InsufficientEvidence` is a successful turn, not an error. Persist it.
        Whether it gets citation rows is a real question — it has no citations
        by construction, so the answer message simply has none.
      - A model or database outage is a 502, and nothing is persisted.
    """
    raise NotImplementedError


async def _persist(answer: ValidatedAnswer) -> None:
    """TODO: fold into run_turn unless it grows a second caller."""
    raise NotImplementedError
