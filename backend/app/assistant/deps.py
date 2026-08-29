"""Per-turn state the agent's tools run against.

One object per chat turn, holding the database session and the turn's ledger of
offered passages. The ledger is what makes "the model cannot cite what was not
retrieved" a structural property rather than a prompt instruction: a handle
exists only because a tool minted it during *this* turn, so a fabricated handle
resolves to nothing and a handle from a previous turn resolves to nothing
either.

architecture.md sketches this holding a `retriever` and a `grounding_validator`.
It does not, deliberately. Both are module-level functions with no state to
configure, and passing a function through a dataclass to get dependency
injection is the "framework where a function would do" case ../CLAUDE.md rules
out. What genuinely varies per request is the session and the user; those are
here, and the tools import `app.retrieval` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.outputs import SourcePassage
from app.retrieval.retriever import RetrievedPassage

# Handles are short and opaque on purpose. A UUID is 36 characters the model has
# to copy without a typo, and it pays that 25 times for a full grid; "S7" is two.
# The prefix keeps them from reading as a footnote number or a year in prose.
HANDLE_PREFIX = "S"

# A ceiling on how many passages one turn may offer the model, across every tool
# call. Measured in Phase 3: a 25-cell grid at per_cell=1 is ~15,000 tokens, so
# two grid calls and a neighbour expansion would fill a context window with
# retrieval alone and leave the instructions competing with it. The agent's
# request limit (settings.openai_agent_request_limit) bounds the number of
# calls; this bounds their size, which is the axis that actually hurts.
MAX_PASSAGES_PER_TURN = 60


class PassageBudgetExceeded(Exception):
    """`offer` was asked for more passages than one turn may hold.

    Its own type because the tool that catches it turns it into a `ModelRetry`
    telling the model to narrow, and "narrow your search" is the wrong advice
    for any other failure `offer` could have. The message should say what the
    call asked for against what was left.
    """


@dataclass
class DocumentAgentDeps:
    """The turn: who is asking, which thread, and what has been offered so far.

    Not reusable across turns. `offer` mints handles from a counter that only
    goes up within one instance, so building one per turn is what makes stale
    handles unresolvable.
    """

    session: AsyncSession
    user_id: UUID
    thread_id: UUID

    # handle -> passage. Insertion-ordered, which is the order the passages were
    # offered, which is the order `ValidatedAnswer.cited_passages` should
    # reasonably come back in.
    ledger: dict[str, SourcePassage] = field(default_factory=dict)

    def offer(self, retrieved: list[RetrievedPassage]) -> list[SourcePassage]:
        """Mint a handle for each passage, record it, and return what to show.

        Called by every tool that surfaces passages. Returning the
        `SourcePassage` list rather than writing to the ledger and letting the
        tool build its own reply keeps one rule true: a passage the model can
        see is a passage the ledger holds.

        TODO: implement.
          - Skip a passage already in the ledger and reuse its handle. A
            comparative question can search twice and hit the same passage, and
            two handles for one row would let the model "corroborate" a claim
            with a single piece of evidence cited twice. Key on
            (source_type, row_id) — the same key `fuse` and `hydrate` use, and
            for the same reason: a chunk id and a table id are both UUIDs.
          - Mint as f"{HANDLE_PREFIX}{n}" where n counts from 1 across the turn,
            not from 1 per tool call. Two calls each starting at S1 would make
            the second call's S1 overwrite the first's.
          - Enforce MAX_PASSAGES_PER_TURN by raising PassageBudgetExceeded.
            The alternative was truncating and letting the model work with what
            it has, which silently drops evidence the answer may need and says
            nothing; a wrong answer built on a shortened grid is exactly the
            failure this product cannot have. `search_filings` already catches
            the raise and turns it into a ModelRetry, so a truncating
            implementation would leave that handler dead.
          - Carry `section` across. `Passage` does not have it (see the TODO in
            agent.py about hydrate), so this is where the gap shows up.
        """
        raise NotImplementedError

    def resolve(self, handle: str) -> SourcePassage | None:
        """The passage behind a handle, or None if this turn never offered it.

        None is the whole point, and it is why this returns rather than raises:
        the validator turns an unresolvable handle into a specific message for
        the model, and one that names *which* handle is wrong is a retry the
        model can act on.

        TODO: implement. Consider whether to accept a handle the model wrote as
         "[S3]" or "s3" — the strict reading is that a handle it cannot copy
         exactly is a handle it may not have read carefully either. Strip
         nothing here; normalise once, in the validator, where the decision is
         visible next to the rule it bends.
        """
        raise NotImplementedError
