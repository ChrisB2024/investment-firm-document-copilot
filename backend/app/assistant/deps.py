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
from app.retrieval.queries import SourceType
from app.retrieval.retriever import RetrievedPassage

# Handles are short and opaque on purpose. A UUID is 36 characters the model has
# to copy without a typo, and it pays that 25 times for a full grid; "S7" is two.
# The prefix keeps them from reading as a footnote number or a year in prose.
HANDLE_PREFIX = "S"

# A ceiling on how many passages one turn may offer the model, across every tool
# call. Measured in Phase 3: a 25-cell grid at per_cell=1 is ~15,000 tokens, so
# two grid calls and a neighbour expansion would fill a context window with
# retrieval alone and leave the instructions competing with it.
#
# It bounds the *count*, which is a proxy for size and not the thing itself:
# `read_surrounding_chunks` swaps a passage for its ~3x window without changing
# the count, so sixty widened passages sit well above the ceiling this number
# implies. Make it a token budget if a real turn ever gets there; none has.
# settings.openai_agent_request_limit bounds how many calls a turn may make,
# this bounds what one call may hand back.
MAX_PASSAGES_PER_TURN = 60

# What a re-offered passage's text is replaced with. Tool returns accumulate in
# the message history, so a second grid overlapping twenty passages would repeat
# ~10,800 tokens the model is already holding verbatim. The passage still comes
# back, because an almost-empty result reads as "nothing more was found" and
# stops the model looking — but its body does not need to arrive twice.
ALREADY_SHOWN = "(already returned earlier this turn under this handle)"


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

    # Counts up, never down, and never derived from len(ledger). The two agree
    # today — nothing removes a handle — but a handle reused after a removal
    # would point a citation at a passage the model never read, and the quote
    # check is the only thing that would notice.
    _minted: int = field(default=0, init=False, repr=False)

    def offer(self, retrieved: list[RetrievedPassage]) -> list[SourcePassage]:
        """Mint a handle for each passage, record it, and return what to show.

        Called by every tool that surfaces passages. Returning the
        `SourcePassage` list rather than writing to the ledger and letting the
        tool build its own reply keeps one rule true: a passage the model can
        see is a passage the ledger holds.

        A passage already in the ledger keeps its original handle and is
        returned again rather than dropped. Both halves matter. One row must
        never hold two handles, or the model can "corroborate" a claim with a
        single piece of evidence cited twice; but a second search that lands on
        the same passages must still come back as a result, or the model reads
        an almost-empty list as "nothing more was found" and stops looking.

        Keyed on (source_type, row_id), the key `fuse` and `hydrate` use, and
        for their reason: a chunk id and a table id are both UUIDs and can
        collide.

        A re-offered passage comes back with its body replaced by ALREADY_SHOWN.
        The model is already holding that text verbatim from the earlier tool
        return, and repeating it buys nothing but tokens. Everything that
        identifies the passage — handle, company, year, caption — still comes
        back, so the model can see that the search found it again.

        The budget counts only genuinely new passages, since re-offering one the
        turn already holds costs nothing. It raises rather than truncating: a
        silently shortened grid produces a confident answer built on evidence
        nobody was told was missing, and `search_filings` turns the raise into a
        ModelRetry asking the model to narrow.

        `title` is carried because for a table it is not decoration.
        `document_tables.markdown` is the grid alone, so "Unconditional Purchase
        Obligations" arrives as a column of years and unlabelled figures without
        it — and brief question 8 asks about purchase commitments. It also
        carries the scale: of the 1,131 tables with a recorded `units`, the unit
        string appears in the title or the markdown for all of them, so passing
        the title through makes `units` redundant rather than a fourth column to
        hydrate. Chunks have no title; `_HYDRATE` selects NULL for them.

        `section` still cannot be carried, because `Passage` does not have it —
        `_HYDRATE` selects neither `document_chunks.section` nor
        `table_data->>'section'`. Chunk text is prefixed with its heading by
        `ingest.chunk`, so the model does see the Item; what is missing is the
        structured field a citation UI renders. That is a column in that query
        and a field on `Passage`, not a change here.
        """
        # Rebuilt per call rather than kept as a second index: the ledger holds
        # at most MAX_PASSAGES_PER_TURN entries, so this is sixty items of work
        # against a class of bug — an index that drifts from the ledger it
        # describes — that would be invisible until a citation resolved wrong.
        handles = {(p.source_type, p.row_id): h for h, p in self.ledger.items()}

        # First occurrence wins, and insertion order is retrieval's order, which
        # for a fan-out or a grid is deliberately balanced across companies.
        unique: dict[tuple[SourceType, UUID], RetrievedPassage] = {}
        for candidate in retrieved:
            unique.setdefault(
                (candidate.passage.source_type, candidate.passage.row_id),
                candidate,
            )

        # Captured before the mint loop writes into `handles`: these are the
        # passages this turn has already shown, and the ones whose body is
        # redundant on the way out.
        known = set(handles)

        fresh = [(key, r) for key, r in unique.items() if key not in handles]
        remaining = MAX_PASSAGES_PER_TURN - len(self.ledger)
        if len(fresh) > remaining:
            raise PassageBudgetExceeded(
                f"This turn can hold {MAX_PASSAGES_PER_TURN} passages and has "
                f"{remaining} left; this search needs room for {len(fresh)}."
            )

        for key, candidate in fresh:
            self._minted += 1
            handle = f"{HANDLE_PREFIX}{self._minted}"
            passage = candidate.passage
            self.ledger[handle] = SourcePassage(
                handle=handle,
                ticker=passage.ticker,
                fiscal_year=passage.fiscal_year,
                form=passage.form,
                title=passage.title,
                text=passage.text,
                source_type=passage.source_type,
                row_id=passage.row_id,
                document_id=passage.document_id,
            )
            handles[key] = handle

        # Read back out of the ledger, never from what was just built: a passage
        # widened by read_surrounding_chunks must come back widened, and this is
        # what makes "what the model sees is what the ledger holds" true for a
        # first offer as well as a re-offer.
        #
        # The elision is a view, not a write. The ledger keeps the real text, so
        # quote checking and read_surrounding_chunks are unaffected — and a quote
        # of the marker itself fails validation, which is the right answer.
        offered: list[SourcePassage] = []
        for key in unique:
            passage = self.ledger[handles[key]]
            if key in known:
                passage = passage.model_copy(update={"text": ALREADY_SHOWN})
            offered.append(passage)
        return offered

    def resolve(self, handle: str) -> SourcePassage | None:
        """The passage behind a handle, or None if this turn never offered it.

        None is the whole point, and it is why this returns rather than raises:
        the validator turns an unresolvable handle into a specific message for
        the model, and one that names *which* handle is wrong is a retry the
        model can act on.

        Exact lookup, deliberately. "[S3]" and "s3" resolve to nothing here.
        Any leniency belongs in the validator, next to the rule it bends and
        applied once, rather than spread across every caller of this method —
        and a model that cannot copy a two-character handle exactly is a model
        whose care with the passage behind it is worth doubting.
        """
        return self.ledger.get(handle)
