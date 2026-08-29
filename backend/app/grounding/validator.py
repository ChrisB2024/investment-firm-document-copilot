"""The trust contract, enforced mechanically.

Three checks, and the order they are stated in is the order of how much they
buy:

1. **Every cited handle was offered this turn.** Structural, and nearly free —
   handles are minted per turn, so a fabricated or stale one resolves to
   nothing. This is the check the architecture asks for.
2. **Every quote appears verbatim in the passage it cites.** This is the one
   that matters. Check 1 says the passage was retrieved; it says nothing about
   whether the passage supports the claim, and "cited a real passage that does
   not say this" is the failure mode a fluent model actually produces. A
   verbatim span is checkable; a paraphrase is not, which is why Citation asks
   for a span.
3. **The prose and the citation list agree.** A handle marked `[S3]` in the
   answer with no entry in `citations` gives the analyst a citation to click
   that resolves to nothing; an entry in `citations` never marked in the prose
   is evidence attached to no claim, which reads as corroboration and is not.

What none of this checks is whether the quote supports the claim it is attached
to. That is a judgement, and it belongs to the analyst — which is the whole
reason the passage is one click away in the UI. The guarantee this module makes
is narrower and worth stating exactly: every claim points at a real passage from
this turn, and the words it attributes to that passage are that passage's words.

A violation is a controlled failure. Never a polished answer with the bad
citation quietly dropped: the claim would survive the citation, and an uncited
claim in a Driftwood report is the exact thing this product exists to prevent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.assistant.outputs import (
    Citation,
    GroundedAnswer,
    SourcePassage,
    ValidatedAnswer,
)

# Handles as the model marks them in prose: [S3], [S12]. Matched rather than
# split on, so ordinary bracketed text in a filing quote does not read as a
# citation.
_MARKER = re.compile(r"\[(S\d+)\]")

# Whitespace only. A quote is compared after folding runs of whitespace to a
# single space, because a passage's line breaks are an artefact of chunking and
# a model copying across one will normalise it. Nothing else is folded — not
# case, not punctuation, not quotation marks — because each of those is a way
# for a "verbatim" quote to stop being verbatim, and a filing's own commas and
# capitals are the thing being attested to.
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Violation:
    """One broken rule, phrased for the model that has to fix it."""

    handle: str
    problem: str


class GroundingError(Exception):
    """The answer does not meet the citation contract.

    Carries the violations rather than a formatted string so both callers can
    use it: the agent turns them into a ModelRetry, and the API turns them into
    a typed error event and a log line.
    """

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        super().__init__(
            # TODO: one line per violation, naming the handle. "Citation S7 was
            #  not returned by any search this turn" is a retry the model can
            #  act on; "grounding failed" is not.
            ...
        )


def validate(
    answer: GroundedAnswer,
    ledger: Mapping[str, SourcePassage],
) -> ValidatedAnswer:
    """Check every citation against the turn's ledger, or raise.

    Takes the ledger as a mapping rather than the whole `DocumentAgentDeps`, so
    the rules can be tested against a dict literal with no agent, no session and
    no model — which is most of what makes them worth having.

    Collects every violation before raising. Reporting the first one costs a
    round trip per bad citation, and an answer with three fabricated handles is
    one the model should see all three of at once.

    TODO: implement.
      - An answer with no citations at all is a violation on its own. The model
        had InsufficientEvidence available and chose to assert something
        instead; that is the case this rule exists for, and it needs its own
        message rather than falling out of an empty loop as a pass.
      - Resolve each citation's handle. Unresolvable -> Violation.
      - Quote check via `_supports`, for citations that resolved.
      - Marker check via `_markers`, both directions.
      - Duplicate handles in `citations`: decide. Two citations on the same
        passage with different quotes is legitimate — one passage can support
        two claims. Two with the *same* quote is the model padding. Probably not
        worth a rule; note the decision either way so the next reader knows it
        was considered.
      - On success, return ValidatedAnswer with cited_passages resolved from the
        ledger, deduplicated, in ledger order.
    """
    raise NotImplementedError


def _supports(quote: str, passage: SourcePassage) -> bool:
    """Whether the quote appears in the passage, ignoring whitespace runs.

    TODO: implement — fold both sides with `_WHITESPACE.sub(" ", ...)`, strip,
     then substring test.

    Two things to settle with the corpus in front of you rather than by
    reasoning about it:

    - **The floor.** A two-word quote appears in almost any passage, so it
      passes this check while attesting to nothing. Some minimum length is
      needed; measure what the model actually emits before picking one, because
      a floor above its natural quote length rejects good answers.
    - **Non-breaking spaces and unicode punctuation.** Filings are full of both,
      `\\xa0` especially, and `\\s` does match it — but a curly apostrophe copied
      as a straight one is a different character and this check will say the
      quote is not there. That may be right (it is not verbatim) or may be the
      rule being pedantic about a difference no analyst would care about. Look
      at real failures before deciding; do not pre-emptively normalise, because
      every normalisation is a way a quote can differ from the filing and still
      pass.
    """
    raise NotImplementedError


def _markers(answer: str) -> list[str]:
    """The handles marked in the answer prose, in order, with duplicates kept.

    Order and duplicates are both wanted: a handle cited twice is two claims
    resting on one passage, and the caller may want to know that even if no rule
    currently forbids it.

    TODO: implement — `_MARKER.findall(answer)`.
    """
    raise NotImplementedError


def _unused(citations: list[Citation], answer: str) -> list[str]:
    """Handles listed in `citations` that never appear in the prose.

    Separate from `_markers` because the two directions fail differently and the
    model needs to hear which one happened: a marker with no citation is a
    dangling reference, and a citation with no marker is evidence attached to no
    claim.

    TODO: implement.
    """
    raise NotImplementedError
