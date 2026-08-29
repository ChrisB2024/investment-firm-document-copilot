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

# Handles as the model marks them in prose: [S3], [S12], and [S3, S4] where two
# passages support one claim. The group form is not indulgence — a model writes
# it unprompted, and matching only the singular form reported *every* citation
# in the answer as unmarked while the prose visibly marked them, which is the
# most confusing retry this module can send.
#
# Still matched rather than split on, so "[Table 3]" or a bracketed aside in a
# filing quote does not read as a citation.
_MARKER = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")
_HANDLE = re.compile(r"S\d+")

# Runs of whitespace fold to one space: a passage's line breaks are an artefact
# of chunking, and a model copying across one will normalise it. Verified that
# `\s` covers the spaces filings actually use — U+00A0, U+202F, U+2007 and
# U+3000 all match — and that the corpus holds no zero-width spaces, which it
# would not cover.
_WHITESPACE = re.compile(r"\s+")

# Curly quotation marks fold to straight ones. This is the one normalisation
# beyond whitespace, and it is here on measurement rather than principle: 54% of
# chunks contain U+2019 and 34% contain U+201C/D, so more than half the corpus
# sits one keystroke from rejecting a quote the model copied correctly. That is
# a *false rejection* — the expensive, invisible direction, for the same reason
# `_supports` gives for leaving the length floor open.
#
# It stops here. Case, spelling, punctuation and word choice stay strict,
# because those are what a citation attests to; the shape of an apostrophe is
# not a word.
_SMART_QUOTES = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)

# The `handle` on a violation that is about the answer as a whole rather than
# one citation. A word rather than an empty string because it is rendered into
# a list of handles for the model to read.
_ANSWER = "answer"

# How much of a quote to echo back in a violation. Enough to tell two citations
# on the same handle apart, and no more — the model is holding the full text it
# wrote, so repeating a 2,000-character quote at it costs a retry's budget to
# say nothing.
_QUOTE_EXCERPT = 60


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
        # One line per violation, each naming its handle. This string is the
        # one the API logs; the agent formats its own from `.violations`, which
        # is why the data comes first and the prose second.
        super().__init__(
            "\n".join(f"{v.handle}: {v.problem}" for v in violations)
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

    Duplicate handles in `citations` are allowed, deliberately. One passage
    supporting two claims is ordinary and correct, so a rule against repeats
    would reject good answers to catch the case that is merely untidy: the same
    handle with the same quote twice, which is padding rather than a false
    claim. `cited_passages` deduplicates, so the analyst sees one passage
    either way.
    """
    # Its own exit rather than a case that falls out of an empty loop. The model
    # had InsufficientEvidence available and asserted something instead, which
    # is the failure this whole module exists for, and it deserves to be told
    # that rather than handed a list of dangling markers saying the same thing
    # one handle at a time.
    if not answer.citations:
        marked = list(dict.fromkeys(_markers(answer.answer)))
        dangling = (
            f"It marks {', '.join(marked)} in the prose against an empty "
            f"citation list. "
            if marked
            else ""
        )
        raise GroundingError(
            [
                Violation(
                    handle=_ANSWER,
                    problem=(
                        f"this answer makes claims but lists no citations. "
                        f"{dangling}"
                        f"Every factual claim needs the handle of a passage from "
                        f"this turn; where the filings do not support one, return "
                        f"InsufficientEvidence instead of asserting it."
                    ),
                )
            ]
        )

    violations: list[Violation] = []

    for citation in answer.citations:
        passage = ledger.get(citation.handle)
        if passage is None:
            violations.append(
                Violation(
                    handle=citation.handle,
                    problem=(
                        "was not returned by any search in this turn, so there "
                        "is no passage to check the quote against. Cite only "
                        "handles from this turn's search results."
                    ),
                )
            )
        elif not _supports(citation.quote, passage):
            excerpt = citation.quote[:_QUOTE_EXCERPT]
            violations.append(
                Violation(
                    handle=citation.handle,
                    problem=(
                        f'the quote "{excerpt}" does not appear in that '
                        f"passage. Copy a span out of the passage character for "
                        f"character, or drop the claim it was meant to support."
                    ),
                )
            )

    cited = {citation.handle for citation in answer.citations}
    for handle in dict.fromkeys(_markers(answer.answer)):
        if handle not in cited:
            violations.append(
                Violation(
                    handle=handle,
                    problem=(
                        "is marked in the answer but has no entry in citations, "
                        "so it is a reference an analyst cannot follow."
                    ),
                )
            )

    for handle in _unused(answer.citations, answer.answer):
        violations.append(
            Violation(
                handle=handle,
                problem=(
                    "is listed in citations but never marked in the answer, so "
                    "it supports no claim and reads as corroboration it is not."
                ),
            )
        )

    if violations:
        raise GroundingError(violations)

    # Ledger order, not citation order: the ledger was filled in the order
    # retrieval returned passages, which for a fan-out or a grid is balanced
    # across companies. A dict comprehension over `cited` would instead give
    # whatever order the model happened to list its citations in.
    return ValidatedAnswer(
        answer=answer,
        cited_passages=[
            passage for handle, passage in ledger.items() if handle in cited
        ],
    )


def _supports(quote: str, passage: SourcePassage) -> bool:
    """Whether the quote appears in any part of the passage the model was shown.

    All three narrative fields, not `text` alone. A table's markdown is the grid
    by itself and its caption lives in `title`, so "Unconditional Purchase
    Obligations" — the natural way to cite a schedule of figures — is quoting
    the passage while failing a check against `text`. The rule is that a quote
    comes from what the model was given, and `text` is one field of that rather
    than the whole. `DocumentTable.embed_text` composes the same three, for the
    same reason.

    Each field is checked whole rather than joined, so a quote cannot span a
    caption into a table body and match across a seam that was never prose.

    An empty or whitespace-only quote is not a match. Without that line it would
    match every passage in the corpus, since "" is a substring of everything —
    the one degenerate quote that has to fail.

    **No minimum length, for now.** A two-word quote attests to almost nothing
    and passes this check, which is a real hole. It is left open because every
    floor that closes it also rejects a legitimate quote: the shortest honest
    span in this corpus is a bare figure, and "$96.2 billion" is two words, so a
    floor high enough to reject "the company" rejects that too. Rejecting a true
    citation is the more expensive mistake, and unlike a thin quote it is
    invisible — an analyst can see that a quote is weak and judge it, and cannot
    see an answer that was never returned. Close this by measuring what the
    model actually emits across the ten brief questions, then putting a floor
    under its shortest legitimate quote.
    """
    folded = _fold(quote)
    if not folded:
        return False
    return any(
        folded in _fold(field)
        for field in (passage.text, passage.title, passage.section)
        if field
    )


def _fold(value: str) -> str:
    """Whitespace runs to one space, curly quotes to straight. Nothing else."""
    return _WHITESPACE.sub(" ", value.translate(_SMART_QUOTES)).strip()


def _markers(answer: str) -> list[str]:
    """The handles marked in the answer prose, in order, with duplicates kept.

    A grouped marker expands: "[S3, S4]" is two handles, in the order written.

    Order and duplicates are both wanted: a handle cited twice is two claims
    resting on one passage, and the caller may want to know that even if no rule
    currently forbids it.
    """
    return [h for group in _MARKER.findall(answer) for h in _HANDLE.findall(group)]


def _unused(citations: list[Citation], answer: str) -> list[str]:
    """Handles listed in `citations` that never appear in the prose.

    Separate from `_markers` because the two directions fail differently and the
    model needs to hear which one happened: a marker with no citation is a
    dangling reference, and a citation with no marker is evidence attached to no
    claim.

    Unique, in the order `citations` lists them, so a model that listed the same
    unused handle twice hears about it once.
    """
    marked = set(_markers(answer))
    return list(
        dict.fromkeys(c.handle for c in citations if c.handle not in marked)
    )
