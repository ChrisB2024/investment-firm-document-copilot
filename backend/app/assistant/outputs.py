"""What a tool may offer the agent, and what the agent may return.

Five models, and the boundaries between them are the grounding contract
expressed as types rather than as prompt text.

`SourcePassage` travels one way. A tool builds it from a `RetrievedPassage` and
hands it to the model; nothing ever reads passage text back off the model's
output. That is the whole reason `Citation` carries a handle and a quote rather
than the passage itself — a model that can restate a passage can restate it
wrong, and the restatement is what the analyst would read and trust.

`GroundedAnswer` and `InsufficientEvidence` are the two ways a turn can end.
Refusal is a *type*, not a `GroundedAnswer` with an empty citation list: that is
what lets the validator say "an answer cites at least one passage" with no
exception to carve out, and what lets the frontend render a refusal deliberately
instead of sniffing prose for one.

`ValidatedAnswer` is what leaves grounding — the model's answer plus the
passages the backend resolved from the turn's ledger.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.retrieval.queries import SourceType


class SourcePassage(BaseModel):
    """One retrieved passage, as the model sees it.

    Built by a tool, never parsed back from the model. The field descriptions
    are prompt text — they are what the model reads to decide whether a passage
    is worth citing — so they are written for a reader, not as type notes.
    """

    handle: str = Field(
        description="Cite this passage by this handle, exactly as written."
    )
    ticker: str
    fiscal_year: int
    form: str
    section: str | None = Field(
        default=None,
        description="The Item this passage came from, e.g. 'Item 1A. Risk Factors'.",
    )
    text: str

    # Identity, for the backend. Not part of what makes a citation checkable —
    # the handle is — but the citation *record* in Phase 5 is a foreign key to a
    # real row, and this is where that row id comes from.
    source_type: SourceType
    row_id: UUID
    document_id: UUID

    # TODO: decide whether `source_type` and the two UUIDs should be excluded
    #  from the JSON schema the model sees (Field(exclude=True) does not do
    #  this; a separate model or `model_dump(include=...)` at the tool boundary
    #  does). Three UUIDs is ~120 characters of pure noise per passage, and a
    #  25-cell grid pays it 25 times. The counter-argument is that the model
    #  never sees this model's schema at all — it sees whatever the tool
    #  returns, serialised — so the cheap fix may just be what the tool returns.


class Citation(BaseModel):
    """One claim in the answer, tied to the passage that supports it.

    The quote is the load-bearing field, and the reason this is not just a list
    of handles. "The citation resolves to a retrieved passage" and "the passage
    actually says this" are different guarantees, and only the second is what
    Driftwood is buying. A verbatim span is mechanically checkable against the
    ledger; a paraphrase is not.
    """

    handle: str = Field(
        description=(
            "The handle of the passage supporting this claim, exactly as it "
            "appeared in the search results."
        )
    )
    quote: str = Field(
        description=(
            "A short verbatim span copied from that passage — the sentence or "
            "clause that supports the claim. Copy it exactly; do not summarise, "
            "reword, correct, or join spans with an ellipsis."
        )
    )


class GroundedAnswer(BaseModel):
    """An answer the corpus supports, with the evidence for each claim."""

    answer: str = Field(
        description=(
            "The answer, in prose an analyst can read. Mark every factual claim "
            "with the handle of its supporting passage in square brackets, e.g. "
            "'Services revenue grew to $96.2B [S3].'"
        )
    )
    citations: list[Citation] = Field(
        description="One entry per handle marked in the answer."
    )
    limitations: str | None = Field(
        default=None,
        description=(
            "What the filings do not establish, when the question asks for more "
            "than they support — causation, intent, or anything outside the "
            "corpus. Leave empty when the filings answer the question as asked."
        ),
    )


class InsufficientEvidence(BaseModel):
    """The corpus cannot answer this, and saying so is the correct answer.

    Not an error path. Brief question 10 asks whether the filings *prove* that
    generative AI improved margins; the honest answer names what is there and
    declines the inference, and the product is worth less if it cannot.
    """

    reason: str = Field(
        description=(
            "What is missing, specifically — which companies, years, or figures "
            "the corpus does not contain, or what inference the filings do not "
            "support. Not 'I could not find anything'."
        )
    )
    searched: list[str] = Field(
        description="The queries tried, so an analyst can judge whether to rephrase."
    )


class ValidatedAnswer(BaseModel):
    """What leaves grounding: the model's answer plus the resolved evidence.

    `cited_passages` comes from the turn's ledger, keyed by the handles the
    model cited — never from the model. architecture.md sketches
    `GroundedAnswer.cited_passages`, which would have the model echo passages
    back into its own output; that is both a few thousand wasted output tokens
    and the one place a hallucination could enter the passage text the analyst
    clicks to verify.
    """

    answer: GroundedAnswer
    cited_passages: list[SourcePassage]
