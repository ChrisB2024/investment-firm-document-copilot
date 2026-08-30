"""normalise section headings

Revision ID: 6ac66b62405c
Revises: e1f6c1db7782

`section` became a user-visible field when `page` was dropped: it is now the
anchor a citation gives an analyst. It was not consistent enough for that job —
65 distinct strings across 23 Items, because roughly half the filers shout.
"Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA" next to "Item 15.
Exhibits and Financial Statement Schedules" in the same answer reads like a bug
in the product rather than a difference between filers.

**Case only, never wording.** Item 6 is "Selected Financial Data" in the 2021
filings and "[Reserved]" in the later ones, because the SEC dropped the
requirement mid-corpus; Item 14 is "Principal Accountant Fees" for four filers
and "Principal Accounting Fees" for one. Mapping Items onto canonical SEC
titles would collapse those, and a citation would then claim a heading the
filing does not contain — which is the one thing this product must not do.
Normalising case alone takes 65 strings to 41.

Also fixes one extraction artefact: "Item 16. FORM 10-K SUMMARYNone", where an
Item with an empty body picked up the literal "None" (MSFT FY2025, one chunk).

**`text` is rewritten alongside `section`, and has to be.** `ingest.chunk`
prefixes every chunk's text with its heading — all 2,321 of them — and
`retriever.neighbours` strips it again with
`removeprefix(f"{row.section}\\n\\n")`. Updating `section` alone would silently
turn that strip into a no-op, and every citation window would carry a
duplicated heading. It is also the heading the *model* reads, so leaving it
shouting would fix the citation and not the passage.

Consequence worth stating: a chunk's stored `embedding` was computed over the
old text and is now very slightly stale — one heading line of a ~700-token
chunk, and only for the filers who shouted. `search_vector` is a generated
column and refreshes itself. Re-embedding the corpus costs about $0.03 and is
the clean fix if it ever matters; it does not belong in a migration, which
should not make network calls.

The transform is duplicated from `ingest.extract.normalise_heading` rather than
imported. A migration has to keep producing the same result years after the
application module it was written against has moved on.
"""

import re
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '6ac66b62405c'
down_revision: str | Sequence[str] | None = 'e1f6c1db7782'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SMALL_WORDS = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from",
    "in", "nor", "of", "on", "or", "the", "to", "with",
})
_ITEM_LABEL = re.compile(r"^(Item\s+\d+[A-Za-z]?\.)\s*(.*)$", re.IGNORECASE)
_WORD_START = re.compile(r"(?<![A-Za-z'’])[A-Za-z]+")


def _recase(word: str, *, first: bool) -> str:
    if word != word.upper():
        return word
    lowered = word.lower()
    if not first and lowered.strip("().,") in _SMALL_WORDS:
        return lowered
    return _WORD_START.sub(lambda m: m.group(0).capitalize(), lowered)


def _normalise(heading: str) -> str:
    heading = heading.strip()
    if heading.endswith("None") and not heading.endswith(" None"):
        heading = heading[: -len("None")].rstrip()
    match = _ITEM_LABEL.match(heading)
    if not match:
        return heading
    label, title = match.group(1), match.group(2)
    label = re.sub(
        r"(\d+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        label[0].upper() + label[1:].lower(),
    )
    words = title.split()
    if not words:
        return label
    return f"{label} " + " ".join(
        _recase(word, first=i == 0) for i, word in enumerate(words)
    )


def upgrade() -> None:
    bind = op.get_bind()

    # One statement per *distinct* heading rather than per row: 65 updates
    # instead of 4,045, and the set of headings is what the transform acts on.
    chunk_sections = bind.execute(
        sa.text(
            "SELECT DISTINCT section FROM document_chunks WHERE section IS NOT NULL"
        )
    ).scalars()
    for old in chunk_sections:
        new = _normalise(old)
        if new == old:
            continue
        # The heading is a prefix of `text` for every chunk, so replacing the
        # first len(old) characters rewrites it without touching the body.
        # `substring(... from N)` is 1-based, hence the +1.
        bind.execute(
            sa.text(
                "UPDATE document_chunks "
                "   SET section = :new, "
                "       text = :new || substring(text from :offset) "
                " WHERE section = :old"
            ),
            {"new": new, "old": old, "offset": len(old) + 1},
        )

    table_sections = bind.execute(
        sa.text(
            "SELECT DISTINCT table_data->>'section' FROM document_tables "
            "WHERE table_data->>'section' IS NOT NULL"
        )
    ).scalars()
    for old in table_sections:
        new = _normalise(old)
        if new == old:
            continue
        # A table's section lives in jsonb, and its markdown carries no heading
        # to keep in step — `embed_text` composes the two at read time.
        bind.execute(
            sa.text(
                "UPDATE document_tables "
                # CAST rather than `:new::text`: SQLAlchemy's text() cannot
                # tell the postgres cast operator from a parameter name.
                "   SET table_data = "
                "       jsonb_set(table_data, '{section}', to_jsonb(CAST(:new AS text))) "
                " WHERE table_data->>'section' = :old"
            ),
            {"new": new, "old": old},
        )


def downgrade() -> None:
    """Deliberately does nothing.

    The original casing is not recoverable: "Item 1A. Risk Factors" is what two
    different filers wrote, one shouting and one not, and this migration merged
    them. Restoring would mean re-extracting from the source HTML, which is
    `ingest.run`'s job and not a downgrade's.

    A no-op rather than a raise, because nothing downstream depends on the old
    casing — the column is read, displayed and joined on equality, and the join
    is `n.section = a.section` within one document, which holds either way.
    """
