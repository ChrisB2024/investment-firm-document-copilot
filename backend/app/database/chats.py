"""Thread, message and citation persistence, always keyed to a user.

Every read and every write in this module takes a `user_id` and filters on it.
Not because the caller might forget — because a query that does not filter on it
is indistinguishable from one that does until the day two analysts are logged in
at once. Ownership is a predicate in the WHERE clause, never a check the route
remembered to run.

Two schema gaps block this module and both are cheap now. See the TODOs on
`record_turn`.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.outputs import ValidatedAnswer
from app.database.models import ChatMessage, ChatThread

# A new thread's title before the first turn names it. The AI SDK shows the
# thread list immediately, so "New chat" is what an analyst sees for the second
# or two before the first answer comes back.
DEFAULT_TITLE = "New chat"


async def list_threads(
    session: AsyncSession, user_id: UUID, *, limit: int = 50
) -> list[ChatThread]:
    """This user's threads, newest first.

    TODO: implement. `(user_id, created_at DESC)` is already indexed — that
     index exists for this query specifically (Phase 1).
    """
    raise NotImplementedError


async def create_thread(
    session: AsyncSession, user_id: UUID, *, title: str = DEFAULT_TITLE
) -> ChatThread:
    """Open a thread owned by this user.

    TODO: implement. This is the place the `public.users` row has to exist — see
     the TODO in auth/dependencies.py. An upsert here keyed on the Supabase user
     id is the narrow fix; do it in the same transaction as the insert, or a
     first-time analyst gets an integrity error instead of a thread.
    """
    raise NotImplementedError


async def load_thread(
    session: AsyncSession, thread_id: UUID, user_id: UUID
) -> ChatThread | None:
    """A thread with its messages, or None if it is not this user's.

    None rather than a raise, and one query rather than two. Fetching the thread
    and then comparing `owner_id` in Python is the version that leaks: it is one
    forgotten `if` from serving another analyst's research, and the mistake
    looks like working code. Filtering on `user_id` in the statement means the
    only possible answer to "someone else's thread" is no rows.

    The route turns None into 403 — see the TODO in api/chat.py about why not
    404.

    TODO: implement. Eager-load messages in the same statement; a lazy load on
     an AsyncSession raises rather than lazily loading.
    """
    raise NotImplementedError


async def record_turn(
    session: AsyncSession,
    thread_id: UUID,
    user_id: UUID,
    *,
    question: str,
    answer: ValidatedAnswer,
) -> ChatMessage:
    """Persist one completed turn: the question, the answer, its citations.

    One transaction. A turn that half-commits leaves a question with no answer
    or an answer with no citations, and the second is the dangerous one — an
    assistant message rendering as uncited prose is exactly what the grounding
    contract exists to prevent, arriving through the persistence layer instead
    of through the model.

    Called only after the agent run and the grounding gate have both succeeded,
    so there is no partial-answer state to model. A failed run persists nothing.

    TODO: implement.
      - Ownership first, in the same statement that loads the thread.
      - `sequence` is `UniqueConstraint(thread_id, sequence)`, so derive it from
        the current max rather than a count — a deleted message would make a
        count collide. Two concurrent turns on one thread will race; decide
        whether to care (one analyst, one browser tab, probably not) and say so.
      - `chat_messages.content` holds the answer prose with its `[S3]` markers
        intact, and `parts` holds the AI SDK JSON once that format is pinned.
      - Bump `chat_threads.updated_at`, which the sidebar sorts on.
      - Title the thread from the first question if it is still DEFAULT_TITLE.

    TODO: **`message_citations` cannot store a table citation.** `chunk_id` is
     `NOT NULL` and references `document_chunks`, and 46% of a filing's figures
     appear only in a table — so brief questions 1, 2 and 8 produce citations
     this table physically cannot hold. Needs a migration: nullable `chunk_id`,
     new nullable `table_id` referencing `document_tables`, and a CHECK that
     exactly one is set. Keep both as real foreign keys rather than a
     `(source_type, row_id)` pair: message_citation.py's own docstring says a
     citation that cannot be resolved must be a database error, and dropping to
     an unconstrained pair gives that up.

    TODO: **the handle has nowhere to go.** `content` contains `[S3]`, and
     nothing persisted maps S3 to a row — `citation_index` is 1..N per message
     and cannot be it, because one passage legitimately supports two claims and
     would need two rows with one handle. Either add a `handle` column, or
     renumber the markers in the prose at persist time. Leaning to the column:
     rewriting the model's prose to fit the schema is the tail wagging the dog,
     and a rewrite that goes wrong corrupts the citation mapping silently.

    TODO: **`hydrate` does not return what a citation row needs.** `Passage`
     carries ticker, fiscal_year, form and title; this table requires
     `filing_date`, `company_name`, `page` and `section`. A chunk's section is a
     column, a table's is `table_data->>'section'`, and `_HYDRATE` selects
     neither. Four columns on the two SELECT branches and four fields on
     `Passage`, or a second query per citation here — which would be N+1 on the
     request path to fetch what the first query already had in hand.
    """
    raise NotImplementedError
