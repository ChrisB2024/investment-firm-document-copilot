"""Aggregator that registers every model on `Base.metadata`.

SQLAlchemy only knows about a table once its module has been imported. Alembic
diffs `Base.metadata` against the live database, so a model in a file nobody
imports is invisible to autogenerate — the migration comes out empty with no
error explaining why.

Importing this one module pulls in all of them. `alembic/env.py` and anything
else that needs the full schema should import from here rather than from the
individual modules.
"""

from app.database.base import Base
from app.database.chat_message import ChatMessage
from app.database.chat_thread import ChatThread
from app.database.document_chunk import DocumentChunk
from app.database.document_table import DocumentTable
from app.database.message_citation import MessageCitation
from app.database.source_document import SourceDocument
from app.database.user import User

__all__ = [
    "Base",
    "ChatMessage",
    "ChatThread",
    "DocumentChunk",
    "DocumentTable",
    "MessageCitation",
    "SourceDocument",
    "User",
]
