"""Database writes for ingestion.

Everything here is idempotent by design. Ingestion gets re-run — after a parser
fix, after a crash, after adding a filing — and the failure mode of getting that
wrong is silent: you end up with two copies of every passage and retrieval
quietly returns duplicates, each with a plausible citation.

Idempotency rests on two constraints already in the schema: `accession_number`
is unique on source_documents, and (document_id, chunk_index) is unique on
document_chunks. Documents upsert on the first; chunks and tables are deleted
and rewritten so a filing that now produces fewer chunks does not leave the
extras behind.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

# Imported from models, not from the individual modules: SQLAlchemy resolves
# relationship targets by name at mapper configuration, so importing only the
# three used here leaves MessageCitation undefined and every mapper fails.
from app.database.models import DocumentChunk, DocumentTable, SourceDocument

# Manifest keys that map straight onto source_documents columns.
_MANIFEST_COLUMNS = (
    "ticker",
    "company_name",
    "cik",
    "form",
    "filing_date",
    "report_date",
    "accession_number",
    "primary_document",
    "source_url",
)


async def fetch_document(session: AsyncSession, accession_number: str) -> SourceDocument | None:
    result = await session.execute(
        select(SourceDocument).where(SourceDocument.accession_number == accession_number)
    )
    return result.scalar_one_or_none()


async def counts(session: AsyncSession, document_id: UUID) -> tuple[int, int]:
    """(chunks, tables) for a document.

    Counted rather than stored on the document: a denormalised count drifts the
    moment a re-run writes a different number, and it is only read for
    reporting.
    """
    chunks = await session.execute(
        select(func.count()).select_from(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
    )
    tables = await session.execute(
        select(func.count()).select_from(DocumentTable).where(
            DocumentTable.document_id == document_id
        )
    )
    return chunks.scalar_one(), tables.scalar_one()


async def upsert_document(
    session: AsyncSession, entry: dict[str, Any], markdown: str, content_hash: str
) -> UUID:
    """Insert or update by accession number, returning the row id.

    ON CONFLICT rather than delete-then-insert: the id has to survive, because
    message_citations point at chunks that point at this document, and
    re-ingesting a filing must not orphan an analyst's existing citations.
    """
    values = {key: entry[key] for key in _MANIFEST_COLUMNS if entry.get(key) is not None}
    values |= {
        "fiscal_year": _fiscal_year(entry),
        "markdown_content": markdown,
        "content_hash": content_hash,
        "ingested_at": datetime.now(UTC),
    }

    statement = insert(SourceDocument).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[SourceDocument.accession_number],
        set_={k: statement.excluded[k] for k in values if k != "accession_number"},
    ).returning(SourceDocument.id)

    return (await session.execute(statement)).scalar_one()


def _fiscal_year(entry: dict[str, Any]) -> int | None:
    """Fiscal year from report_date, not filing_date.

    A filing dated 2025-02-07 reports on fiscal 2024. Using the filing date
    would label every Amazon and Alphabet filing a year late, and "compare 2024
    across companies" would silently mix periods.
    """
    report_date = entry.get("report_date") or entry.get("filing_date")
    return int(str(report_date)[:4]) if report_date else None


async def replace_chunks(session: AsyncSession, document_id: UUID, chunks: list) -> None:
    """Delete this document's chunks, then insert the new set.

    Deleting first is what makes a re-run idempotent: an upsert alone would
    leave behind chunks with indices the new extraction no longer produces.
    """
    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    if not chunks:
        return
    await session.execute(
        insert(DocumentChunk),
        [
            {
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "chunk_metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ],
    )


async def replace_tables(session: AsyncSession, document_id: UUID, tables: list) -> None:
    """Same contract as replace_chunks, for document_tables."""
    await session.execute(
        delete(DocumentTable).where(DocumentTable.document_id == document_id)
    )
    if not tables:
        return
    await session.execute(
        insert(DocumentTable),
        [
            {
                "document_id": document_id,
                "table_index": table.table_index,
                "title": table.title,
                "units": table.units,
                "markdown": table.markdown,
                "table_data": {"rows": table.rows, "section": table.section},
                "source_html_hash": table.source_html_hash,
                "embedding": table.embedding,
            }
            for table in tables
        ],
    )


async def pending_chunks(session: AsyncSession, document_id: UUID) -> list[DocumentChunk]:
    """Chunks still missing an embedding, in index order.

    This is what makes a run resumable: the embedding column is nullable, so a
    crash mid-corpus leaves rows a later run picks up instead of re-embedding
    and re-paying for the whole filing.
    """
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id, DocumentChunk.embedding.is_(None))
        .order_by(DocumentChunk.chunk_index)
    )
    return list(result.scalars())


async def pending_tables(session: AsyncSession, document_id: UUID) -> list[DocumentTable]:
    """Tables still missing an embedding, in index order."""
    result = await session.execute(
        select(DocumentTable)
        .where(DocumentTable.document_id == document_id, DocumentTable.embedding.is_(None))
        .order_by(DocumentTable.table_index)
    )
    return list(result.scalars())
