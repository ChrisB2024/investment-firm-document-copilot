"""Ingestion CLI: read the manifest, write documents and chunks to Supabase.

    uv run python -m ingest.run              # everything in the manifest
    uv run python -m ingest.run --ticker AAPL
    uv run python -m ingest.run --dry-run    # parse and chunk, write nothing

The corpus lives in data/downloads/ with a manifest.json whose entries map
one-to-one onto source_documents columns: ticker, cik, form, filing_date,
report_date, accession_number, primary_document, source_url, local_path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.database.session import dispose, session
from ingest.chunk import chunk_filing
from ingest.embed import embed_chunks, embed_tables
from ingest.extract import extract
from ingest.persist import (
    counts,
    fetch_document,
    pending_chunks,
    pending_tables,
    replace_chunks,
    replace_tables,
    upsert_document,
)

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "data" / "downloads"
MANIFEST = CORPUS_ROOT / "manifest.json"

# The manifest carries tickers, not names. Chunk metadata is what retrieval
# filters on and what the citation UI renders, and an analyst asks about
# "Alphabet", not "GOOGL".
COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {number}")
    return number

def parse_args() -> argparse.Namespace:
    """Supports --ticker, --year, --dry-run, --limit."""
    parser = argparse.ArgumentParser(
        prog="python -m ingest.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=str.upper,
        help="Only ingest filings for this ticker (case-insensitive).",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Only ingest filings whose report_date falls in this fiscal year. "
             "This is the fiscal period, not the filing date — a FY2024 10-K "
             "may carry a 2025 filing_date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract, chunk, and report counts without writing to Supabase.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Stop after this many filings, applied after --ticker/--year.",
    )
    return parser.parse_args()


EXTRACTOR_VERSION = "ingest-1"

def content_hash(path: Path) -> str:
    """Fingerprint of the source file plus the extractor that read it.

    Bumping EXTRACTOR_VERSION invalidates every document, which is what you want
    when a parsing fix changes what the same HTML produces.
    """
    digest = hashlib.sha256(EXTRACTOR_VERSION.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _metadata(entry: dict[str, Any]) -> dict[str, Any]:
    """Filing-level metadata copied onto every chunk.

    fiscal_year comes from report_date, not filing_date: a filing dated
    2025-02-07 reports on fiscal 2024, and using the filing date would label
    every Amazon and Alphabet filing a year late.
    """
    report_date = entry.get("report_date") or entry["filing_date"]
    return {
        "ticker": entry["ticker"],
        "company": COMPANY_NAMES.get(entry["ticker"], entry["ticker"]),
        "form": entry["form"],
        "filing_date": entry["filing_date"],
        "fiscal_year": int(str(report_date)[:4]),
        "accession_number": entry["accession_number"],
    }


async def ingest_filing(entry: dict, *, dry_run: bool = False) -> str:
    """Extract, chunk, embed, and persist one filing.

    Must be idempotent. Re-running must replace, not
    duplicate: `accession_number` is unique, so upsert on it and delete the
    document's existing chunks before inserting new ones. Getting this wrong is
    silent — you end up with two copies of every passage and retrieval quietly
    returns duplicates.

    Order matters. Write the document, then chunks, then embeddings. A run that
    dies mid-embedding should leave chunks with a NULL embedding (the column is
    nullable for exactly this reason) so a re-run can finish the job instead of
    starting over.
    """
    path = CORPUS_ROOT / entry["local_path"]
    digest = content_hash(path)
    label = f"{entry['ticker']} FY{_metadata(entry)['fiscal_year']}"

    async with session() as conn:
        existing = await fetch_document(conn, entry["accession_number"])
        resumable = existing is not None and existing.content_hash == digest

        if resumable:
            # Unchanged since the last run: skip parsing and go straight to
            # whatever embedding did not finish.
            document_id = existing.id
            chunk_count, table_count = await counts(conn, document_id)
        else:
            filing = extract(str(path))
            chunks = chunk_filing(filing.sections, _metadata(entry))
            chunk_count, table_count = len(chunks), len(filing.tables)

            if dry_run:
                return (f"{label}: {len(filing.sections)} sections, "
                        f"{chunk_count} chunks, {table_count} tables (dry run)")

            document_id = await upsert_document(conn, entry, filing.markdown, digest)
            await replace_chunks(conn, document_id, chunks)
            await replace_tables(conn, document_id, filing.tables)
            # Committed before embedding so a crash in the paid half leaves the
            # parsed corpus durable and a re-run resumes instead of re-parsing.
            await conn.commit()

        if dry_run:
            return f"{label}: unchanged, nothing to do (dry run)"

        chunks_todo = await pending_chunks(conn, document_id)
        tables_todo = await pending_tables(conn, document_id)
        if chunks_todo:
            await embed_chunks(chunks_todo)
        if tables_todo:
            await embed_tables(tables_todo)
        # The rows are attached to this session, so setting .embedding is the
        # write; the commit flushes it.
        await conn.commit()

    return (f"{label}: {chunk_count} chunks, {table_count} tables, "
            f"embedded {len(chunks_todo)} chunks + {len(tables_todo)} tables"
            f"{' (resumed)' if resumable else ''}")



def _select(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.ticker:
        entries = [e for e in entries if e["ticker"].upper() == args.ticker]
    if args.year:
        entries = [e for e in entries
                   if int(str(e.get("report_date") or e["filing_date"])[:4]) == args.year]
    return entries[: args.limit] if args.limit else entries


async def _run(args: argparse.Namespace) -> int:
    entries = _select(json.loads(MANIFEST.read_text())["filings"], args)
    if not entries:
        print("no filings matched")
        return 1

    print(f"{len(entries)} filing(s)" + (" — dry run, nothing will be written" if args.dry_run else ""))
    failures = 0
    try:
        for entry in entries:
            try:
                print("  " + await ingest_filing(entry, dry_run=args.dry_run))
            except Exception as error:  # noqa: BLE001 — per-filing boundary
                # One bad filing should not abandon the other 24. The run is
                # resumable and the exit code is non-zero, so nothing is hidden;
                # a narrower except here would just be a guess at which parser
                # or network failure a new filing might produce.
                failures += 1
                print(f"  {entry['ticker']} {entry['accession_number']}: FAILED "
                      f"{type(error).__name__}: {error}")
    finally:
        await dispose()

    print(f"done: {len(entries) - failures} ok, {failures} failed")
    return 1 if failures else 0


def main() -> None:
    """Reports per-filing counts and a total at the end — a silent success over
    25 filings tells you nothing about what happened."""
    raise SystemExit(asyncio.run(_run(parse_args())))


if __name__ == "__main__":
    main()
