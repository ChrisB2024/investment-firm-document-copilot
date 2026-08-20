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
from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "data" / "downloads"
MANIFEST = CORPUS_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    """TODO: implement --ticker, --year, --dry-run, --limit."""
    raise NotImplementedError


def ingest_filing(entry: dict, *, dry_run: bool = False) -> None:
    """Extract, chunk, embed, and persist one filing.

    TODO: implement, and make it idempotent. Re-running must replace, not
    duplicate: `accession_number` is unique, so upsert on it and delete the
    document's existing chunks before inserting new ones. Getting this wrong is
    silent — you end up with two copies of every passage and retrieval quietly
    returns duplicates.

    Order matters. Write the document, then chunks, then embeddings. A run that
    dies mid-embedding should leave chunks with a NULL embedding (the column is
    nullable for exactly this reason) so a re-run can finish the job instead of
    starting over.
    """
    raise NotImplementedError


def main() -> None:
    """TODO: implement. Report per-filing counts and a total at the end —
    a silent success over 25 filings tells you nothing about what happened."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
