"""Tests for ingest.run — the CLI and the per-filing orchestration."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from ingest import run


def _parse(monkeypatch, argv: list[str]):
    """parse_args() against a fake command line.

    Driven through argparse rather than by building a Namespace, because the
    behaviour under test lives in the parser: `type=str.upper` is what makes
    --ticker case-insensitive, and `_positive_int` is what rejects --limit 0.
    """
    monkeypatch.setattr(sys, "argv", ["python -m ingest.run", *argv])
    return run.parse_args()


def _entry(**overrides) -> dict:
    entry = {
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2024-11-01",
        "report_date": "2024-09-28",
        "accession_number": "0000320193-24-000123",
        "local_path": "apple_style.htm",
    }
    return {**entry, **overrides}


@asynccontextmanager
async def _fake_session(commits: list):
    class _Conn:
        async def commit(self) -> None:
            commits.append(True)

    yield _Conn()


def _forbidden(name: str):
    async def _boom(*args, **kwargs):
        raise AssertionError(f"{name} was called")
    return _boom


def test_ticker_filter_is_case_insensitive(monkeypatch):
    """--ticker aapl selects the AAPL filings.

    Both sides are normalised — the parser upper-cases the argument and
    _select upper-cases the entry — so the test covers a lower-case manifest
    value too.
    """
    entries = [_entry(), _entry(ticker="msft"), _entry(ticker="MSFT")]

    for flag in ("aapl", "AAPL", "AaPl"):
        args = _parse(monkeypatch, ["--ticker", flag])
        assert [e["ticker"] for e in run._select(entries, args)] == ["AAPL"]

    args = _parse(monkeypatch, ["--ticker", "msft"])
    assert len(run._select(entries, args)) == 2


def test_limit_rejects_zero_and_negatives(monkeypatch):
    """argparse errors on --limit 0.

    Failing at parse time beats a run that silently does nothing: `entries[:0]`
    is empty, so a zero limit would print "no filings matched" and exit 1,
    which reads like a bad --ticker rather than a bad --limit.
    """
    for bad in ("0", "-1", "abc"):
        with pytest.raises(SystemExit) as excinfo:
            _parse(monkeypatch, ["--limit", bad])
        assert excinfo.value.code == 2

    assert _parse(monkeypatch, ["--limit", "3"]).limit == 3
    assert _parse(monkeypatch, []).limit is None


def test_year_filters_on_fiscal_year_not_filing_date(monkeypatch):
    """--year 2024 selects on report_date, not filing_date.

    Half the corpus files the following calendar year; filtering on
    filing_date would return the wrong filings for four of the five companies.
    Both halves of that matter, so the fixture holds a filing that a
    filing_date filter would wrongly *include* as well as one it would miss.
    """
    fiscal_2024 = _entry(ticker="AMZN", filing_date="2025-02-07", report_date="2024-12-31")
    fiscal_2023 = _entry(ticker="AMZN", filing_date="2024-02-02", report_date="2023-12-31",
                         accession_number="0001018724-24-000008")
    entries = [fiscal_2024, fiscal_2023, _entry()]

    selected = run._select(entries, _parse(monkeypatch, ["--year", "2024"]))

    assert fiscal_2024 in selected, "missed a FY2024 filing filed in 2025"
    assert fiscal_2023 not in selected, "matched on filing_date; that is FY2023"
    assert len(selected) == 2

    # No report_date at all falls back to filing_date rather than crashing.
    undated = _entry(ticker="GOOGL", accession_number="x")
    del undated["report_date"]
    assert run._select([undated], _parse(monkeypatch, ["--year", "2024"])) == [undated]


def test_content_hash_changes_with_extractor_version(monkeypatch, tmp_path):
    """The same bytes hash differently when EXTRACTOR_VERSION changes.

    That is the mechanism for invalidating the corpus after a parser fix: the
    same HTML now produces different chunks, so every document must be re-read.
    Without it a fix to extract.py leaves every filing looking unchanged and
    the corpus silently keeps the old, wrong output.
    """
    path = tmp_path / "filing.htm"
    path.write_text("<html><body>Item 1. Business</body></html>")
    other = tmp_path / "other.htm"
    other.write_text("<html><body>Item 1. Different</body></html>")

    original = run.content_hash(path)

    assert run.content_hash(path) == original, "not stable for identical bytes"
    assert run.content_hash(other) != original, "ignores the file's contents"

    monkeypatch.setattr(run, "EXTRACTOR_VERSION", "ingest-2")
    assert run.content_hash(path) != original


async def test_dry_run_writes_nothing(monkeypatch, fixture_html, manifest_entry):
    """--dry-run reaches no persistence function and makes no embedding call.

    Note it still opens a session and reads fetch_document: a dry run needs a
    live database to tell a new filing from an unchanged one. That is
    deliberate, but it means --dry-run is not an offline mode.
    """
    monkeypatch.setattr(run, "CORPUS_ROOT", fixture_html("apple_style").parent)
    monkeypatch.setattr(run, "session", lambda: _fake_session([]))

    async def no_document(conn, accession_number):
        return None
    monkeypatch.setattr(run, "fetch_document", no_document)

    for name in ("upsert_document", "replace_chunks", "replace_tables",
                 "pending_chunks", "pending_tables", "counts",
                 "embed_chunks", "embed_tables"):
        monkeypatch.setattr(run, name, _forbidden(name))

    entry = {**manifest_entry, "local_path": "apple_style.htm"}
    message = await run.ingest_filing(entry, dry_run=True)

    assert message == "AAPL FY2024: 5 sections, 5 chunks, 1 tables (dry run)"


async def test_changed_content_hash_forces_re_extraction(
    monkeypatch, fixture_html, manifest_entry
):
    """A stored document whose content_hash differs is parsed again.

    The unchanged-filing test below passes just as well if the hash comparison
    is dropped and mere existence is treated as resumable, because there the
    hashes match either way. This is the half that pins it — and it is the
    whole point of content_hash. After a parser fix every document must be
    re-read; a resume that trusted existence alone would never re-parse
    anything, and the corpus would keep the old, wrong output forever.
    """
    monkeypatch.setattr(run, "CORPUS_ROOT", fixture_html("apple_style").parent)

    class _Stale:
        id = uuid4()
        content_hash = "a hash from before the parser was fixed"

    async def stale(conn, accession_number):
        return _Stale
    monkeypatch.setattr(run, "fetch_document", stale)
    monkeypatch.setattr(run, "counts", _forbidden("counts"))

    written: dict = {}

    async def upsert(conn, entry, markdown, digest):
        written["markdown"] = markdown
        written["digest"] = digest
        return _Stale.id
    async def replace_chunks(conn, document_id, chunks):
        written["chunks"] = len(chunks)
    async def replace_tables(conn, document_id, tables):
        written["tables"] = len(tables)
    monkeypatch.setattr(run, "upsert_document", upsert)
    monkeypatch.setattr(run, "replace_chunks", replace_chunks)
    monkeypatch.setattr(run, "replace_tables", replace_tables)

    async def nothing_pending(conn, document_id):
        return []
    monkeypatch.setattr(run, "pending_chunks", nothing_pending)
    monkeypatch.setattr(run, "pending_tables", nothing_pending)
    monkeypatch.setattr(run, "embed_chunks", _forbidden("embed_chunks"))
    monkeypatch.setattr(run, "embed_tables", _forbidden("embed_tables"))

    commits: list = []
    monkeypatch.setattr(run, "session", lambda: _fake_session(commits))

    entry = {**manifest_entry, "local_path": "apple_style.htm"}
    message = await run.ingest_filing(entry)

    assert written["chunks"] == 5
    assert written["tables"] == 1
    assert written["digest"] != _Stale.content_hash
    assert "Item 1. Business" in written["markdown"]
    assert "resumed" not in message
    assert commits == [True, True], "parsed corpus must commit before embedding"


async def test_unchanged_filing_skips_extraction(monkeypatch, fixture_html, manifest_entry):
    """A filing whose stored content_hash matches is not re-extracted, and
    embedding still resumes for the items missing one.

    This is what makes a run that died in the paid half cheap to finish:
    parsing is skipped, but the embeddings that were never written are still
    filled in. Skipping the resume half instead would leave chunks with a NULL
    embedding that no later run ever revisits.
    """
    corpus = fixture_html("apple_style").parent
    monkeypatch.setattr(run, "CORPUS_ROOT", corpus)
    digest = run.content_hash(corpus / "apple_style.htm")

    def re_extracted(*args, **kwargs):
        raise AssertionError("re-extracted a filing whose content_hash matched")
    monkeypatch.setattr(run, "extract", re_extracted)
    monkeypatch.setattr(run, "chunk_filing", re_extracted)
    for name in ("upsert_document", "replace_chunks", "replace_tables"):
        monkeypatch.setattr(run, name, _forbidden(name))

    class _Document:
        id = uuid4()
        content_hash = digest

    async def existing(conn, accession_number):
        return _Document
    monkeypatch.setattr(run, "fetch_document", existing)

    async def stored_counts(conn, document_id):
        return (5, 1)
    monkeypatch.setattr(run, "counts", stored_counts)

    async def two_chunks(conn, document_id):
        return ["chunk", "chunk"]
    async def one_table(conn, document_id):
        return ["table"]
    monkeypatch.setattr(run, "pending_chunks", two_chunks)
    monkeypatch.setattr(run, "pending_tables", one_table)

    embedded: list[tuple[str, int]] = []
    async def record_chunks(items):
        embedded.append(("chunks", len(items)))
    async def record_tables(items):
        embedded.append(("tables", len(items)))
    monkeypatch.setattr(run, "embed_chunks", record_chunks)
    monkeypatch.setattr(run, "embed_tables", record_tables)

    commits: list = []
    monkeypatch.setattr(run, "session", lambda: _fake_session(commits))

    message = await run.ingest_filing(entry := {**manifest_entry, "local_path": "apple_style.htm"})

    assert embedded == [("chunks", 2), ("tables", 1)]
    assert commits == [True], "embeddings were set but never committed"
    assert message == (
        f"{entry['ticker']} FY2024: 5 chunks, 1 tables, "
        "embedded 2 chunks + 1 tables (resumed)"
    )
