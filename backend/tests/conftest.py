"""Shared fixtures.

Two rules from backend/claude.md shape everything here:

  - The fast suite (`pytest -m "not integration"`) touches no network and no
    database. That is why the HTML fixtures exist: the real corpus lives in
    data/downloads/ and is gitignored, so tests cannot depend on it.
  - Anything needing live Supabase or OpenAI credentials is marked
    `@pytest.mark.integration` and is expected to be skipped by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_ROOT = Path(__file__).resolve().parents[2] / "data" / "downloads"


@pytest.fixture
def fixture_html():
    """Path to a miniature filing by name, e.g. fixture_html("apple_style").

    Each reproduces one quirk that cost a real bug; see fixtures/README.md.
    """
    def _path(name: str) -> Path:
        path = FIXTURES / f"{name}.htm"
        assert path.exists(), f"missing fixture {path}"
        return path
    return _path


@pytest.fixture
def corpus_filing():
    """Path to a real filing, or skip.

    For tests that genuinely need the full corpus. It is gitignored, so this
    skips rather than fails on a machine that has not run data/download.py.
    """
    def _path(ticker: str, fiscal_year: int) -> Path:
        manifest = CORPUS_ROOT / "manifest.json"
        if not manifest.exists():
            pytest.skip("corpus not downloaded; run `uv run data/download.py`")
        for entry in json.loads(manifest.read_text())["filings"]:
            if entry["ticker"] == ticker and entry["report_date"][:4] == str(fiscal_year):
                path = CORPUS_ROOT / entry["local_path"]
                if not path.exists():
                    pytest.skip(f"{path} listed in manifest but missing")
                return path
        pytest.skip(f"no {ticker} FY{fiscal_year} filing in manifest")
    return _path


@pytest.fixture
def manifest_entry() -> dict:
    """One manifest entry, for tests that need filing metadata without the file.

    Mirrors the real manifest's keys exactly — ingestion maps these onto
    source_documents columns, so a drift here would hide a drift there.

    The accession number is deliberately not a real one. accession_number is
    unique on source_documents and ingestion upserts on it, so an integration
    test writing under a genuine number would overwrite that filing's row.
    Tests that write should still override this with something unique to the
    run; see the `entry` fixture in test_persist.py.
    """
    return {
        "ticker": "AAPL",
        "cik": "0000320193",
        "form": "10-K",
        "filing_date": "2024-11-01",
        "report_date": "2024-09-28",
        "accession_number": "9999999999-99-999999",
        "primary_document": "aapl-20240928.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/x/aapl-20240928.htm",
        "local_path": "2024/aapl_10-k_2024-11-01_9999999999-99-999999.htm",
    }


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Replace the OpenAI client with a deterministic stub.

    Working infrastructure, not a TODO: the monkeypatch target is fiddly and
    getting it wrong makes a test spend money instead of failing.

    Install it inside a test, then assert on the returned list, which records
    the size of each batch actually requested:

        calls = fake_embeddings()
        await embed_chunks(chunks)
        assert calls == [100, 100, 50]
        assert calls.texts[0][0] == chunks[0].text

    Each vector is filled with its own position in the batch, so a test can
    prove a response was mapped back to the right item.
    """
    class _Calls(list):
        """Batch sizes, plus the texts of each batch on `.texts`.

        A list subclass so `calls == [100, 100, 50]` keeps working; the texts
        are what proves *which* attribute of an item was sent to embed.
        """
        def __init__(self) -> None:
            super().__init__()
            self.texts: list[list[str]] = []

    def _install(*, dimensions: int = 1536, shuffle: bool = False, drop: int = 0) -> _Calls:
        # Patched on app.embeddings, which owns the client — ingest.embed only
        # re-exports embed_texts, and patching there would leave the real client
        # in place and spend money.
        from app import embeddings

        # Bound here because `create` takes its own `dimensions` argument, which
        # would otherwise shadow this one and make the parameter dead — the stub
        # would always hand back exactly the width the caller asked for, and a
        # dimension-drift test could never fail.
        returned_dimensions = dimensions
        calls = _Calls()

        class _Item:
            def __init__(self, index: int, embedding: list[float]) -> None:
                self.index = index
                self.embedding = embedding

        class _Response:
            def __init__(self, data: list[_Item]) -> None:
                self.data = data

        class _Embeddings:
            @staticmethod
            async def create(*, model: str, dimensions: int, input: list[str]) -> _Response:
                calls.append(len(input))
                calls.texts.append(list(input))
                data = [_Item(i, [float(i)] * returned_dimensions)
                        for i in range(len(input) - drop)]
                if shuffle:
                    # Deliberately not the order sent. embed_texts must sort by
                    # index, or every vector lands on the wrong passage.
                    data.reverse()
                return _Response(data)

        class _Client:
            embeddings = _Embeddings()

        monkeypatch.setattr(embeddings, "_client", _Client())
        return calls

    return _install
