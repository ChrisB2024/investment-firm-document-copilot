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
from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import Citation, GroundedAnswer, SourcePassage
from app.database.models import SourceDocument
from app.database.session import dispose, engine, session
from app.retrieval.queries import Passage, SourceType
from app.retrieval.retriever import RetrievedPassage

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


# Enough of the corpus that a partition has something to partition. Retrieval
# tests assert on shape — every company covered, every year present — which is
# meaningless against two filings, and so is an agent answer built from them.
MIN_DOCUMENTS = 25


@pytest.fixture
async def corpus():
    """A read-only session over the ingested corpus, or skip.

    At root scope for the same reason the assistant fixtures below are: the
    retrieval tests and the agent's two integration tests both need a session
    over the real corpus, and pytest resolves fixtures per directory.

    The schema existing is not the same as the corpus existing. A fresh clone
    that has run migrations but not ingestion has every table and no rows, and
    a retrieval suite failing there says nothing about the code — so count the
    documents first and skip with the command that fixes it.

    Nothing here writes, so unlike `db` in test_persist.py there is nothing to
    roll back. The engine is still disposed and its cache cleared: `engine()`
    is process-wide and cached while pytest-asyncio gives each test its own
    event loop, so a pooled connection opened in one test fails in the next
    rather than where the mistake was.
    """
    try:
        async with session() as connection:
            documents = await connection.scalar(
                select(func.count()).select_from(SourceDocument)
            )
            if documents < MIN_DOCUMENTS:
                pytest.skip(
                    f"corpus has {documents} documents, needs at least "
                    f"{MIN_DOCUMENTS}; run `uv run python -m ingest.run`"
                )
            yield connection
    finally:
        await dispose()
        engine.cache_clear()


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


# --- assistant and grounding ------------------------------------------------
#
# At root scope rather than in tests/assistant/, because pytest resolves
# fixtures per directory and tests/grounding/ needs the same four. That is the
# mechanism forcing the move, not a third caller: the alternative is two copies
# that drift, and a drifting fixture shows up as a suite that quietly stops
# covering something.

# Row ids and document ids live in separate ranges, so a passage carrying its
# row id through as its document id is visible rather than plausible.
DOCUMENT_OFFSET = 2**64


@pytest.fixture
def passage():
    """Build a `SourcePassage` — what a tool hands the model.

    `title` and `section` are keywords because the interesting tests set exactly
    one of them: a quote must be checkable against every field the model was
    shown, and a test that sets all three cannot tell which field was read.
    """

    def _passage(
        handle: str,
        text: str,
        *,
        title: str | None = None,
        section: str | None = None,
        source_type: SourceType = "chunk",
        n: int = 1,
        ticker: str = "AAPL",
        fiscal_year: int = 2024,
    ) -> SourcePassage:
        return SourcePassage(
            handle=handle,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form="10-K",
            section=section,
            title=title,
            text=text,
            source_type=source_type,
            row_id=UUID(int=n),
            document_id=UUID(int=DOCUMENT_OFFSET + n),
        )

    return _passage


@pytest.fixture
def retrieved():
    """Build a `RetrievedPassage` — what `offer` consumes.

    Deliberately not built on the `passage` fixture. `offer`'s whole job is the
    translation between the two, and a factory that shared their construction
    could not fail when the translation drops a field. It has already dropped
    `title` once.
    """

    def _retrieved(
        n: int,
        *,
        source_type: SourceType = "chunk",
        text: str | None = None,
        title: str | None = None,
        ticker: str = "AAPL",
        fiscal_year: int = 2024,
    ) -> RetrievedPassage:
        return RetrievedPassage(
            passage=Passage(
                source_type=source_type,
                row_id=UUID(int=n),
                document_id=UUID(int=DOCUMENT_OFFSET + n),
                text=text if text is not None else f"body {n}",
                title=title,
                ticker=ticker,
                fiscal_year=fiscal_year,
                form="10-K",
            ),
            score=1 / n,
            rank=n,
            contributions={"vector": n},
        )

    return _retrieved


@pytest.fixture
def deps():
    """A fresh `DocumentAgentDeps` with no database.

    `session=None` is honest rather than lazy: `offer` and `resolve` never touch
    it, so a test that starts to will fail loudly instead of quietly holding a
    mock that returns whatever it was told to. The tool tests that do need a
    session build their own.
    """
    return DocumentAgentDeps(session=None, user_id=UUID(int=1), thread_id=UUID(int=2))


@pytest.fixture
def answer():
    """Build a `GroundedAnswer` from (handle, quote) pairs.

    The default prose marks every handle passed, so a test about fabricated
    handles fails on the fabricated handle and nothing else. `prose` overrides
    it for the tests that are actually about markers.
    """

    def _answer(
        *citations: tuple[str, str],
        prose: str | None = None,
        limitations: str | None = None,
    ) -> GroundedAnswer:
        handles = list(dict.fromkeys(handle for handle, _ in citations))
        return GroundedAnswer(
            answer=(
                prose
                if prose is not None
                else " ".join(f"A claim [{handle}]." for handle in handles)
            ),
            citations=[Citation(handle=h, quote=q) for h, q in citations],
            limitations=limitations,
        )

    return _answer
