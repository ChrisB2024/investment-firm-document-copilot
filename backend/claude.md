# Backend — agent notes

This is the FastAPI service for Document Copilot. Read [../claude.md](../claude.md) first — universal building rules live there. This file adds backend-specific conventions.

## Stack

- Python 3.12+
- FastAPI + uvicorn
- Pydantic v2 + pydantic-settings
- `httpx` for outbound HTTP
- `pytest` for tests
- Supabase Python client (DB + auth)
- SQLAlchemy models + Alembic migrations for database schema changes
- OpenAI SDK for LLM & embeddings
- Supabase `pgvector` for semantic search and Postgres full-text search for keyword retrieval. Hybrid search should run vector and full-text queries separately, then fuse ranked results in Python with Reciprocal Rank Fusion.
- `structlog` for logging
- `uv` for dependency + project management

## Dependency policy

See universal policy in [../claude.md](../claude.md). Backend-specific:

- **Prefer stdlib:** `pathlib`, `datetime`, `uuid`, `enum`, `dataclasses`, `asyncio`, `collections`, `itertools`, `json`, `urllib`.
- **Not OK without justification:** `python-dateutil`, `toolz`, `funcy`, `more-itertools`, small JSON/string micro-libs, "ergonomic" wrappers on top of declared SDKs.
- Dev deps (test/lint/build) have a looser bar but still pick widely-used, low-footprint tools (`pytest`, `ruff`, `httpx`).

## Layout (to be created during build)

```text
backend/
├── alembic/
│   ├── env.py           # Imports app database metadata for autogenerate
│   └── versions/        # Reviewed migration files
├── alembic.ini
├── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── config.py        # Pydantic settings — single source of truth for env
│   ├── api/             # FastAPI routers (chat, ingest, auth)
│   ├── auth/            # Supabase JWT verification + current user dependency
│   ├── chat/            # turn orchestration, AI SDK message conversion, streaming
│   ├── assistant/       # PydanticAI agent, deps, outputs, instructions
│   ├── retrieval/       # pgvector/full-text queries, RRF fusion, source passage lookup
│   ├── grounding/       # citation validation and answer grounding checks
│   ├── database/        # one model per file + models.py aggregator, Supabase client, query helpers
│   └── prompts/         # prompt/instruction templates if not colocated with assistant
├── ingest/              # one-off ingestion scripts (Markdown extraction, chunking, embedding, Supabase writes)
├── tests/
└── pyproject.toml
```

## Code style (backend-specific)

- **Type hints on public functions and module-level things.** Don't annotate every local.
- **Async by default in request-path code.** Don't run blocking I/O on the event loop. Tempfile + small synchronous file reads are OK (they're fast); network calls must be async.
- **Use `async def` for all route handlers** and any I/O service function.
- **Validate at boundaries only.** HTTP input is validated by Pydantic models. External API responses are validated when parsed. Internal callers are trusted.

## Configuration

- `app.config.settings` is the single source of truth. Import settings where needed; never call `os.getenv` in app code, never call `load_dotenv`.
- If a third-party SDK reads `os.environ` directly, add the mirror in `config.py` — don't sprinkle `setdefault` elsewhere.
- Fail fast on startup when required env vars are missing.

## Database models

- **One model per file**, named after the model (`user.py` -> `User`,
  `document_chunk.py` -> `DocumentChunk`). `base.py` holds the shared `Base`.
- `models.py` imports all of them and re-exports. SQLAlchemy only registers a table
  once its module is imported, so a model no one imports is invisible to Alembic
  autogenerate — the migration comes out empty with no error saying why. Anything
  needing the full schema (`alembic/env.py`, tests) imports from `models.py`.
- Adding a model means adding the file *and* the import in `models.py`.
- `supabase_auth.py` declares a stub for Supabase's `auth.users` so cross-schema
  foreign keys can compile. `alembic/env.py` filters the `auth` schema out of
  autogenerate. `metadata.create_all()` does *not* honour that filter — pass
  `tables=[t for t in Base.metadata.sorted_tables if t.schema != "auth"]`.

## Database migrations

- Alembic is the source of truth for schema changes. Do not change production tables manually in the Supabase dashboard.
- SQLAlchemy models describe normal tables and columns. Alembic autogenerate creates candidate migrations, but every generated migration must be reviewed before applying.
- Supabase/Postgres-specific features belong in explicit migration operations: `create extension vector`, generated `tsvector` columns, HNSW/GIN indexes, RLS enablement, and RLS policies.
- Alembic must use a **session-level** connection: the session pooler (port `5432`) or the direct connection (`db.<ref>.supabase.co:5432`). Never the transaction pooler (port `6543`) — migrations, extension setup, and index creation need session-level behaviour.
- Run migrations from `backend/` with `uv run alembic upgrade head`.

## Tests

- **Prefer unit over integration.** Mock at the service boundary.
- Fast suite (`pytest -m "not integration"`) must stay green and hit no network / no DB.
- Integration tests go behind `@pytest.mark.integration` and may require live OpenAI / Supabase credentials.
- Tests live next to what they test (`retrieval/retriever.py` → `tests/retrieval/test_retriever.py`).
- Required test coverage: ingestion logic, retrieval, citation extraction, grounding enforcement.

## Anti-patterns (rejected)

- `os.getenv` / `load_dotenv` in modules.
- Wrapping FastAPI responses in custom envelope classes.
- Over-catching `Exception` just to log and re-raise; let it propagate.
- Shared state through globals instead of FastAPI `app.state` or DI.
- Silent fallbacks that hide real config errors.
- Mocking the LLM in unit tests without also testing the grounding contract — the prompt is the product.
