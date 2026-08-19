# Build todos

Sequenced so each phase ends in something you can run and check. Do not start a phase until the
previous one's **Done when** holds.

Reference: [architecture.md](architecture.md) for the target shape, [client-brief.md](client-brief.md)
for what the answers must satisfy.

---

## Phase 0 — Foundations

Goal: both services boot, config fails loudly, Supabase project exists.

Scaffolded — files exist with TODOs, implementation is yours:

- [x] Supabase project created; `backend/.env` + `frontend/.env` filled.
- [x] Backend deps installed; `[build-system]` + hatch wheel target added, so
      `from app...` resolves from any cwd.
- [x] Frontend scaffolded: Vite + React 19 + TS strict, Tailwind v4, shadcn (radix/nova),
      React Router, `@/*` alias. `pnpm build` and `pnpm lint` pass.
- [ ] `backend/app/config.py` — fill in the four TODOs: `model_config`, `allowed_origins`
      parsing, transaction-pooler guard, `alembic_url` driver, and the settings singleton.
- [ ] `backend/app/main.py` — fill in CORS wiring, `/health`, structlog setup.
- [ ] `frontend/src/lib/env.ts` — implement `loadEnv()`; report all missing vars at once.
- [ ] Import `env` somewhere real so a missing var actually fails the boot — an unimported
      module never runs.
- [ ] Fix stale `../AGENTS.md` links in `claude.md` / `backend/claude.md` / `frontend/claude.md`
      (9 references, incl. `README.md`).

**Done when:** `uv run uvicorn app.main:app --reload` serves `/health`, `pnpm dev` serves a page,
and deleting a required env var crashes each on boot.

---

## Phase 1 — Schema

Goal: the tables retrieval needs exist in Supabase, created by Alembic.

- [ ] `app/database/models.py` — SQLAlchemy models: `profiles`, `chat_threads`, `chat_messages`,
      `message_citations`, `source_documents`, `document_chunks`.
- [ ] ~~`uv run alembic init alembic`~~ (done — still stock boilerplate); point `env.py` at the models' metadata and
      `settings.database_url` (session pooler on `5432` or direct — **never** the transaction pooler on `6543`).
- [ ] First migration, hand-edited after autogenerate:
  - [ ] `create extension if not exists vector`
  - [ ] `document_chunks.embedding vector(1536)`
  - [ ] generated `tsvector` column over chunk text
  - [ ] HNSW index on `embedding`, GIN index on the tsvector and on `metadata`
  - [ ] unique constraint on `(document_id, chunk_index)`
- [ ] Decide RLS now: backend uses the service-role key, so either enable RLS with deny-all and rely
      on backend scoping, or leave it off deliberately. Write the choice in a comment.

**Done when:** `uv run alembic upgrade head` against Supabase succeeds from a clean DB, and
`upgrade head` on an already-migrated DB is a no-op.

---

## Phase 2 — Ingestion

Goal: the 25 downloaded 10-Ks are in Postgres as embedded, searchable chunks. This is the phase that
decides whether the product works — spend the time here.

- [ ] `ingest/extract.py` — SEC HTML → normalized Markdown. Drop nav/XBRL noise, keep tables in a
      readable form, keep a byte offset per block so citations can point back.
- [ ] `ingest/chunk.py` — split Markdown into ~500–800 token chunks with overlap, never splitting a
      table mid-row. Carry metadata: ticker, company, filing type, filing date, fiscal year,
      accession number, section heading, chunk index, source offsets.
- [ ] `ingest/embed.py` — batch chunks to OpenAI embeddings (batch, don't loop one-by-one); retry on
      rate limit.
- [ ] `ingest/run.py` — CLI: read `data/downloads/manifest.json`, insert `source_documents` +
      `document_chunks`, idempotent by accession number (re-running replaces, not duplicates).
- [ ] Tests: chunk boundaries, metadata propagation, idempotency. No network.

**Done when:** all 25 filings ingested; a spot check on Apple FY2024 shows chunks whose text you can
find in the original filing, with correct ticker/year/section metadata.

---

## Phase 3 — Retrieval

Goal: given a question, the right passages come back — provable before any LLM is involved.

- [ ] `app/retrieval/queries.py` — pgvector cosine query and Postgres full-text query, each bounded
      (top-k, optional ticker/year filters).
- [ ] `app/retrieval/fusion.py` — Reciprocal Rank Fusion over the two ranked lists.
- [ ] `app/retrieval/retriever.py` — embed query → both searches → fuse → hydrate chunks + document
      metadata + neighbouring chunks.
- [ ] A scratch CLI that prints top-k passages for a question, so you can eyeball quality.
- [ ] Run the 10 questions in [client-brief.md](client-brief.md) through it. Note which fail and why
      (missing chunk? bad chunking? lexical vs semantic?) — fix in Phase 2, not with a prompt.
- [ ] Tests: RRF ordering with known inputs, filter application, empty-result handling.

**Done when:** for each of the 10 brief questions, the passages needed to answer it appear in the
top ~10 results.

---

## Phase 4 — Agent + grounding

Goal: a typed answer that cannot cite what wasn't retrieved.

- [ ] `app/assistant/outputs.py` — `GroundedAnswer`, `Citation`, `SourcePassage`.
- [ ] `app/assistant/deps.py` — `DocumentAgentDeps` (user id, thread id, retriever, validator).
- [ ] `app/assistant/instructions.md` — the product contract: answer only from retrieved passages,
      cite every factual claim, say plainly when the corpus doesn't support an answer, no investment
      advice.
- [ ] `app/assistant/agent.py` — PydanticAI agent with tools `search_filings`, `read_chunk`,
      `read_surrounding_chunks`. No agent-authored SQL.
- [ ] `app/grounding/validator.py` — every citation resolves to a chunk retrieved this turn;
      violation = controlled failure, never a polished unsourced answer.
- [ ] Tests: validator rejects fabricated chunk ids; a no-evidence question yields the refusal path.

**Done when:** brief question 10 ("does the corpus prove gen-AI improved margins?") produces a
refusal, and questions 1–9 produce answers whose citations all validate.

---

## Phase 5 — API surface

Goal: the agent is reachable over HTTP, scoped to a signed-in user.

- [ ] `app/auth/dependencies.py` — verify the Supabase bearer token, expose `get_current_user`.
- [ ] `app/database/chats.py` — thread + message + citation persistence, always keyed to `user_id`.
- [ ] `app/api/chat.py` — list/create threads, load messages, `POST /chat/stream`.
- [ ] `app/chat/messages.py` — AI SDK wire format ↔ internal Pydantic models.
- [ ] `app/chat/streaming.py` — text deltas, citation parts, typed error events.
- [ ] `app/chat/orchestrator.py` — one turn end to end; persist only after a successful run.
- [ ] Enforce 403 when a thread belongs to another user.

**Done when:** `curl` with a real Supabase token streams a cited answer and the thread survives a
reload; no token gives 401.

---

## Phase 6 — Frontend

Goal: an analyst can use it.

- [ ] `src/lib/supabase.ts`, `src/lib/http.ts` (bearer injection, timeout, typed `ApiError` with
      `isNetworkError`), `src/lib/api.ts`.
- [ ] Email sign-in / sign-up page + protected route wrapper.
- [ ] Thread list sidebar (own threads only).
- [ ] Chat page on `useChat` + `DefaultChatTransport` pointed at `/chat/stream`.
- [ ] Citation rendering: company, filing, date, page/section — one click expands the source passage.
- [ ] Empty, streaming, refusal, and error states.

**Done when:** you sign in in a fresh browser, ask a brief question, watch it stream, and verify a
claim by expanding its passage.

---

## Phase 7 — Deploy

- [ ] Railway backend service (uvicorn) + frontend static service.
- [ ] Env vars set per service; `ALLOWED_ORIGINS` includes the deployed frontend.
- [ ] Migrations run against Supabase from CI or manually, deliberately.
- [ ] Re-enable Supabase email confirmation.
- [ ] structlog output readable in Railway logs.

**Done when:** the pilot flow works end to end on the deployed URLs.

---

## Deliberately not doing

Trading recommendations · external data sources · multi-tenant · billing · mobile · frontend tests ·
SSR/Next.js · a separate vector database.
