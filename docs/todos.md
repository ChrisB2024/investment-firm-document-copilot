# Build todos

Sequenced so each phase ends in something you can run and check. Do not start a phase until the
previous one's **Done when** holds.

Reference: [architecture.md](architecture.md) for the target shape, [client-brief.md](client-brief.md)
for what the answers must satisfy.

---

## Phase 0 — Foundations

Goal: both services boot, config fails loudly, Supabase project exists.

- [x] Supabase project created; `backend/.env` + `frontend/.env` filled.
- [x] Backend deps installed; `[build-system]` + hatch wheel target added, so
      `from app...` resolves from any cwd.
- [x] Frontend scaffolded: Vite + React 19 + TS strict, Tailwind v4, shadcn (radix/nova),
      React Router, `@/*` alias.
- [x] `backend/app/config.py` — settings load from `.env` regardless of cwd, secrets as
      `SecretStr`, comma-separated `ALLOWED_ORIGINS` via `NoDecode`, transaction-pooler
      (`:6543`) rejected at boot, `alembic_url` names the `psycopg` driver.
- [x] `backend/app/main.py` — CORS from `settings.allowed_origins` with credentials;
      `/health` as a pure liveness probe.
- [x] `frontend/src/lib/env.ts` — `loadEnv()` validates and trims all three vars and
      reports every missing one in a single error.
- [x] `env` imported in `main.tsx` for its side effect, so the check actually runs.
- [x] `backend/README.md` — setup, config, migrations, troubleshooting.

**Done when:** ~~`uv run uvicorn app.main:app --reload` serves `/health`, `pnpm dev` serves a
page, and deleting a required env var crashes each on boot.~~ **Met.**

Deferred out of Phase 0 (neither blocks Phase 1):

- [ ] structlog configuration in `main.py` — nothing logs structurally yet, so this lands
      with the first real request path in Phase 5.
- [ ] Fix stale `../AGENTS.md` links — 9 references across `README.md`, `claude.md`,
      `backend/claude.md`, `frontend/claude.md`. The files are now `claude.md`.

---

## Phase 1 — Schema

Goal: the tables retrieval needs exist in Supabase, created by Alembic.

- [x] `alembic/env.py` reads the URL from `settings.alembic_url`, points `target_metadata`
      at `Base.metadata`, sets `compare_type=True`, and filters Supabase's `auth` schema
      out of autogenerate via `include_object`.
- [x] `alembic.ini`'s placeholder `sqlalchemy.url` removed — the file is tracked in git.
- [x] Seven models, one per file, under `app/database/`: `user`, `chat_thread`,
      `chat_message`, `message_citation`, `source_document`, `document_chunk`,
      `document_table`. `base.py` holds `Base` + constraint naming convention;
      `models.py` imports all of them so Alembic can see them.
- [x] Integrity constraints: unique `accession_number`, unique
      `(document_id, chunk_index)`, unique `(thread_id, sequence)`, unique `email`,
      and a `role` CHECK backed by `MessageRole`.
- [x] Indexes on every foreign key, plus `(user_id, created_at DESC)` for the thread
      sidebar query.
- [x] `users.id` references `auth.users(id)` ON DELETE CASCADE. `supabase_auth.py`
      declares the stub that makes the cross-schema FK compile.
- [x] First migration applied (`aed0df9ad92c`), hand-edited after autogenerate:
  - [x] `create extension if not exists vector` — **first statement in `upgrade()`**,
        before any `vector(1536)` column is created
  - [x] generated `tsvector` column over chunk text
  - [x] HNSW index on `embedding`, GIN index on the tsvector and on `chunk_metadata`
- [x] RLS enabled deny-all (no policies) on all seven tables, in a migration rather
      than by hand: Supabase exposes `public` through PostgREST using the anon key,
      which ships in the browser bundle. The backend's service-role key bypasses RLS,
      so this costs the app nothing. Verified: with a real row present, the anon key
      reads `[]` while service-role sees it.

Verified against the live database: all seven tables create, the cross-schema FK
rejects a user id with no matching auth user, and a duplicate `accession_number` is
rejected.

Notes:

- `uv run alembic check` reports pending changes without writing a migration file.
- `metadata.create_all()` ignores the `include_object` filter, so any test fixture
  building the schema must pass
  `tables=[t for t in Base.metadata.sorted_tables if t.schema != "auth"]`.
- Migrations now require a database that has Supabase Auth installed; a plain
  Postgres will fail on the `auth.users` foreign key.

Anything created by hand in a migration must also be declared on the model, or
autogenerate sees it only in the database and emits a DROP on the next revision.
RLS is the exception it cannot see at all — autogenerate never compares
`relrowsecurity`, so it lives only in the migration.

**Done when:** ~~`uv run alembic upgrade head` against Supabase succeeds from a clean DB, and
`upgrade head` on an already-migrated DB is a no-op.~~ **Met** — head is `b764413cd363`,
`alembic check` reports no drift.

---

## Phase 2 — Ingestion

Goal: the 25 downloaded 10-Ks are in Postgres as embedded, searchable chunks. This is the phase
that decides whether the product works — spend the time here.

What the corpus is, measured rather than assumed:

- Inline XBRL from filing agents. 1.5–6.5 MB each, mostly markup (one Apple filing has 11,098
  inline `style` attributes, 963 `ix:nonfraction` tags).
- **No `<h1>`–`<h6>` anywhere.** Headings are bold `<span>`s; structure must come from text.
- `ix:hidden` holds non-rendering XBRL facts — strip it or numbers duplicate without context.
- Every filing opens with a TOC whose entries read exactly like real headings. TOC entries sit
  inside `<a href>`; real headings do not. That is the whole distinction.
- Prose cross-references items too ("...discussed in Part I, Item 1A..."), so a heading must be a
  short standalone block, not any line containing "Item 1A".
- Table cells split values: `$` / `201,183` / `(2)` / `%` are four separate `<td>`s.

Heading detection is verified across all 25 filings — 17–20 Items each, no misses, no TOC bleed.
Section bodies come out as clean prose (Apple 2024 Item 1A = 68,735 chars; Item 1B = "None.").

- [x] `lxml` and `tiktoken` added. HTML parsing and BPE tokenisation are both in the
      "genuinely hard to get right" category the dependency policy allows.
- [x] `ingest/` scaffolded with the validated selectors and heuristics.
- [x] `ingest/extract.py` — HTML → sections, tables, normalized Markdown. All 25 filings:
      21-23 sections, 1,724 tables, every core item present, contiguous table_index, 3.5s.
      Tables are emitted as events in the single document-order walk, so each knows its
      section by construction and the walk never descends into cells (raw-figure leakage
      into prose: 35 chars across the whole corpus).
- [ ] `ingest/chunk.py` — ~700 token chunks with overlap, split on paragraph boundaries,
      never mid-sentence and never mid-table-row. Prepend the section heading to every chunk;
      a chunk that does not name its company and section is unfindable and uncitable.
- [x] `ingest/embed.py` — batches under both the input and token caps, reorders responses
      by index (a reorder would otherwise mislabel every passage), verifies dimension and
      count. Embeds chunks and tables through one path.
- [x] Tables made a second retrievable source type: `document_tables` has its own
      `embedding` and generated `search_vector`, with HNSW and GIN indexes (migration
      `dedf64806f7f`). 46% of a filing's comma-formatted figures appear only inside a
      table, so a chunk-only corpus cannot answer the brief's numeric questions.
      Measured: 1,724 tables, median 177 tokens, max 1,762 — all fit the 8191 limit whole.
- [ ] `ingest/run.py` — CLI over `manifest.json`, idempotent by accession number.
- [ ] Tests: chunk boundaries, metadata propagation, idempotency. No network.

**Done when:** all 25 filings ingested; a spot check on Apple FY2024 shows chunks whose text you
can find in the original filing, with correct ticker/year/section metadata.

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
