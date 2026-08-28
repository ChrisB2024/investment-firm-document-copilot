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
- Every filing opens with a TOC whose entries read exactly like real headings. Apple's sit
  inside `<a href>` and real headings do not — but Microsoft's use dot leaders and no anchors
  at all, so a link filter alone lets every Item through twice. The second rule is that a
  heading's title must contain a real word, which a row of dots does not.
- Prose cross-references items too ("...discussed in Part I, Item 1A..."), so a heading must be a
  short standalone block, not any line containing "Item 1A".
- Table cells split values: `$` / `201,183` / `(2)` / `%` are four separate `<td>`s.

Heading detection is verified across all 25 filings — 21–23 Items each, every core item present,
no TOC bleed. Section bodies come out as clean prose (Apple 2024 Item 1A = 68,735 chars;
Item 1B = "None.").

- [x] `lxml` and `tiktoken` added. HTML parsing and BPE tokenisation are both in the
      "genuinely hard to get right" category the dependency policy allows.
- [x] `ingest/` scaffolded with the validated selectors and heuristics.
- [x] `ingest/extract.py` — HTML → sections, tables, normalized Markdown. All 25 filings:
      21-23 sections, 1,724 tables, every core item present, contiguous table_index, 3.5s.
      Tables are emitted as events in the single document-order walk, so each knows its
      section by construction and the walk never descends into cells (raw-figure leakage
      into prose: 35 chars across the whole corpus).
- [x] `ingest/chunk.py` — ~700 token chunks with overlap, split on paragraph boundaries,
      never mid-sentence and never mid-table-row. Prepend the section heading to every chunk;
      a chunk that does not name its company and section is unfindable and uncitable.
      Two measured gaps, both deliberate rather than unnoticed: the budget is enforced on
      summed atom counts while the chunk text is a span slice, so uncounted separators put
      184 of 2,321 chunks (7.9%) over 700, max 730; and the walk-back only takes an atom
      back if it fits in `OVERLAP_TOKENS`, so 377 of 1,760 consecutive pairs (21.4%) share
      no text — paragraph atoms are simply larger than the overlap allowance.
- [x] `ingest/embed.py` — batches under both the input and token caps, reorders responses
      by index (a reorder would otherwise mislabel every passage), verifies dimension and
      count. Embeds chunks and tables through one path.
- [x] Tables made a second retrievable source type: `document_tables` has its own
      `embedding` and generated `search_vector`, with HNSW and GIN indexes (migration
      `dedf64806f7f`). 46% of a filing's comma-formatted figures appear only inside a
      table, so a chunk-only corpus cannot answer the brief's numeric questions.
      Measured: 1,724 tables, median 177 tokens, max 1,762 — all fit the 8191 limit whole.
- [x] `ingest/run.py` — CLI over `manifest.json` with `--ticker/--year/--dry-run/--limit`,
      idempotent by accession number and resumable via `source_documents.content_hash`
      (migration `b13d3bb1bb88`). A failing filing does not abandon the rest.
- [x] `app/database/session.py` — lazy async engine + session factory.
- [x] `ingest/persist.py` — upsert-by-accession, replace chunks/tables, pending-embedding
      queries. Verified: re-running writes nothing; bumping the extractor version
      re-extracts and still leaves exactly one copy.
- [x] Tests: 47 passing, 1 xfailed. 39 fast (`-m "not integration"`, no network, no DB)
      plus 8 integration against live Supabase that roll back rather than delete. Every
      rule was mutation-tested — deleted from the source, then checked that a test went
      red. That found three tests passing without proving anything: two fixtures placed
      so the rule under test could not fail, and `fake_embeddings(dimensions=...)` shadowed
      into a no-op. The strict xfail records an unsplittable paragraph going out at 8,008
      tokens against a 700 budget; no corpus atom exceeds 680, so it is latent.

**Done when:** ~~all 25 filings ingested; a spot check on Apple FY2024 shows chunks whose text you
can find in the original filing, with correct ticker/year/section metadata.~~ **Met.**
25 documents, 2,321 chunks, 1,724 tables; zero rows missing an embedding or a search_vector,
zero duplicate accession numbers. Apple FY2024: 65 chunks, contiguous indices, correct
metadata on every one, and all 666 of its paragraphs found verbatim in the source HTML.

Carried into Phase 3, both found by querying the ingested corpus rather than predicted:

- **Rank lexical results with `ts_rank_cd`, not `ts_rank`.** `ts_rank` ignores term
  proximity, so "supplier concentration risk" returned four NVDA *Item 15* chunks tied at
  exactly 0.2464 — one occurrence of each term, scattered, no length normalisation to
  separate them. `ts_rank_cd` scores cover density and returns Apple's Item 1A instead.
  Tested and rejected: `setweight` on heading vs body changes nothing here.
- **NVIDIA files its financial statements under Item 15, not Item 8** (135 chunks vs 5;
  the other four filers are the reverse). The extraction is faithful — NVDA's Item 8 is a
  one-chunk "see Part IV, Item 15" — but a filter on Item 8 silently drops NVIDIA from any
  cross-company comparison, and an NVIDIA revenue citation reads "Exhibits and Financial
  Statement Schedules". Wants a normalised topic in `chunk_metadata` rather than a relabelled
  heading; costs a re-ingest (~$0.03) once Phase 3 shows what retrieval actually filters on.

---

## Phase 3 — Retrieval

Goal: given a question, the right passages come back — provable before any LLM is involved.

- [x] `app/retrieval/queries.py` — bounded pgvector and full-text queries over both source types,
      with ticker / fiscal-year / form filters. Rank lexically with `ts_rank_cd`, never `ts_rank`:
      the latter ignores term proximity and returned four NVDA *Item 15* chunks tied at exactly
      0.2464 for "supplier concentration risk". Require every term, falling back to any term only
      when that returns *zero* rows — `plainto_tsquery` ANDs fifteen-odd lexemes, so two real
      analyst questions matched nothing at all.
- [x] `app/retrieval/fusion.py` — RRF on rank alone. Mean overlap between the two arms is 4 of 20
      and one question overlaps in nothing, which is the case for fusing at all. `k` is not worth
      tuning (top-10 membership identical at k = 10, 60, 200). Ties are the normal case, so the
      tie-break is load-bearing and the output is order-independent.
- [x] `app/retrieval/retriever.py` — `retrieve` and `retrieve_per_ticker`. Neighbour expansion is
      constrained to the same *section*, not just the same document: 23.3% of adjacent chunk pairs
      straddle a section boundary, so `chunk_index ± 1` alone attaches another Item's prose to the
      citation one time in four.
- [x] A scratch CLI (`python -m app.retrieval.cli`, `--brief` runs all ten).
- [x] Run the 10 questions in [client-brief.md](client-brief.md) through it. Findings below.
- [ ] `retrieve_per_cell` — one statement covering the (ticker, fiscal_year) grid. See below.
- [ ] Tests: RRF ordering with known inputs, filter application, empty-result handling.

### What the ten questions showed

Two distinct failures, and only the first was predicted.

**Company coverage.** Five of the ten questions name several companies, and a single top-10
returns two or three of them:

| Q | asks about | `retrieve(limit=10)` returned |
| - | ---------- | ----------------------------- |
| 6 | all five | `NVDA×6 AAPL×4` — Alphabet, Microsoft, Amazon absent |
| 7 | Apple + NVIDIA | `AAPL×9 NVDA×1` |
| 8 | MSFT, GOOGL, AMZN, NVDA | `GOOGL×5 NVDA×3 MSFT×2` — Amazon absent |
| 9 | all five | `AAPL×5 GOOGL×5` |
| 10 | all five | `AAPL×5 GOOGL×3 MSFT×2` |

The four single-company questions (1, 3, 4, 5) pass cleanly, so this is not a ranking bug —
ranking by relevance is doing the right thing and the shape is wrong for the question.
`retrieve_per_ticker` fixes all five: every company asked about is covered.

**Year coverage, which fan-out makes worse.** Nine of the ten questions ask how something
*changed* across 2021–2025, and fan-out at 2 passages per company gives at most 2 of 5 years:

```
Q6   AAPL:2/5  AMZN:2/5  GOOGL:1/5  MSFT:1/5  NVDA:1/5
Q9   AAPL:2/5  AMZN:2/5  GOOGL:2/5  MSFT:2/5  NVDA:2/5
```

The arithmetic is the constraint, not the implementation: five companies across five years is 25
filings, and a top-10 cannot hold them. Company coverage and year coverage compete for the same
ten slots. Full grid coverage costs 25 passages / ~15,000 tokens, which a context window absorbs
comfortably — so the answer is to fan out over (ticker, fiscal_year), not ticker alone.

**Do the grid in one statement.** `EXPLAIN ANALYZE` reports *server* time; against remote Supabase
a round trip is ~100 ms, so the per-cell loop is ~5 s for 50 queries — too slow for interactive
chat. A single `row_number() over (partition by ticker, fiscal_year order by distance)` returns all
25 cells in **69–758 ms**. One round trip instead of fifty.

This also corrects the note in `retrieve_per_ticker` claiming each arm costs single-digit
milliseconds: that is server time. Five companies is ~1 s of wall clock, not ~30 ms.

**Still open, and now with evidence available:** whether to weight the arms. `contributions` shows
plenty of passages found by one arm only — Q1 rank 6 is `text=1` with no vector entry, rank 7 is
`vector=1` with no text entry, both scoring 0.01639.

**Done when:** ~~for each of the 10 brief questions, the passages needed to answer it appear in the
top ~10 results.~~ Company coverage met via `retrieve_per_ticker`; year coverage needs the per-cell
query above.

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
