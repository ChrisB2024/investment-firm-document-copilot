# Backend

FastAPI service for Document Copilot. Everything below runs from `backend/`.

For the *why* behind these choices, see [../docs/guides/backend-setup.md](../docs/guides/backend-setup.md).
For coding conventions, see [CLAUDE.md](CLAUDE.md).

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) — manages the venv, the lockfile, and every command below
- A Supabase project ([setup guide](../docs/guides/supabase-setup.md)) and an OpenAI API key

You never activate the venv by hand. `uv run <cmd>` resolves it for you.

## First-time setup

```bash
cd backend
uv sync                 # creates .venv, installs deps + app/ as editable package
cp .env.example .env    # then fill in real values
```

`.env` is gitignored. `.env.example` is the tracked template — if you add a
setting, add it to both.

Verify:

```bash
uv run python -c "from app.config import settings; print(settings.supabase_url)"
```

If that prints your URL, config and imports are both wired correctly.

## Run the server

```bash
uv run uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Health: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`

`--reload` watches the filesystem and restarts on save. Leave it off in production.

Direct execution also works (`app/main.py` has a `__main__` block), which is what
the IDE run button uses:

```bash
uv run python app/main.py
```

Pick a different port when 8000 is taken:

```bash
uv run uvicorn app.main:app --reload --port 8001
```

## Configuration

`app/config.py` is the **single source of truth** for env vars. No module under
`app/` may call `os.getenv` or `load_dotenv`.

`Settings` is instantiated at import time, so a missing or malformed required var
crashes the process on boot rather than on the first request. That is deliberate.

To add a new setting:

1. Add the field to `Settings` in [app/config.py](app/config.py) — with a type, and a default if it's optional.
2. Add it to `.env.example` with a placeholder value and a comment.
3. Add the real value to your local `.env`.
4. Import it where needed: `from app.config import settings`.

Required (no default): `SUPABASE_URL`, `SUPABASE_ANON_KEY`,
`SUPABASE_SERVICE_ROLE_KEY`, `DATABASE_URL`, `OPENAI_API_KEY`, `ALLOWED_ORIGINS`.
Everything else — model names, retrieval tuning knobs — has a default and is
optional in `.env`.

`ALLOWED_ORIGINS` is comma-separated in `.env` and becomes a list on `settings`.
CORS uses explicit origins, never `*`, because browsers ignore a wildcard once
credentials are allowed.

## Dependencies

`uv` owns `pyproject.toml` and `uv.lock`. Don't hand-edit either.

```bash
uv add <package>            # runtime dependency
uv add --dev <package>      # dev-only (test, lint)
uv remove <package>
uv sync                     # install exactly what uv.lock says
uv lock --upgrade           # bump the lockfile deliberately
```

`add-bounds = "exact"` pins exact versions on `uv add`. `uv.lock` is committed —
after a `git pull` that touched it, run `uv sync`.

## Database migrations

Alembic owns the schema. Never edit tables by hand in the Supabase dashboard.

```bash
uv run alembic revision --autogenerate -m "add document tables"
uv run alembic upgrade head       # apply
uv run alembic current            # what's applied now
uv run alembic history            # all revisions
uv run alembic downgrade -1       # roll back one
```

Always read the generated migration before applying it. Autogenerate cannot infer
Postgres/Supabase-specific things — write those as explicit operations:
`create extension vector`, `vector(1536)` columns, generated `tsvector` columns,
HNSW/GIN indexes, RLS enablement and policies.

`DATABASE_URL` must be a **session-level** connection on port `5432` (session
pooler or direct). Port `6543` is the transaction pooler and cannot run
migrations — `Settings` rejects it at boot with an explanatory error.

> **Not wired up yet.** `alembic/env.py` is still the stock template:
> `target_metadata = None`, and the URL comes from the placeholder
> `sqlalchemy.url` in `alembic.ini` rather than from settings. Before the first
> migration, `env.py` needs to import the app's SQLAlchemy metadata and set the
> URL from `settings.alembic_url` (which already swaps in the `psycopg` driver).
> Credentials never go in `alembic.ini` — it's tracked in git.

## Tests

```bash
uv run pytest                        # everything
uv run pytest -m "not integration"   # fast suite: no network, no DB
uv run pytest tests/path/test_x.py   # one file
uv run pytest -k retrieval           # match by name
uv run pytest -x -vv                 # stop at first failure, verbose
```

The fast suite must stay green and must not touch the network or the database.
Anything needing live OpenAI/Supabase credentials goes behind
`@pytest.mark.integration`.

## Lint & format

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # autofix
uv run ruff format .         # format
```

## Layout

```text
backend/
├── app/
│   ├── main.py        # FastAPI entrypoint, CORS, /health
│   ├── config.py      # pydantic-settings — all env vars live here
│   └── ...            # api/, auth/, chat/, assistant/, retrieval/, grounding/, database/
├── alembic/           # migration env + versions/
├── alembic.ini
├── tests/
├── pyproject.toml     # deps, managed by uv
├── uv.lock            # committed; the source of truth for installs
└── .env               # gitignored; copy from .env.example
```

`app/` is installed as an editable package by `uv sync`, so `from app.config import settings`
resolves from uvicorn, pytest, notebooks, and IDE run buttons — not just from `backend/`.

## Troubleshooting

**`ValidationError` on startup** — a required var is missing from `.env`. The error
names the field. Compare `.env` against `.env.example`.

**`DATABASE_URL uses port 6543...`** — you grabbed the transaction pooler string
from Supabase. Use the session pooler or direct connection on `5432`.

**`ModuleNotFoundError: No module named 'app'`** — the editable install is missing.
Run `uv sync`. If your IDE still can't resolve it, point its interpreter at
`backend/.venv/bin/python`.

**CORS errors in the browser** — the frontend origin isn't in `ALLOWED_ORIGINS`.
Add it (comma-separated, exact scheme + host + port) and restart the server.
Settings are read once at import, so `.env` edits need a restart.

**`Address already in use`** — something is on 8000. `lsof -ti:8000 | xargs kill`
or run on another port.

**Config change seems to have no effect** — `--reload` restarts on `.py` changes,
not on `.env` changes. Restart manually.
