"""FastAPI entrypoint.

Run locally from backend/:

    uv run uvicorn app.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Document Copilot API")

# TODO: add CORS middleware.
# Origins come from `settings.allowed_origins` — never "*", because the browser
# sends a Supabase bearer token and (later) needs credentialed requests to work.
# Importing settings here is also what makes a missing env var crash the boot.


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for Railway and for your own smoke test.

    TODO: decide what this should assert. A route that always returns "ok"
    proves only that the process is up. Consider whether it should confirm
    config loaded — and deliberately do NOT make it check Supabase or OpenAI,
    or a third-party outage will make Railway kill a healthy container.
    """
    raise NotImplementedError


# TODO: configure structlog once, at startup, before anything logs.
# TODO: mount routers here as they land (Phase 5: app/api/chat.py).
