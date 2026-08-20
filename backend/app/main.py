"""FastAPI entrypoint.

Run locally from backend/:

    uv run uvicorn app.main:app --reload
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(title="Document Copilot")

# Explicit origins, never "*": browsers ignore a wildcard when credentials are
# allowed, which would break the Supabase bearer token the SPA sends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for Railway and for local smoke tests.

    Deliberately does not touch Supabase or OpenAI: a third-party outage
    should not make Railway kill an otherwise healthy container.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
