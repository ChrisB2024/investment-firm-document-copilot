"""Single source of truth for backend configuration.

Every environment variable the backend reads is declared on `Settings`. Nothing
else under `app/` may call `os.getenv` or `load_dotenv` (see ../CLAUDE.md).

Missing or malformed required vars must raise at import time, not at first
request — a backend that boots with half its config is worse than one that
refuses to boot.
"""
from functools import cached_property
from pathlib import Path
from typing import Annotated

from pydantic import PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Module level, not class attributes: pydantic turns leading-underscore class
# attributes into ModelPrivateAttr, so they are not plain values at runtime.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_TRANSACTION_POOLER_PORT = 6543


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- Supabase (Auth + API) ---
    supabase_url: str
    supabase_anon_key: SecretStr
    supabase_service_role_key: SecretStr

    # --- Postgres (Alembic + direct DB access) ---
    # Session-level connection only: port 5432 (session pooler or direct).
    # Port 6543 is the transaction pooler and cannot run migrations.
    database_url: PostgresDsn

    # --- OpenAI ---
    openai_api_key: SecretStr
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_chat_model: str = "gpt-5.5"
    openai_grounding_model: str = "gpt-4.1-mini"
    openai_agent_request_limit: int = 20

    # --- retrieval ---
    retrieval_candidate_k: int = 50
    retrieval_top_k: int = 10
    retrieval_rrf_k: int = 60
    retrieval_neighbor_radius: int = 1
    retrieval_fts_config: str = "english"
    retrieval_fts_keyword_model: str = "gpt-4.1-mini"
    retrieval_fts_keyword_min: int = 3
    retrieval_fts_keyword_max: int = 5
    retrieval_fts_keyword_fast_path_tokens: int = 5

    # --- Server ---
    # Comma-separated in .env ("http://a,http://b"), a list here.
    allowed_origins: Annotated[list[str], NoDecode]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("database_url")
    @classmethod
    def _reject_transaction_pooler(cls, value: PostgresDsn) -> PostgresDsn:
        if any(host["port"] == _TRANSACTION_POOLER_PORT for host in value.hosts()):
            raise ValueError(
                "DATABASE_URL uses port 6543 (Supabase transaction pooler), which "
                "cannot run migrations or hold session state. Use port 5432 — the "
                "session pooler or the direct connection string."
            )
        return value

    @cached_property
    def alembic_url(self) -> str:
        """SQLAlchemy-compatible URL for Alembic and direct DB access.

        The driver is named explicitly because a bare `postgresql://` URL makes
        SQLAlchemy reach for psycopg2, which is not installed — psycopg v3 is.
        """
        _, separator, rest = str(self.database_url).partition("://")
        return f"postgresql+psycopg{separator}{rest}"


settings = Settings()

__all__ = ["Settings", "settings"]