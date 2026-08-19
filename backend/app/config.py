"""Single source of truth for backend configuration.

Every environment variable the backend reads is declared on `Settings`. Nothing
else under `app/` may call `os.getenv` or `load_dotenv` (see ../CLAUDE.md).

Missing or malformed required vars must raise at import time, not at first
request — a backend that boots with half its config is worse than one that
refuses to boot.
"""

from functools import cached_property

from pydantic import PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # TODO: point this at backend/.env, ignore unknown keys, and decide whether
    # env var names are case-sensitive. Note the field names below are lowercase
    # while .env uses SCREAMING_CASE — check which SettingsConfigDict option
    # makes that mapping work before assuming it's automatic.
    model_config = SettingsConfigDict()

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
    openai_embedding_model: str
    openai_embedding_dimensions: int

    # --- Server ---
    # Comma-separated in .env ("http://a,http://b"), a list here.
    allowed_origins: list[str]

    # TODO: parse allowed_origins from the comma-separated string.
    # Watch out: pydantic-settings tries to JSON-decode env values for complex
    # types *before* your validator runs, so a bare comma-separated string will
    # blow up unless you handle it. Look up `mode="before"` and, if that isn't
    # enough, how to stop pydantic-settings from JSON-parsing this field.
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        raise NotImplementedError

    # TODO: guard against the transaction pooler. Raise a clear error if the
    # port is 6543 — this is the single most likely config mistake in this
    # project and a validator here turns a confusing Alembic hang into a
    # readable startup crash.
    @field_validator("database_url")
    @classmethod
    def _reject_transaction_pooler(cls, value: PostgresDsn) -> PostgresDsn:
        raise NotImplementedError

    @cached_property
    def alembic_url(self) -> str:
        """SQLAlchemy-compatible URL for Alembic and direct DB access.

        TODO: `psycopg` (v3) is installed, but a plain `postgresql://` URL makes
        SQLAlchemy reach for psycopg2, which is not. Return the URL with the
        driver SQLAlchemy should actually use.
        """
        raise NotImplementedError


# TODO: export a single instance the rest of the app imports.
#
# Decide deliberately between a module-level `settings = Settings()` and a
# lazily-cached factory. Module-level is simpler and fails fast at import — but
# it also means importing anything from `app` requires a valid environment,
# which affects how tests and `alembic` behave. Pick one and know why.
