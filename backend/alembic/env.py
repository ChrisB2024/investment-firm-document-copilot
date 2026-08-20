"""Alembic environment.

Two things differ from the stock template:

1. The database URL comes from `app.config.settings`, never from `alembic.ini`.
   `alembic.ini` is tracked in git, so a URL there would commit credentials.
2. `target_metadata` points at the app's models so `--autogenerate` can diff
   them against the live database. With the stock `None`, autogenerate silently
   produces empty migrations.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.config import settings
from app.database.models import Base
from app.database.supabase_auth import AUTH_SCHEMA

config = context.config

# Injected at runtime rather than read from alembic.ini — see note above.
# `%` is doubled because set_main_option writes through configparser, which
# treats a lone `%` as interpolation syntax — and URL-encoded passwords are full
# of them (%40 for @, %24 for $).
config.set_main_option("sqlalchemy.url", settings.alembic_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Keep Supabase-owned objects out of autogenerate.

    `auth.users` is in our metadata only so foreign keys to it can compile. It is
    Supabase's table — without this filter, autogenerate would emit a CREATE
    TABLE for it on the first migration and try to drop it on the next.
    """
    return getattr(obj, "schema", None) != AUTH_SCHEMA


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it, for review."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            # Without this, a column type change (e.g. vector dimensions) is
            # invisible to autogenerate.
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
