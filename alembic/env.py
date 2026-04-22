"""
alembic/env.py
--------------
ClinIQ Alembic environment.

- Reads DATABASE_URL from the environment (falls back to alembic.ini value).
- Targets db.models.Base.metadata for autogenerate support.
- Works with both SQLite (local dev) and PostgreSQL (production).

Usage
-----
  # Local dev (SQLite auto-fallback)
  alembic upgrade head

  # Production (set env var first)
  DATABASE_URL=postgresql://cliniq_user:pass@localhost/cliniq_db alembic upgrade head
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Wire Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Inject DATABASE_URL from environment (overrides alembic.ini) ──────────────
db_url = os.environ.get("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# ── Import models so autogenerate can diff them ───────────────────────────────
from db.models import Base          # noqa: E402  — must be after sys.path is set
target_metadata = Base.metadata


# ── Offline mode (no live DB connection — just emit SQL) ──────────────────────

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (live DB connection) ─────────────────────────────────────────

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
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
