"""
db/database.py
--------------
SQLAlchemy engine + session factory.
Reads DATABASE_URL from environment (falls back to SQLite for local dev).
"""

from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./cliniq.db",   # local dev fallback — no Postgres required
)

# PostgreSQL via psycopg2, SQLite for local dev
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_size=10 if not DATABASE_URL.startswith("sqlite") else 1,
    max_overflow=20 if not DATABASE_URL.startswith("sqlite") else 0,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't exist (called on API startup)."""
    from db import models  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)
