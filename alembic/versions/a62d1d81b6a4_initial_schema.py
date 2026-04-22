"""initial_schema — users, prediction_logs, audit_logs

Revision ID: a62d1d81b6a4
Revises:
Create Date: 2026-04-22

Manually crafted initial migration that creates all three core tables.
Works with both SQLite (local dev) and PostgreSQL (production).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "a62d1d81b6a4"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _guid():
    """Return dialect-aware GUID column type (UUID on PG, VARCHAR(36) elsewhere)."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(36)


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",              _guid(),                    primary_key=True),
        sa.Column("email",           sa.String(255),             nullable=False),
        sa.Column("full_name",       sa.String(255),             nullable=False),
        sa.Column("hashed_password", sa.String(255),             nullable=False),
        sa.Column("role",            sa.String(20),              nullable=False, server_default="doctor"),
        sa.Column("hospital",        sa.String(255),             nullable=True),
        sa.Column("is_active",       sa.Boolean(),               nullable=False, server_default=sa.true()),
        sa.Column("created_at",      sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login",      sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── prediction_logs ───────────────────────────────────────────────────────
    op.create_table(
        "prediction_logs",
        sa.Column("id",               _guid(),                    primary_key=True),
        sa.Column("user_id",          _guid(),                    sa.ForeignKey("users.id"), nullable=False),
        sa.Column("disease",          sa.String(50),              nullable=False),
        sa.Column("model_id",         sa.String(100),             nullable=False),
        sa.Column("model_name",       sa.String(100),             nullable=True),
        sa.Column("patient_data",     sa.Text(),                  nullable=True),
        sa.Column("risk_label",       sa.String(20),              nullable=False),
        sa.Column("risk_probability", sa.Float(),                 nullable=False),
        sa.Column("ip_address",       sa.String(45),              nullable=True),
        sa.Column("created_at",       sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_prediction_logs_user_id", "prediction_logs", ["user_id"])
    op.create_index("ix_prediction_logs_disease",  "prediction_logs", ["disease"])

    # ── audit_logs ────────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id",            _guid(),                    primary_key=True),
        sa.Column("user_id",       _guid(),                    sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action",        sa.String(50),              nullable=False),
        sa.Column("resource_type", sa.String(50),              nullable=True),
        sa.Column("resource_id",   sa.String(100),             nullable=True),
        sa.Column("details",       sa.Text(),                  nullable=True),
        sa.Column("ip_address",    sa.String(45),              nullable=True),
        sa.Column("status",        sa.String(20),              nullable=True, server_default="success"),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action",  "audit_logs", ["action"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("prediction_logs")
    op.drop_table("users")
