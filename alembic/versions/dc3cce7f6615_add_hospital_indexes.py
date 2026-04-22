"""add_hospital_indexes — multi-tenant: index users.hospital + denorm column on prediction_logs

Revision ID: dc3cce7f6615
Revises: a62d1d81b6a4
Create Date: 2026-04-22

Adds:
  - Index on users.hospital for fast per-hospital user queries.
  - hospital VARCHAR(255) column on prediction_logs (denormalised for direct
    filtering without a JOIN when listing a hospital's predictions).
  - Index on prediction_logs.hospital.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dc3cce7f6615'
down_revision: Union[str, Sequence[str], None] = 'a62d1d81b6a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_users_hospital", "users", ["hospital"])

    op.add_column(
        "prediction_logs",
        sa.Column("hospital", sa.String(255), nullable=True),
    )
    op.create_index("ix_prediction_logs_hospital", "prediction_logs", ["hospital"])


def downgrade() -> None:
    op.drop_index("ix_prediction_logs_hospital", "prediction_logs")
    op.drop_column("prediction_logs", "hospital")
    op.drop_index("ix_users_hospital", "users")
