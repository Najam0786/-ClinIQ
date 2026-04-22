"""
db/models.py
-------------
SQLAlchemy ORM models: User, PredictionLog, AuditLog.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, DateTime, Float, Text,
    ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy import TypeDecorator, String as SAString
import enum

from db.database import Base


# ── UUID that works with both SQLite and PostgreSQL ───────────────────────────

class GUID(TypeDecorator):
    """Platform-independent GUID — uses PostgreSQL UUID, SQLite VARCHAR(36)."""
    impl = SAString
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(SAString(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# ── Enums ─────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    admin   = "admin"
    doctor  = "doctor"
    viewer  = "viewer"


# ── User ──────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(GUID, primary_key=True, default=uuid.uuid4)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    full_name       = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(SAEnum(UserRole), default=UserRole.doctor, nullable=False)
    hospital        = Column(String(255), nullable=True)
    is_active       = Column(Boolean, default=True, nullable=False)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login      = Column(DateTime(timezone=True), nullable=True)

    predictions = relationship("PredictionLog", back_populates="user", cascade="all, delete-orphan")
    audit_logs  = relationship("AuditLog",      back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email} [{self.role}]>"


# ── PredictionLog ─────────────────────────────────────────────────────────────

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id               = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id          = Column(GUID, ForeignKey("users.id"), nullable=False, index=True)
    disease          = Column(String(50),  nullable=False, index=True)
    model_id         = Column(String(100), nullable=False)
    model_name       = Column(String(100), nullable=True)
    patient_data     = Column(Text, nullable=True)   # JSON string
    risk_label       = Column(String(20),  nullable=False)
    risk_probability = Column(Float,       nullable=False)
    ip_address       = Column(String(45),  nullable=True)
    created_at       = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="predictions")

    def __repr__(self):
        return f"<PredictionLog {self.disease} {self.risk_label} {self.risk_probability:.2f}>"


# ── AuditLog ──────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id            = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id       = Column(GUID, ForeignKey("users.id"), nullable=True, index=True)
    action        = Column(String(50),  nullable=False, index=True)
    resource_type = Column(String(50),  nullable=True)
    resource_id   = Column(String(100), nullable=True)
    details       = Column(Text, nullable=True)   # JSON string
    ip_address    = Column(String(45),  nullable=True)
    status        = Column(String(20),  default="success")
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} by user={self.user_id}>"
