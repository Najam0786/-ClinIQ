"""
api/auth_utils.py
------------------
JWT token creation/verification, password hashing, role-based access guards.
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import bcrypt as _bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, UserRole, AuditLog

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY       = os.environ.get("JWT_SECRET_KEY", "cliniq-dev-secret-change-in-production")
ALGORITHM        = "HS256"
ACCESS_TOKEN_TTL = int(os.environ.get("ACCESS_TOKEN_TTL_MINUTES", "480"))  # 8 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(subject: str, role: str, extra: dict | None = None) -> str:
    payload = {
        "sub":  subject,
        "role": role,
        "iat":  datetime.now(timezone.utc),
        "exp":  datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_TTL),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid or expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ── Current-user dependency ───────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db:    Session = Depends(get_db),
) -> User:
    payload = decode_token(token)
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload.")

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or deactivated.")
    return user


def require_role(*roles: UserRole):
    """Role guard — usage: Depends(require_role(UserRole.admin))"""
    def _guard(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role(s): {[r.value for r in roles]}",
            )
        return current_user
    return _guard


# ── Audit logging helper ──────────────────────────────────────────────────────

def write_audit_log(
    db:            Session,
    action:        str,
    user_id:       str | None  = None,
    resource_type: str | None  = None,
    resource_id:   str | None  = None,
    details:       dict | None = None,
    ip_address:    str | None  = None,
    status:        str         = "success",
) -> None:
    try:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=json.dumps(details) if details else None,
            ip_address=ip_address,
            status=status,
        )
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
