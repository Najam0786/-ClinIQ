"""
api/routes/auth.py
-------------------
POST /api/v1/auth/register  — create account
POST /api/v1/auth/login     — get JWT token
GET  /api/v1/auth/me        — current user profile
PUT  /api/v1/auth/me        — update profile
GET  /api/v1/auth/users     — list all users (admin only)
PUT  /api/v1/auth/users/{id}/role — change role (admin only)
"""

from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, UserRole
from api.auth_utils import (
    hash_password, verify_password, create_access_token,
    get_current_user, require_role, write_audit_log, get_client_ip,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:     EmailStr
    full_name: str        = Field(..., min_length=2, max_length=100)
    password:  str        = Field(..., min_length=8)
    hospital:  Optional[str] = None
    role:      UserRole   = UserRole.doctor


class UserResponse(BaseModel):
    id:         str
    email:      str
    full_name:  str
    role:       str
    hospital:   Optional[str]
    is_active:  bool
    created_at: datetime
    last_login: Optional[datetime]

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserResponse


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    hospital:  Optional[str] = None


class ChangeRoleRequest(BaseModel):
    role: UserRole


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_to_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        hospital=user.hospital,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    body:       RegisterRequest,
    db:         Session = Depends(get_db),
    x_real_ip: Optional[str] = Header(None, alias="X-Real-IP"),
    x_forwarded_for: Optional[str] = Header(None, alias="X-Forwarded-For"),
):
    """Register a new ClinIQ user. Default role: doctor."""
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(
        id=uuid.uuid4(),
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role=body.role,
        hospital=body.hospital,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    ip = (x_forwarded_for or x_real_ip or "unknown").split(",")[0].strip()
    write_audit_log(
        db, action="register", user_id=str(user.id),
        resource_type="user", resource_id=str(user.id),
        details={"email": user.email, "role": user.role.value},
        ip_address=ip,
    )
    return _user_to_response(user)


@router.post("/login", response_model=TokenResponse)
def login(
    form:            OAuth2PasswordRequestForm = Depends(),
    db:              Session = Depends(get_db),
    x_forwarded_for: Optional[str] = Header(None, alias="X-Forwarded-For"),
    x_real_ip:       Optional[str] = Header(None, alias="X-Real-IP"),
):
    """Login with email + password → returns JWT Bearer token."""
    ip   = (x_forwarded_for or x_real_ip or "unknown").split(",")[0].strip()
    user = db.query(User).filter(User.email == form.username).first()

    if not user or not verify_password(form.password, user.hashed_password):
        write_audit_log(db, action="login_failed", user_id=None,
            details={"email": form.username}, ip_address=ip, status="failure")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated. Contact your admin.")

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(
        subject=str(user.id),
        role=user.role.value,
        extra={"email": user.email, "hospital": user.hospital},
    )
    write_audit_log(db, action="login", user_id=str(user.id),
        resource_type="session", details={"email": user.email}, ip_address=ip)

    return TokenResponse(access_token=token, user=_user_to_response(user))


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return _user_to_response(current_user)


@router.put("/me", response_model=UserResponse)
def update_me(
    body:         UpdateProfileRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Update current user's name or hospital."""
    if body.full_name:
        current_user.full_name = body.full_name
    if body.hospital is not None:
        current_user.hospital = body.hospital
    db.commit()
    db.refresh(current_user)
    return _user_to_response(current_user)


@router.get("/users", response_model=List[UserResponse])
def list_users(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(require_role(UserRole.admin)),
):
    """List all registered users. Admin only."""
    return [_user_to_response(u) for u in db.query(User).all()]


@router.put("/users/{user_id}/role", response_model=UserResponse)
def change_user_role(
    user_id:      str,
    body:         ChangeRoleRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(require_role(UserRole.admin)),
):
    """Change a user's role. Admin only."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")
    if str(target.id) == str(current_user.id):
        raise HTTPException(status_code=400, detail="Cannot change your own role.")

    old_role = target.role.value
    target.role = body.role
    db.commit()
    db.refresh(target)

    write_audit_log(
        db, action="role_change", user_id=str(current_user.id),
        resource_type="user", resource_id=str(target.id),
        details={"old_role": old_role, "new_role": body.role.value},
    )
    return _user_to_response(target)
