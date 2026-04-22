"""
api/routes/hospital.py
-----------------------
Multi-tenant hospital-scoped endpoints.

GET  /api/v1/hospital/me                    — current user's hospital summary
GET  /api/v1/hospital/{hospital}/users      — all users in a hospital (admin only)
GET  /api/v1/hospital/{hospital}/predictions — all predictions for a hospital (admin only)
GET  /api/v1/hospital/{hospital}/stats      — aggregate stats for a hospital (admin only)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import AuditLog, PredictionLog, User, UserRole
from api.auth_utils import get_current_user, require_role

router = APIRouter(prefix="/hospital", tags=["Multi-Tenant / Hospital"])


# ── Response schemas ──────────────────────────────────────────────────────────

class HospitalUser(BaseModel):
    id:         str
    email:      str
    full_name:  str
    role:       str
    is_active:  bool
    created_at: Optional[str] = None


class HospitalPrediction(BaseModel):
    id:               str
    user_email:       str
    disease:          str
    risk_label:       str
    risk_probability: float
    model_id:         str
    created_at:       Optional[str] = None


class HospitalStats(BaseModel):
    hospital:          str
    total_users:       int
    active_users:      int
    total_predictions: int
    high_risk_count:   int
    low_risk_count:    int
    diseases_covered:  List[str]
    roles:             Dict[str, int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/me", response_model=HospitalStats)
def my_hospital_stats(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return aggregate stats for the current user's hospital."""
    hospital = current_user.hospital
    if not hospital:
        raise HTTPException(
            status_code=404,
            detail="Your account has no hospital assigned. "
                   "Update your profile via PUT /api/v1/auth/me.",
        )
    return _hospital_stats(hospital, db)


@router.get("/{hospital}/users", response_model=List[HospitalUser])
def hospital_users(
    hospital:     str,
    limit:        int     = Query(50,  ge=1, le=500),
    skip:         int     = Query(0,   ge=0),
    current_user: User    = Depends(require_role(UserRole.admin)),
    db:           Session = Depends(get_db),
):
    """List users registered under a hospital (admin only). Supports skip/limit pagination."""
    users = (
        db.query(User)
        .filter(User.hospital == hospital)
        .order_by(User.created_at)
        .offset(skip)
        .limit(limit)
        .all()
    )
    if not users:
        raise HTTPException(
            status_code=404,
            detail=f"No users found for hospital '{hospital}'.",
        )
    return [
        HospitalUser(
            id=str(u.id),
            email=u.email,
            full_name=u.full_name,
            role=u.role.value,
            is_active=u.is_active,
            created_at=u.created_at.isoformat() if u.created_at else None,
        )
        for u in users
    ]


@router.get("/{hospital}/predictions", response_model=List[HospitalPrediction])
def hospital_predictions(
    hospital:     str,
    limit:        int     = Query(100, ge=1, le=500),
    skip:         int     = Query(0,   ge=0),
    current_user: User    = Depends(require_role(UserRole.admin)),
    db:           Session = Depends(get_db),
):
    """List predictions logged under a hospital (admin only). Supports skip/limit pagination."""
    rows = (
        db.query(PredictionLog, User.email)
        .join(User, PredictionLog.user_id == User.id)
        .filter(PredictionLog.hospital == hospital)
        .order_by(PredictionLog.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for hospital '{hospital}'.",
        )
    return [
        HospitalPrediction(
            id=str(p.id),
            user_email=email,
            disease=p.disease,
            risk_label=p.risk_label,
            risk_probability=round(p.risk_probability, 4),
            model_id=p.model_id,
            created_at=p.created_at.isoformat() if p.created_at else None,
        )
        for p, email in rows
    ]


@router.get("/{hospital}/stats", response_model=HospitalStats)
def hospital_stats(
    hospital:     str,
    current_user: User    = Depends(require_role(UserRole.admin)),
    db:           Session = Depends(get_db),
):
    """Aggregate statistics for a hospital (admin only)."""
    return _hospital_stats(hospital, db)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _hospital_stats(hospital: str, db: Session) -> HospitalStats:
    users = db.query(User).filter(User.hospital == hospital).all()
    if not users:
        raise HTTPException(
            status_code=404,
            detail=f"No users found for hospital '{hospital}'.",
        )

    preds = (
        db.query(PredictionLog)
        .filter(PredictionLog.hospital == hospital)
        .all()
    )

    roles: Dict[str, int] = {}
    for u in users:
        roles[u.role.value] = roles.get(u.role.value, 0) + 1

    diseases = list({p.disease for p in preds})

    return HospitalStats(
        hospital=hospital,
        total_users=len(users),
        active_users=sum(1 for u in users if u.is_active),
        total_predictions=len(preds),
        high_risk_count=sum(1 for p in preds if p.risk_label == "High Risk"),
        low_risk_count=sum(1 for p in preds if p.risk_label == "Low Risk"),
        diseases_covered=sorted(diseases),
        roles=roles,
    )
