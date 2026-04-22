"""
api/routes/audit.py
--------------------
GET /api/v1/audit        — full audit log (admin only)
GET /api/v1/audit/me     — current user's own prediction + action history
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import AuditLog, PredictionLog, User, UserRole
from api.auth_utils import get_current_user, require_role

router = APIRouter(prefix="/audit", tags=["Audit & Compliance"])


class AuditEntry(BaseModel):
    id:            str
    user_id:       Optional[str]
    action:        str
    resource_type: Optional[str]
    resource_id:   Optional[str]
    details:       Optional[dict]
    ip_address:    Optional[str]
    status:        str
    created_at:    datetime

    model_config = {"from_attributes": True}


class PredictionEntry(BaseModel):
    id:               str
    disease:          str
    model_id:         str
    model_name:       Optional[str]
    risk_label:       str
    risk_probability: float
    ip_address:       Optional[str]
    created_at:       datetime

    model_config = {"from_attributes": True}


def _parse_log(log: AuditLog) -> AuditEntry:
    details = None
    if log.details:
        try:
            details = json.loads(log.details)
        except Exception:
            details = {"raw": log.details}
    return AuditEntry(
        id=str(log.id),
        user_id=str(log.user_id) if log.user_id else None,
        action=log.action,
        resource_type=log.resource_type,
        resource_id=log.resource_id,
        details=details,
        ip_address=log.ip_address,
        status=log.status,
        created_at=log.created_at,
    )


@router.get("", response_model=List[AuditEntry])
def get_audit_log(
    limit:        int     = Query(100, le=500),
    action:       Optional[str] = Query(None),
    db:           Session       = Depends(get_db),
    current_user: User          = Depends(require_role(UserRole.admin)),
):
    """Full audit trail — admin only. Filter by action if needed."""
    q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if action:
        q = q.filter(AuditLog.action == action)
    return [_parse_log(log) for log in q.limit(limit).all()]


@router.get("/me/predictions", response_model=List[PredictionEntry])
def my_predictions(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Current user's prediction history."""
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == current_user.id)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        PredictionEntry(
            id=str(p.id),
            disease=p.disease,
            model_id=p.model_id,
            model_name=p.model_name,
            risk_label=p.risk_label,
            risk_probability=p.risk_probability,
            ip_address=p.ip_address,
            created_at=p.created_at,
        )
        for p in logs
    ]


@router.get("/me/actions", response_model=List[AuditEntry])
def my_actions(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """Current user's login + action history."""
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_parse_log(log) for log in logs]
