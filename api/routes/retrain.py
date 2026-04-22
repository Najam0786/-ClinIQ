"""
api/routes/retrain.py
----------------------
POST /api/v1/retrain/{disease}          — trigger a retrain job (doctor+ role)
POST /api/v1/retrain/{disease}/from-csv — retrain on a freshly-uploaded CSV
GET  /api/v1/retrain/{disease}/status   — job status for a disease
GET  /api/v1/retrain/jobs               — all retrain jobs across diseases
"""

from __future__ import annotations
import io
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.retrainer import (
    get_retrain_status,
    list_retrain_jobs,
    maybe_trigger_retrain,
    run_retrain_job,
)
from core.drift_detector import load_baseline
from db.models import User, UserRole
from api.auth_utils import get_current_user, require_role

router = APIRouter(prefix="/retrain", tags=["Automated Retraining"])


# ── Response schemas ──────────────────────────────────────────────────────────

class RetrainStatus(BaseModel):
    job_id:       str
    disease:      str
    status:       str
    triggered_by: str
    drift_ratio:  Optional[float] = None
    started_at:   str
    finished_at:  Optional[str]  = None
    model_id:     Optional[str]  = None
    test_auc:     Optional[float] = None
    best_model:   Optional[str]  = None
    error:        Optional[str]  = None


class RetrainQueued(BaseModel):
    message:     str
    disease:     str
    job_status:  str = "queued"
    triggered_by: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/{disease}", response_model=RetrainQueued, status_code=202)
def trigger_retrain(
    disease:          str,
    background_tasks: BackgroundTasks,
    current_user:     User = Depends(require_role(UserRole.doctor)),
):
    """
    Manually trigger a retrain job using the stored drift baseline data.
    The job runs in the background — poll GET /retrain/{disease}/status for progress.
    Requires doctor or admin role.
    """
    baseline = load_baseline(disease)
    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline data found for '{disease}'. "
                   "Train a model via POST /api/v1/analyse first, or "
                   "upload one via POST /api/v1/drift/baseline/{disease}.",
        )

    current = get_retrain_status(disease)
    if current and current.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail=f"A retrain job is already running for '{disease}'. "
                   f"Job ID: {current.get('job_id')}. "
                   "Wait for it to finish or check GET /retrain/{disease}/status.",
        )

    background_tasks.add_task(
        run_retrain_job,
        disease=disease,
        triggered_by=f"manual:{current_user.email}",
    )

    return RetrainQueued(
        message=f"Retrain job queued for '{disease}'. "
                "Poll GET /api/v1/retrain/{disease}/status for progress.",
        disease=disease,
        triggered_by=f"manual:{current_user.email}",
    )


@router.post("/{disease}/from-csv", response_model=RetrainQueued, status_code=202)
def retrain_from_upload(
    disease:          str,
    background_tasks: BackgroundTasks,
    file:             UploadFile        = File(..., description="Fresh labelled CSV to retrain on"),
    current_user:     User              = Depends(require_role(UserRole.doctor)),
):
    """
    Retrain using a freshly uploaded CSV instead of the stored baseline.
    The CSV is saved to a temp file; the background task reads it and cleans up.
    """
    import tempfile, os

    current = get_retrain_status(disease)
    if current and current.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail=f"A retrain job is already running for '{disease}'.",
        )

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if len(df) < 50:
        raise HTTPException(
            status_code=422,
            detail=f"CSV too small ({len(df)} rows). Minimum 50 required.",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()

    background_tasks.add_task(
        run_retrain_job,
        disease=disease,
        data_path=tmp.name,
        triggered_by=f"upload:{current_user.email}",
    )

    return RetrainQueued(
        message=f"Retrain job queued for '{disease}' from uploaded CSV ({len(df)} rows). "
                "Poll GET /api/v1/retrain/{disease}/status for progress.",
        disease=disease,
        triggered_by=f"upload:{current_user.email}",
    )


@router.get("/jobs", response_model=List[Dict[str, Any]])
def all_retrain_jobs(current_user: User = Depends(get_current_user)):
    """List retrain job history across all disease models."""
    return list_retrain_jobs()


@router.get("/{disease}/status", response_model=RetrainStatus)
def retrain_status(
    disease:      str,
    current_user: User = Depends(get_current_user),
):
    """Get the latest retrain job status for a specific disease model."""
    status = get_retrain_status(disease)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No retrain jobs found for '{disease}'.",
        )
    return status
