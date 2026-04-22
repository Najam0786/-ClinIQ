"""
api/routes/drift.py
--------------------
POST /api/v1/drift/detect/{disease}   — upload CSV → compare vs baseline → drift report
POST /api/v1/drift/baseline/{disease} — manually upload a new baseline CSV
GET  /api/v1/drift/report/{disease}   — latest stored drift report
GET  /api/v1/drift/status             — list all diseases with baselines
"""

from __future__ import annotations
import io
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.drift_detector import (
    compute_drift,
    list_baselines,
    load_baseline,
    load_latest_report,
    save_baseline,
)
from db.models import User, UserRole
from api.auth_utils import get_current_user, require_role

router = APIRouter(prefix="/drift", tags=["Data Drift Detection"])


# ── Response schemas ──────────────────────────────────────────────────────────

class ColumnDriftResult(BaseModel):
    column:          str
    type:            str
    drift_detected:  bool
    drift_score:     float
    test:            Optional[str]   = None
    p_value:         Optional[float] = None
    reference_mean:  Optional[float] = None
    current_mean:    Optional[float] = None
    error:           Optional[str]   = None


class DriftReport(BaseModel):
    disease:          str
    generated_at:     str
    reference_rows:   int
    current_rows:     int
    columns_checked:  int
    columns_drifted:  int
    drift_ratio:      float
    dataset_drift:    bool
    alert:            bool
    drifted_features: List[str]
    summary:          str
    column_reports:   List[Dict[str, Any]]


class BaselineStatus(BaseModel):
    disease:          str
    baseline_exists:  bool
    last_report_at:   Optional[str]   = None
    last_drift_ratio: Optional[float] = None
    last_alert:       Optional[bool]  = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=List[BaselineStatus])
def drift_status(current_user: User = Depends(get_current_user)):
    """List all disease models that have a stored baseline."""
    return list_baselines()


@router.get("/report/{disease}", response_model=DriftReport)
def get_drift_report(
    disease:      str,
    current_user: User = Depends(get_current_user),
):
    """Return the most recent drift report for a disease model."""
    report = load_latest_report(disease)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No drift report found for '{disease}'. "
                   "Run POST /api/v1/drift/detect/{disease} first.",
        )
    return report


@router.post("/baseline/{disease}", status_code=201)
def upload_baseline(
    disease:      str,
    file:         UploadFile        = File(..., description="CSV file to use as the new baseline"),
    current_user: User              = Depends(require_role(UserRole.doctor)),
):
    """
    Manually upload a CSV to replace the drift baseline for a disease.
    The /analyse endpoint does this automatically after training.
    """
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=422, detail="CSV must have at least 2 columns and 1 row.")

    path = save_baseline(disease, df, list(df.columns))
    return {
        "message":  f"Baseline saved for '{disease}'.",
        "rows":     len(df),
        "columns":  list(df.columns),
        "saved_to": str(path),
    }


@router.post("/detect/{disease}", response_model=DriftReport)
def detect_drift(
    disease:      str,
    file:         UploadFile = File(..., description="New data CSV to compare against the baseline"),
    current_user: User       = Depends(get_current_user),
):
    """
    Upload a CSV with recent/incoming data and compare it against the stored
    baseline for the given disease model.

    - `drift_ratio` ≥ 0.30 → **alert** (consider investigating)
    - `drift_ratio` ≥ 0.50 → **dataset_drift** (consider retraining)

    The report is stored as `drift_reports/{disease}/latest_report.json`.
    """
    baseline = load_baseline(disease)
    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline found for '{disease}'. "
                   "Train a model via POST /api/v1/analyse first, or "
                   "upload one via POST /api/v1/drift/baseline/{disease}.",
        )

    try:
        contents   = file.file.read()
        current_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if current_df.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV is empty.")

    try:
        report = compute_drift(disease, current_df, reference_df=baseline)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift computation failed: {exc}")

    return report
