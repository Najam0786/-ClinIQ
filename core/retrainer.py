"""
core/retrainer.py
------------------
Automated model retraining engine.

Workflow
--------
1. The drift /detect endpoint calls `maybe_trigger_retrain()` when drift_ratio >= 0.5.
2. POST /api/v1/retrain/{disease} can also trigger a retrain manually.
3. Retraining runs as a FastAPI BackgroundTask (non-blocking).
4. Job state is persisted in memory (process-scoped) and to a JSON sidecar file
   so it survives a uvicorn --reload cycle but NOT a container restart.
   Use Redis/DB for full persistence in production.

Job lifecycle
-------------
  queued → running → completed | failed
"""

from __future__ import annotations

import importlib
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from core.drift_detector import save_baseline, DRIFT_DIR

RETRAIN_STATE_DIR = Path(os.environ.get("CLINIQ_RETRAIN_DIR", "drift_reports"))
_state_lock = Lock()

_MODULE_MAP = {
    "breast_cancer":  "modules.breast_cancer",
    "heart_disease":  "modules.heart_disease",
    "diabetes":       "modules.diabetes",
    "stroke":         "modules.stroke",
    "respiratory":    "modules.respiratory",
    "universal":      "modules.universal",
}


# ── Job state helpers ─────────────────────────────────────────────────────────

def _state_path(disease: str) -> Path:
    d = RETRAIN_STATE_DIR / disease
    d.mkdir(parents=True, exist_ok=True)
    return d / "retrain_status.json"


def _write_state(disease: str, state: Dict[str, Any]) -> None:
    with _state_lock:
        with open(_state_path(disease), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


def get_retrain_status(disease: str) -> Optional[Dict[str, Any]]:
    path = _state_path(disease)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_retrain_jobs() -> list[Dict[str, Any]]:
    """Return status for all diseases that have been retrained at least once."""
    results = []
    if not RETRAIN_STATE_DIR.exists():
        return results
    for d in sorted(RETRAIN_STATE_DIR.iterdir()):
        if d.is_dir():
            status = get_retrain_status(d.name)
            if status:
                results.append(status)
    return results


# ── Core retraining logic ─────────────────────────────────────────────────────

def run_retrain_job(
    disease:        str,
    data_path:      Optional[str] = None,
    triggered_by:   str = "manual",
    drift_ratio:    Optional[float] = None,
) -> None:
    """
    Execute the full ClinIQ ML pipeline for a disease using the stored baseline
    data (or a fresh CSV from data_path).  Runs synchronously — call via
    BackgroundTasks so it doesn't block the HTTP response.
    """
    import pandas as pd
    from api.dependencies import save_model

    job_id = str(uuid.uuid4())[:8]
    started_at = datetime.now(timezone.utc).isoformat()

    _write_state(disease, {
        "job_id":       job_id,
        "disease":      disease,
        "status":       "running",
        "triggered_by": triggered_by,
        "drift_ratio":  drift_ratio,
        "started_at":   started_at,
        "finished_at":  None,
        "model_id":     None,
        "test_auc":     None,
        "error":        None,
    })

    try:
        # ── Load training data ─────────────────────────────────────────────────
        if data_path:
            df = pd.read_csv(data_path)
        else:
            baseline_path = DRIFT_DIR / disease / "baseline.parquet"
            if not baseline_path.exists():
                raise FileNotFoundError(
                    f"No baseline data for '{disease}'. "
                    "Upload via POST /api/v1/drift/baseline/{disease} first."
                )
            df = pd.read_parquet(baseline_path)

        if len(df) < 50:
            raise ValueError(f"Not enough rows ({len(df)}) to retrain. Minimum 50 required.")

        module_path = _MODULE_MAP.get(disease)
        if not module_path:
            raise ValueError(f"Unknown disease module: '{disease}'")

        # ── Run ML pipeline ────────────────────────────────────────────────────
        mod = importlib.import_module(module_path)
        result_tuple = mod.run(df)
        result = result_tuple[0]

        # ── Save model ─────────────────────────────────────────────────────────
        from core.mlflow_utils import log_training_run
        mlflow_run_id = log_training_run(
            disease=disease,
            best_model_name=result.best_model_name,
            cv_scores=result.cv_scores,
            test_auc=result.test_auc,
            n_rows=len(df),
            n_features=len(result.numeric_cols) + len(result.categorical_cols),
        )
        model_id = save_model(
            pipeline=result.pipeline,
            disease=disease,
            best_model_name=result.best_model_name,
            test_auc=result.test_auc,
            numeric_cols=result.numeric_cols,
            categorical_cols=result.categorical_cols,
            mlflow_run_id=mlflow_run_id,
        )

        # ── Refresh drift baseline with raw input columns ─────────────────────
        # Use only columns that exist in df (pre-engineering) to avoid KeyErrors
        raw_cols = list(df.columns)
        save_baseline(disease, df, raw_cols)

        _write_state(disease, {
            "job_id":       job_id,
            "disease":      disease,
            "status":       "completed",
            "triggered_by": triggered_by,
            "drift_ratio":  drift_ratio,
            "started_at":   started_at,
            "finished_at":  datetime.now(timezone.utc).isoformat(),
            "model_id":     model_id,
            "test_auc":     round(result.test_auc, 4),
            "best_model":   result.best_model_name,
            "error":        None,
        })

    except Exception as exc:
        _write_state(disease, {
            "job_id":       job_id,
            "disease":      disease,
            "status":       "failed",
            "triggered_by": triggered_by,
            "drift_ratio":  drift_ratio,
            "started_at":   started_at,
            "finished_at":  datetime.now(timezone.utc).isoformat(),
            "model_id":     None,
            "test_auc":     None,
            "error":        f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        })
        raise


# ── Auto-trigger helper ───────────────────────────────────────────────────────

def maybe_trigger_retrain(
    disease:     str,
    drift_ratio: float,
    background_tasks: Any,
) -> bool:
    """
    Called by the drift /detect endpoint.
    Returns True if a retrain job was enqueued (drift_ratio >= 0.5).
    Skips if a job is already running for this disease.
    """
    if drift_ratio < 0.5:
        return False

    current = get_retrain_status(disease)
    if current and current.get("status") == "running":
        return False

    background_tasks.add_task(
        run_retrain_job,
        disease=disease,
        triggered_by="auto_drift",
        drift_ratio=drift_ratio,
    )
    return True
