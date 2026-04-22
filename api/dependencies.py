"""
api/dependencies.py
--------------------
Shared model store — loads/saves trained pipelines to disk.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"
MODELS_DIR.mkdir(exist_ok=True)


def _load_registry() -> Dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_registry(registry: Dict) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def save_model(
    pipeline,
    disease: str,
    best_model_name: str,
    test_auc: float,
    numeric_cols: list,
    categorical_cols: list,
    mlflow_run_id: Optional[str] = None,
) -> str:
    model_id = f"{disease}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.pkl"
    joblib.dump(pipeline, model_path)

    registry = _load_registry()
    registry[model_id] = {
        "model_id":      model_id,
        "disease":       disease,
        "best_model":    best_model_name,
        "test_auc":      test_auc,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "n_features":    len(numeric_cols) + len(categorical_cols),
        "numeric_cols":  numeric_cols,
        "categorical_cols": categorical_cols,
        "mlflow_run_id": mlflow_run_id,
    }
    _save_registry(registry)
    return model_id


def load_model(model_id: str):
    model_path = MODELS_DIR / f"{model_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_id}' not found.")
    return joblib.load(model_path)


def get_latest_model_id(disease: str) -> Optional[str]:
    registry = _load_registry()
    matches = [v for v in registry.values() if v["disease"] == disease]
    if not matches:
        return None
    return sorted(matches, key=lambda x: x["trained_at"], reverse=True)[0]["model_id"]


def get_model_info(model_id: str) -> Optional[Dict]:
    return _load_registry().get(model_id)


def list_all_models() -> list:
    return list(_load_registry().values())
