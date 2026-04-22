"""
core/mlflow_utils.py
---------------------
MLflow experiment logging utilities for ClinIQ.
Safe-fail: if MLflow is unavailable or disabled, functions return None quietly.

Set MLFLOW_TRACKING_URI env var to point to a remote MLflow server.
Defaults to local ./mlruns directory.
"""

from __future__ import annotations
import os
from typing import Dict, Optional

MLFLOW_ENABLED = os.environ.get("CLINIQ_MLFLOW_ENABLED", "true").lower() == "true"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = os.environ.get("CLINIQ_MLFLOW_EXPERIMENT", "ClinIQ-Training")


def log_training_run(
    disease: str,
    best_model_name: str,
    cv_scores: Dict[str, float],
    test_auc: float,
    n_rows: int,
    n_features: int,
) -> Optional[str]:
    """
    Log a training run to MLflow.
    Returns the run_id string or None if MLflow is disabled / unavailable.
    """
    if not MLFLOW_ENABLED:
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{disease}__{best_model_name}") as run:
            mlflow.set_tags({
                "disease":    disease,
                "model_type": best_model_name,
                "source":     "ClinIQ API",
            })

            mlflow.log_params({
                "disease":         disease,
                "best_model":      best_model_name,
                "n_rows":          n_rows,
                "n_features":      n_features,
            })

            mlflow.log_metric("test_auc", test_auc)
            for model_name, auc in cv_scores.items():
                safe_name = model_name.lower().replace(" ", "_")
                mlflow.log_metric(f"cv_auc_{safe_name}", auc)

            return run.info.run_id

    except Exception:
        return None


def log_model_artifact(model_id: str, pipeline, run_id: str) -> None:
    """Register a trained sklearn pipeline to the MLflow model registry."""
    if not MLFLOW_ENABLED or run_id is None:
        return
    try:
        import mlflow.sklearn
        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"ClinIQ-{model_id}",
            )
    except Exception:
        pass


def get_best_run(disease: str) -> Optional[Dict]:
    """Fetch the best run for a disease from MLflow by test_auc."""
    if not MLFLOW_ENABLED:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.disease = '{disease}'",
            order_by=["metrics.test_auc DESC"],
            max_results=1,
        )
        if not runs:
            return None

        run = runs[0]
        return {
            "run_id":     run.info.run_id,
            "best_model": run.data.params.get("best_model"),
            "test_auc":   run.data.metrics.get("test_auc"),
            "disease":    disease,
        }
    except Exception:
        return None
