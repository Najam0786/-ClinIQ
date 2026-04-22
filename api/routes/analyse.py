"""
api/routes/analyse.py
---------------------
POST /api/v1/analyse
Accepts a CSV/Excel file upload + disease type, runs the full ClinIQ
ML pipeline, saves the model to the registry, logs to MLflow.
"""

from __future__ import annotations
import io
import traceback
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas import AnalyseResponse, DiseaseModule
from api.dependencies import save_model
from core.mlflow_utils import log_training_run
from core.drift_detector import save_baseline

router = APIRouter()

_MODULE_MAP = {
    "breast_cancer":  "modules.breast_cancer",
    "heart_disease":  "modules.heart_disease",
    "diabetes":       "modules.diabetes",
    "stroke":         "modules.stroke",
    "respiratory":    "modules.respiratory",
    "universal":      "modules.universal",
}


def _load_df(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(content))
    else:
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Upload a .csv or .xlsx file.",
            )


@router.post("/analyse", response_model=AnalyseResponse, tags=["Analysis"])
async def analyse(
    file:       UploadFile = File(..., description="CSV or Excel dataset"),
    disease:    DiseaseModule = Form(..., description="Disease module to use"),
    target_col: Optional[str] = Form(None, description="Target column name (auto-detected if omitted)"),
):
    """
    Run the full ClinIQ pipeline on an uploaded dataset.
    Returns model metrics, feature importances, and a model_id for future predictions.
    """
    import importlib

    df = _load_df(file)

    if df.empty or len(df) < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset too small ({len(df)} rows). Minimum 50 rows required.",
        )

    try:
        mod = importlib.import_module(_MODULE_MAP[disease.value])
        result_tuple = mod.run(df, target_col=target_col or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}\n{traceback.format_exc()}")

    result = result_tuple[0]

    mlflow_run_id = log_training_run(
        disease=disease.value,
        best_model_name=result.best_model_name,
        cv_scores=result.cv_scores,
        test_auc=result.test_auc,
        n_rows=len(df),
        n_features=len(result.numeric_cols) + len(result.categorical_cols),
    )

    model_id = save_model(
        pipeline=result.pipeline,
        disease=disease.value,
        best_model_name=result.best_model_name,
        test_auc=result.test_auc,
        numeric_cols=result.numeric_cols,
        categorical_cols=result.categorical_cols,
        mlflow_run_id=mlflow_run_id,
    )

    feature_cols = result.numeric_cols + result.categorical_cols
    try:
        save_baseline(disease.value, df, feature_cols)
    except Exception:
        pass

    pred_df = result.predictions_df
    high_risk = int((pred_df["prediction"] == 1).sum())
    low_risk  = int((pred_df["prediction"] == 0).sum())

    top_features = (
        result.feature_importances.head(10)
        .to_dict(orient="records")
    )

    return AnalyseResponse(
        disease=disease.value,
        best_model=result.best_model_name,
        test_auc=result.test_auc,
        cv_scores=result.cv_scores,
        classification_report=result.classification_report,
        top_features=top_features,
        total_patients=len(pred_df),
        high_risk_count=high_risk,
        low_risk_count=low_risk,
        model_id=model_id,
        mlflow_run_id=mlflow_run_id,
    )
