"""
api/routes/predict.py
---------------------
POST /api/v1/predict
Real-time single-patient risk prediction using a saved model.
"""

from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PatientPredictRequest, PatientPredictResponse
from api.dependencies import load_model, get_model_info, get_latest_model_id

router = APIRouter()


def _build_recommendation(risk_label: str, disease: str, prob: float) -> str:
    pct = round(prob * 100, 1)
    if risk_label == "High Risk":
        return (
            f"HIGH RISK ({pct}%) — Immediate clinical review recommended. "
            f"Refer to specialist for {disease.replace('_', ' ')} assessment. "
            "Consider further diagnostics and preventive intervention."
        )
    else:
        return (
            f"LOW RISK ({pct}%) — Routine monitoring advised. "
            f"Continue standard {disease.replace('_', ' ')} prevention protocols."
        )


@router.post("/predict", response_model=PatientPredictResponse, tags=["Prediction"])
def predict(request: PatientPredictRequest):
    """
    Real-time single-patient risk prediction.
    Requires a previously trained model (via POST /analyse).
    """
    disease  = request.disease.value
    model_id = request.model_id

    if model_id is None:
        model_id = get_latest_model_id(disease)
        if model_id is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for '{disease}'. "
                       "Please upload a dataset via POST /api/v1/analyse first.",
            )

    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not in registry.")

    try:
        pipeline = load_model(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    patient_df = pd.DataFrame([request.patient])

    expected_cols = info["numeric_cols"] + info["categorical_cols"]
    for col in expected_cols:
        if col not in patient_df.columns:
            patient_df[col] = None

    patient_df = patient_df[expected_cols]

    try:
        pred       = pipeline.predict(patient_df)[0]
        proba      = float(pipeline.predict_proba(patient_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    risk_label = "High Risk" if pred == 1 else "Low Risk"

    return PatientPredictResponse(
        disease=disease,
        model_id=model_id,
        model_name=info["best_model"],
        risk_label=risk_label,
        risk_probability=round(proba, 4),
        risk_percent=f"{round(proba * 100, 1)}%",
        recommendation=_build_recommendation(risk_label, disease, proba),
    )
