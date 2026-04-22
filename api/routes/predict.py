"""
api/routes/predict.py
---------------------
POST /api/v1/predict
Real-time single-patient risk prediction using a saved model.
Auth is optional — predictions work without a token, but are logged to DB when a token is present.
"""

from __future__ import annotations
import json
import uuid
import pandas as pd
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from api.schemas import PatientPredictRequest, PatientPredictResponse
from api.dependencies import load_model, get_model_info, get_latest_model_id
from db.database import get_db
from db.models import PredictionLog, User

router = APIRouter()

_optional_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def _get_optional_user(
    token: Optional[str] = Depends(_optional_oauth2),
    db:    Session        = Depends(get_db),
) -> Optional[User]:
    if not token:
        return None
    try:
        from api.auth_utils import decode_token
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        return db.query(User).filter(User.id == user_id, User.is_active == True).first()
    except Exception:
        return None


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
def predict(
    request:      PatientPredictRequest,
    http_request: Request,
    db:           Session        = Depends(get_db),
    current_user: Optional[User] = Depends(_get_optional_user),
):
    """
    Real-time single-patient risk prediction.
    Auth is optional — provide a Bearer token to log the prediction to the audit trail.
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
        pred  = pipeline.predict(patient_df)[0]
        proba = float(pipeline.predict_proba(patient_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    risk_label = "High Risk" if pred == 1 else "Low Risk"

    if current_user is not None:
        try:
            ip = http_request.client.host if http_request.client else "unknown"
            log = PredictionLog(
                id=uuid.uuid4(),
                user_id=current_user.id,
                disease=disease,
                model_id=model_id,
                model_name=info["best_model"],
                hospital=current_user.hospital,
                patient_data=json.dumps(request.patient),
                risk_label=risk_label,
                risk_probability=round(proba, 4),
                ip_address=ip,
            )
            db.add(log)
            db.commit()
        except Exception:
            db.rollback()

    return PatientPredictResponse(
        disease=disease,
        model_id=model_id,
        model_name=info["best_model"],
        risk_label=risk_label,
        risk_probability=round(proba, 4),
        risk_percent=f"{round(proba * 100, 1)}%",
        recommendation=_build_recommendation(risk_label, disease, proba),
    )
