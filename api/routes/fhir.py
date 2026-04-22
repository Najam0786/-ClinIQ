"""
api/routes/fhir.py
------------------
POST /api/v1/fhir/predict
Accepts a simplified FHIR R4 Patient + Observation bundle,
maps it to ClinIQ features, runs prediction, returns a FHIR RiskAssessment.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_latest_model_id, get_model_info, load_model
from api.schemas import FHIRPredictResponse

router = APIRouter()


# ── FHIR R4 minimal inbound models ───────────────────────────────────────────

class FHIRCoding(BaseModel):
    system: Optional[str] = None
    code:   Optional[str] = None
    display: Optional[str] = None


class FHIRCodeableConcept(BaseModel):
    coding: Optional[List[FHIRCoding]] = None
    text:   Optional[str] = None


class FHIRQuantity(BaseModel):
    value: Optional[float] = None
    unit:  Optional[str]   = None


class FHIRObservation(BaseModel):
    resourceType:   str = "Observation"
    code:           Optional[FHIRCodeableConcept] = None
    valueQuantity:  Optional[FHIRQuantity]        = None
    valueBoolean:   Optional[bool]                = None
    valueString:    Optional[str]                 = None
    valueInteger:   Optional[int]                 = None


class FHIRPatient(BaseModel):
    resourceType: str = "Patient"
    id:           Optional[str] = None
    birthDate:    Optional[str] = None
    gender:       Optional[str] = None


class FHIRPredictRequest(BaseModel):
    disease:      str
    patient:      FHIRPatient
    observations: Optional[List[FHIRObservation]] = None
    model_id:     Optional[str] = None


# ── LOINC / SNOMED → ClinIQ feature name mapping ─────────────────────────────
_LOINC_MAP: Dict[str, str] = {
    "8302-2":  "height",
    "29463-7": "weight",
    "39156-5": "bmi",
    "8480-6":  "systolic_bp",
    "8462-4":  "diastolic_bp",
    "2339-0":  "avg_glucose",
    "4548-4":  "avg_hba1c",
    "2093-3":  "avg_cholesterol",
    "718-7":   "avg_hemoglobin",
    "2160-0":  "avg_creatinine",
    "55284-4": "blood_pressure",
    "8867-4":  "heart_rate",
}

_DISPLAY_MAP: Dict[str, str] = {
    "body mass index":  "bmi",
    "bmi":              "bmi",
    "glucose":          "avg_glucose",
    "hba1c":            "avg_hba1c",
    "cholesterol":      "avg_cholesterol",
    "hemoglobin":       "avg_hemoglobin",
    "creatinine":       "avg_creatinine",
    "systolic":         "systolic_bp",
    "diastolic":        "diastolic_bp",
    "blood pressure":   "systolic_bp",
}


def _map_observations(observations: List[FHIRObservation]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for obs in observations:
        feature_name = None

        if obs.code and obs.code.coding:
            for coding in obs.code.coding:
                if coding.code and coding.code in _LOINC_MAP:
                    feature_name = _LOINC_MAP[coding.code]
                    break

        if feature_name is None and obs.code:
            text = (obs.code.text or "").lower()
            for keyword, fname in _DISPLAY_MAP.items():
                if keyword in text:
                    feature_name = fname
                    break

        if feature_name is None:
            continue

        if obs.valueQuantity and obs.valueQuantity.value is not None:
            features[feature_name] = obs.valueQuantity.value
        elif obs.valueBoolean is not None:
            features[feature_name] = int(obs.valueBoolean)
        elif obs.valueInteger is not None:
            features[feature_name] = obs.valueInteger
        elif obs.valueString is not None:
            features[feature_name] = obs.valueString

    return features


@router.post("/fhir/predict", response_model=FHIRPredictResponse, tags=["FHIR"])
def fhir_predict(request: FHIRPredictRequest):
    """
    FHIR R4 clinical risk prediction.
    Send a Patient resource + Observation bundle → receive a RiskAssessment.
    Supports LOINC codes for automatic feature mapping.
    """
    disease  = request.disease.lower().replace(" ", "_")
    model_id = request.model_id or get_latest_model_id(disease)

    if model_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for '{disease}'. POST to /api/v1/analyse first.",
        )

    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    try:
        pipeline = load_model(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    features: Dict[str, Any] = {}

    if request.patient.gender:
        features["gender"] = request.patient.gender

    if request.observations:
        features.update(_map_observations(request.observations))

    patient_df = pd.DataFrame([features])
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
    pct = round(proba * 100, 1)

    return FHIRPredictResponse(
        resourceType="RiskAssessment",
        status="final",
        subject={"reference": f"Patient/{request.patient.id or 'unknown'}"},
        disease=disease,
        model_id=model_id,
        risk_label=risk_label,
        risk_probability=round(proba, 4),
        basis=f"ClinIQ v2.0 — {info['best_model']} (AUC {info['test_auc']})",
        note=(
            f"Risk probability {pct}%. "
            f"{'Immediate clinical review recommended.' if pred == 1 else 'Routine monitoring advised.'}"
        ),
    )
