"""
api/schemas.py
--------------
Pydantic request / response schemas for the ClinIQ REST API.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class DiseaseModule(str, Enum):
    breast_cancer  = "breast_cancer"
    heart_disease  = "heart_disease"
    diabetes       = "diabetes"
    stroke         = "stroke"
    respiratory    = "respiratory"
    universal      = "universal"


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    trained_models_available: int


# ── Analyse (batch CSV upload) ────────────────────────────────────────────────

class CVScores(BaseModel):
    model_name: str
    mean_auc: float


class AnalyseResponse(BaseModel):
    disease:           str
    best_model:        str
    test_auc:          float
    cv_scores:         Dict[str, float]
    classification_report: str
    top_features:      List[Dict[str, Any]]
    total_patients:    int
    high_risk_count:   int
    low_risk_count:    int
    model_id:          str
    mlflow_run_id:     Optional[str] = None


# ── Single-patient prediction ─────────────────────────────────────────────────

class PatientPredictRequest(BaseModel):
    disease:    DiseaseModule
    model_id:   Optional[str] = Field(
        None,
        description="Specific model ID to use. Omit to use the latest trained model for this disease."
    )
    patient:    Dict[str, Any] = Field(
        ...,
        description="Feature key-value pairs matching the training columns.",
        example={"age": 55, "bmi": 28.4, "avg_glucose": 110.0}
    )


class PatientPredictResponse(BaseModel):
    disease:          str
    model_id:         str
    model_name:       str
    risk_label:       str
    risk_probability: float
    risk_percent:     str
    recommendation:   str


# ── Saved model registry ──────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_id:     str
    disease:      str
    best_model:   str
    test_auc:     float
    trained_at:   str
    n_features:   int
    mlflow_run_id: Optional[str] = None


class ModelsListResponse(BaseModel):
    count:  int
    models: List[ModelInfo]


# ── FHIR ──────────────────────────────────────────────────────────────────────

class FHIRPredictResponse(BaseModel):
    resourceType:     str = "RiskAssessment"
    status:           str = "final"
    subject:          Dict[str, str]
    disease:          str
    model_id:         str
    risk_label:       str
    risk_probability: float
    basis:            str
    note:             str
