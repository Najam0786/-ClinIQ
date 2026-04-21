"""
modules/stroke.py
------------------
Stroke Module — ClinIQ Disease Module

Accepts any stroke / cerebrovascular CSV and auto-detects the target.
Supports: Kaggle Stroke Prediction Dataset and similar clinical records.
"""

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from core import feature_engine
from core.model_selector import select_and_train, ModelResult
from core.target_detector import detect_target, select_features_auto


_DROP_ALWAYS = {"id", "patient_id", "index"}

_POSITIVE_LABELS = {
    "1", "1.0", "true", "yes", "y", "positive",
    "stroke", "occurred", "affected",
    "died", "dead", "deceased",
    "severe",
}

_PREFERRED_FEATURES = [
    "age", "gender", "sex",
    "hypertension", "heart_disease",
    "ever_married", "married",
    "work_type", "Residence_type", "residence_type",
    "avg_glucose_level", "glucose", "blood_glucose",
    "bmi",
    "smoking_status", "smoking",
    "alcohol_intake",
    "physical_activity",
    "stress_level",
]


def _engineer_stroke_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_col  = next((c for c in ["age", "Age"] if c in df.columns), None)
    gluc_col = next((c for c in ["avg_glucose_level", "glucose", "blood_glucose"] if c in df.columns), None)
    bmi_col  = next((c for c in ["bmi", "BMI"] if c in df.columns), None)
    htn_col  = next((c for c in ["hypertension"] if c in df.columns), None)
    hd_col   = next((c for c in ["heart_disease"] if c in df.columns), None)

    # Vascular risk score: age + hypertension + heart_disease
    risk_parts = []
    if age_col:
        df["sk_age_risk"] = (df[age_col] / 10).round(1)
        risk_parts.append("sk_age_risk")
    if htn_col:
        risk_parts.append(htn_col)
    if hd_col:
        risk_parts.append(hd_col)
    if len(risk_parts) >= 2:
        df["sk_vascular_risk_score"] = df[risk_parts].sum(axis=1)

    # Glucose risk bands
    if gluc_col:
        df["sk_high_glucose"]  = (df[gluc_col] >= 140).astype(int)
        df["sk_severe_glucose"] = (df[gluc_col] >= 200).astype(int)

    # Age-glucose interaction
    if age_col and gluc_col:
        df["sk_age_glucose_risk"] = df[age_col] * df[gluc_col] / 10000

    # Obesity flag
    if bmi_col:
        df["sk_obese"] = (df[bmi_col] >= 30).astype(int)

    # Age bands
    if age_col:
        df["sk_elderly"]     = (df[age_col] >= 65).astype(int)
        df["sk_very_elderly"] = (df[age_col] >= 80).astype(int)

    # Composite high-risk: elderly + hypertension + high glucose
    if age_col and htn_col and gluc_col:
        df["sk_high_risk_composite"] = (
            df["sk_elderly"] & df[htn_col].astype(bool) & df["sk_high_glucose"]
        ).astype(int)

    return df


def _map_target(y: pd.Series) -> pd.Series:
    uniq_raw  = y.dropna().unique()
    uniq_norm = {str(v).strip().lower() for v in uniq_raw}

    if len(uniq_norm) == 2:
        try:
            num_uniq = {float(v) for v in uniq_norm}
            if num_uniq <= {0.0, 1.0}:
                return pd.to_numeric(y, errors="coerce")
        except (ValueError, TypeError):
            pass
        matched = uniq_norm & _POSITIVE_LABELS
        if matched:
            pos_val = matched.pop()
            return y.map(lambda v: 1 if str(v).strip().lower() == pos_val else 0)
        sorted_vals = sorted(uniq_norm)
        return y.map(lambda v: 1 if str(v).strip().lower() == sorted_vals[1] else 0)

    return pd.to_numeric(y, errors="coerce")


def run(df: pd.DataFrame, target_col=None):
    """
    Returns
    -------
    result      : ModelResult
    target_used : str
    confidence  : float
    candidates  : list[(col, score)]
    """
    drop = [c for c in df.columns if c.lower().strip() in _DROP_ALWAYS]
    df = df.drop(columns=drop, errors="ignore")

    if target_col is not None:
        best_target = target_col
        confidence  = 1.0
        candidates  = [(target_col, 1.0)]
    else:
        best_target, confidence, candidates = detect_target(df)
        if best_target is None:
            raise ValueError(
                "Could not auto-detect the target column. Please use the Override "
                "dropdown to select your target (e.g. 'stroke', 'death', 'outcome')."
            )

    y_raw        = df[best_target].copy()
    feature_cols = select_features_auto(df, target_col=best_target)
    if len(feature_cols) == 0:
        raise ValueError("No usable feature columns found after filtering.")

    X_df = _engineer_stroke_features(df[feature_cols].copy())
    X_df = feature_engine.engineer(X_df)

    y          = _map_target(y_raw)
    valid_mask = y.notna()
    X_df       = X_df.loc[valid_mask].reset_index(drop=True)
    y          = y.loc[valid_mask].astype(int).reset_index(drop=True)

    n_classes    = y.nunique()
    uniq_display = list(df[best_target].dropna().unique()[:6])
    if n_classes < 2:
        raise ValueError(
            f"Target column '{best_target}' has only one class after preprocessing "
            f"(raw values: {uniq_display}). Please select a different target column."
        )
    if n_classes != 2:
        raise ValueError(
            f"Target column '{best_target}' has {n_classes} distinct classes "
            f"(sample values: {uniq_display}). "
            "ClinIQ requires exactly 2 classes (binary classification). "
            "Please select a binary outcome column, e.g. 0/1, Yes/No, disease/no disease."
        )

    _stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=_stratify
    )

    result = select_and_train(X_train, y_train, X_test, y_test)
    return result, best_target, confidence, candidates
