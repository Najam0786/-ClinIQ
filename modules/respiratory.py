"""
modules/respiratory.py
-----------------------
Respiratory / Pneumonia Module — ClinIQ Disease Module

Accepts any respiratory disease CSV and auto-detects the target.
Supports: COVID-19 clinical records, Pneumonia datasets, ICU respiratory patients.
"""

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from core import feature_engine
from core.model_selector import select_and_train, ModelResult
from core.target_detector import detect_target, select_features_auto


_DROP_ALWAYS = {"id", "patient_id", "case_id", "index"}

_POSITIVE_LABELS = {
    "1", "1.0", "true", "yes", "y", "positive",
    "pneumonia", "covid", "infected", "affected",
    "icu", "admitted", "icu_admitted",
    "died", "dead", "deceased", "death",
    "severe", "critical",
}

_PREFERRED_FEATURES = [
    # Demographics
    "age", "gender", "sex",
    # Vitals
    "oxygen_saturation", "spo2", "o2_saturation",
    "respiratory_rate", "heart_rate", "pulse",
    "temperature", "fever",
    "systolic_bp", "diastolic_bp", "blood_pressure",
    # Symptoms
    "cough", "shortness_of_breath", "dyspnoea", "dyspnea",
    "chest_pain", "fatigue", "headache",
    "loss_of_smell", "anosmia", "loss_of_taste",
    # Labs
    "wbc", "white_blood_cells", "lymphocytes", "neutrophils",
    "crp", "c_reactive_protein", "ferritin",
    "d_dimer", "procalcitonin",
    "pcr_result", "ct_score", "ct_severity_score",
    "ldh", "interleukin_6",
    # Comorbidities
    "hypertension", "diabetes", "heart_disease", "obesity",
    "asthma", "copd", "chronic_disease",
    "smoking", "smoking_status",
    # ICU/ventilation
    "ventilator_required", "icu_stay", "icu_days",
    "comorbidities", "n_comorbidities",
]


def _engineer_respiratory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_col  = next((c for c in ["age", "Age"] if c in df.columns), None)
    spo2_col = next((c for c in ["oxygen_saturation", "spo2", "o2_saturation"] if c in df.columns), None)
    rr_col   = next((c for c in ["respiratory_rate"] if c in df.columns), None)
    crp_col  = next((c for c in ["crp", "c_reactive_protein"] if c in df.columns), None)
    temp_col = next((c for c in ["temperature", "fever"] if c in df.columns), None)

    # Hypoxia flag
    if spo2_col:
        df["rp_hypoxia"]        = (df[spo2_col] < 94).astype(int)
        df["rp_severe_hypoxia"] = (df[spo2_col] < 88).astype(int)

    # Tachypnoea flag
    if rr_col:
        df["rp_tachypnoea"] = (df[rr_col] > 20).astype(int)
        df["rp_severe_resp_distress"] = (df[rr_col] > 30).astype(int)

    # Severity composite: age + hypoxia + tachypnoea
    severity_parts = []
    if age_col:
        df["rp_age_risk"] = (df[age_col] >= 60).astype(int)
        severity_parts.append("rp_age_risk")
    if spo2_col:
        severity_parts.append("rp_hypoxia")
    if rr_col:
        severity_parts.append("rp_tachypnoea")
    if len(severity_parts) >= 2:
        df["rp_severity_score"] = df[severity_parts].sum(axis=1)

    # High CRP (inflammation)
    if crp_col:
        df["rp_high_crp"]    = (df[crp_col] > 10).astype(int)
        df["rp_extreme_crp"] = (df[crp_col] > 100).astype(int)

    # Fever flag
    if temp_col:
        df["rp_fever"]      = (df[temp_col] >= 37.5).astype(int)
        df["rp_high_fever"] = (df[temp_col] >= 39.0).astype(int)

    # Age-spo2 interaction (older + low O2 = highest risk)
    if age_col and spo2_col:
        df["rp_age_spo2_risk"] = df[age_col] * (100 - df[spo2_col]) / 1000

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
                "dropdown to select your target (e.g. 'pneumonia', 'covid_positive', "
                "'icu_admitted', 'death')."
            )

    y_raw        = df[best_target].copy()
    feature_cols = select_features_auto(df, target_col=best_target)
    if len(feature_cols) == 0:
        raise ValueError("No usable feature columns found after filtering.")

    X_df = _engineer_respiratory_features(df[feature_cols].copy())
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
