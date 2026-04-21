"""
modules/heart_disease.py
-------------------------
Heart Disease Module — ClinIQ Disease Module

Accepts any heart disease / cardiovascular CSV and auto-detects the target.
Supports: UCI Heart Disease, Cleveland, Heart Failure Clinical Records, etc.
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
    "disease", "presence", "high_risk",
    "died", "dead", "deceased", "death",
    "heart disease", "heart_disease",
}

# Preferred feature names across known heart disease datasets
_PREFERRED_FEATURES = [
    "age", "sex", "gender",
    "cp", "chest_pain", "chest_pain_type",
    "trestbps", "resting_bp", "blood_pressure",
    "chol", "cholesterol", "serum_cholesterol",
    "fbs", "fasting_blood_sugar",
    "restecg", "resting_ecg",
    "thalach", "max_heart_rate",
    "exang", "exercise_angina",
    "oldpeak", "st_depression",
    "slope", "st_slope",
    "ca", "num_vessels",
    "thal", "thalassemia",
    # Heart failure features
    "ejection_fraction", "serum_creatinine", "serum_sodium",
    "creatinine_phosphokinase", "platelets",
    "anaemia", "diabetes", "high_blood_pressure", "smoking",
    "time",
]


def _engineer_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cholesterol risk: chol / age
    if "chol" in df.columns and "age" in df.columns:
        df["hd_chol_age_ratio"] = df["chol"] / (df["age"].abs() + 1e-9)

    # Resting BP severity
    bp_col = next((c for c in ["trestbps", "resting_bp", "blood_pressure"] if c in df.columns), None)
    if bp_col and "age" in df.columns:
        df["hd_bp_age_risk"] = df[bp_col] * df["age"] / 1000

    # Heart rate efficiency: max heart rate vs age
    hr_col = next((c for c in ["thalach", "max_heart_rate"] if c in df.columns), None)
    if hr_col and "age" in df.columns:
        df["hd_heart_rate_reserve"] = (220 - df["age"]) - df[hr_col]
        df["hd_hr_age_ratio"] = df[hr_col] / (df["age"].abs() + 1e-9)

    # ST depression severity (oldpeak normalised by slope if available)
    if "oldpeak" in df.columns:
        df["hd_st_severity"] = df["oldpeak"].clip(lower=0)

    # Ejection fraction risk bands (heart failure datasets)
    if "ejection_fraction" in df.columns:
        df["hd_reduced_ef"] = (df["ejection_fraction"] < 40).astype(int)

    # Creatinine risk (kidney-heart axis)
    if "serum_creatinine" in df.columns:
        df["hd_renal_risk"] = (df["serum_creatinine"] > 1.5).astype(int)

    return df


def _map_target(y: pd.Series) -> pd.Series:
    uniq_raw  = y.dropna().unique()
    uniq_norm = {str(v).strip().lower() for v in uniq_raw}

    if len(uniq_norm) == 2:
        try:
            num_uniq = {float(v) for v in uniq_norm}
            if num_uniq <= {0.0, 1.0}:
                return pd.to_numeric(y, errors="coerce")
            # Handle 0/2 or 1/2 numeric encoding
            if len(num_uniq) == 2:
                pos = max(num_uniq)
                return y.map(lambda v: 1 if float(v) == pos else 0)
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
                "dropdown to select your target (e.g. 'target', 'condition', "
                "'DEATH_EVENT', 'heart_disease')."
            )

    y_raw        = df[best_target].copy()
    feature_cols = select_features_auto(df, target_col=best_target)
    if len(feature_cols) == 0:
        raise ValueError("No usable feature columns found after filtering.")

    X_df = _engineer_heart_features(df[feature_cols].copy())
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
