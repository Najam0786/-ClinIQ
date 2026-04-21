"""
modules/diabetes.py
--------------------
Diabetes Module — ClinIQ Disease Module

Accepts any diabetes CSV and auto-detects the target.
Supports: Pima Indians, 130-US Hospitals, Diabetic Retinopathy, etc.
"""

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from core import feature_engine
from core.model_selector import select_and_train, ModelResult
from core.target_detector import detect_target, select_features_auto


_DROP_ALWAYS = {"id", "patient_id", "encounter_id", "index"}

_POSITIVE_LABELS = {
    "1", "1.0", "true", "yes", "y", "positive",
    "diabetic", "diabetes", "disease",
    "readmitted", ">30", "<30",
    "proliferative", "advanced",
}

_PREFERRED_FEATURES = [
    # Pima Indians
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    # Clinical labs
    "hba1c", "hba1c_level", "fasting_glucose", "postprandial_glucose",
    "glucose", "blood_glucose_level",
    "bmi", "age", "gender", "sex",
    "cholesterol", "triglycerides", "hdl", "ldl",
    "creatinine", "egfr", "urea",
    "blood_pressure", "systolic_bp", "diastolic_bp",
    "smoking_history", "smoking", "hypertension", "heart_disease",
    # 130-US hospital features
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses",
    "a1cresult", "max_glu_serum",
    "metformin", "insulin", "change", "diabetesMed",
]


def _engineer_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Glucose-insulin ratio (insulin resistance proxy)
    gluc = next((c for c in ["Glucose", "glucose", "blood_glucose_level"] if c in df.columns), None)
    ins  = next((c for c in ["Insulin", "insulin"] if c in df.columns), None)
    if gluc and ins:
        df["db_glucose_insulin_ratio"] = df[gluc] / (df[ins].abs() + 1e-9)

    # BMI risk category
    bmi_col = next((c for c in ["BMI", "bmi"] if c in df.columns), None)
    if bmi_col:
        df["db_obese"]         = (df[bmi_col] >= 30).astype(int)
        df["db_severely_obese"]= (df[bmi_col] >= 35).astype(int)

    # Hyperglycaemia flag
    if gluc:
        df["db_hyperglycaemia"] = (df[gluc] >= 140).astype(int)
        df["db_severe_hyper"]   = (df[gluc] >= 200).astype(int)

    # Age-glucose interaction (older + high glucose → high risk)
    age_col = next((c for c in ["Age", "age"] if c in df.columns), None)
    if age_col and gluc:
        df["db_age_glucose_risk"] = df[age_col] * df[gluc] / 10000

    # Diabetes pedigree × BMI interaction
    dpf_col = next((c for c in ["DiabetesPedigreeFunction"] if c in df.columns), None)
    if dpf_col and bmi_col:
        df["db_genetic_bmi_risk"] = df[dpf_col] * df[bmi_col]

    # Elevated skin thickness (insulin resistance proxy)
    st_col = next((c for c in ["SkinThickness"] if c in df.columns), None)
    if st_col:
        df["db_high_skin_thickness"] = (df[st_col] > 30).astype(int)

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
                "dropdown to select your target (e.g. 'Outcome', 'diabetes', "
                "'diabetic', 'readmitted')."
            )

    y_raw        = df[best_target].copy()
    feature_cols = select_features_auto(df, target_col=best_target)
    if len(feature_cols) == 0:
        raise ValueError("No usable feature columns found after filtering.")

    X_df = _engineer_diabetes_features(df[feature_cols].copy())
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
