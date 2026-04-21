"""
core/feature_engine.py
-----------------------
Medical-domain feature engineering applied automatically based on
which columns are present in the uploaded dataset.
All functions operate on copies — input is never mutated.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master entry point. Detects available columns and applies all
    relevant medical feature engineering transforms.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    cols = set(df.columns)

    if "age" in cols:
        df = _age_group(df)
    if "bmi" in cols:
        df = _bmi_category(df)
    if {"systolic_bp", "diastolic_bp"}.issubset(cols):
        df = _bp_category(df)
    if "n_meds" in cols:
        df = _polypharmacy(df)
    if "n_meds_at_discharge" in cols:
        df = _polypharmacy_discharge(df)
    if "n_visits" in cols:
        df = _high_utilisation(df)
    if {"avg_glucose", "avg_cholesterol", "avg_hemoglobin"}.issubset(cols):
        df = _lab_composite(df)
    if "avg_glucose" in cols:
        df = _glucose_flag(df)
    if "avg_hba1c" in cols:
        df = _hba1c_flag(df)
    if "avg_cholesterol" in cols:
        df = _cholesterol_flag(df)
    if "avg_creatinine" in cols:
        df = _creatinine_flag(df)
    if "avg_hemoglobin" in cols:
        df = _anaemia_flag(df)
    if "avg_wbc" in cols:
        df = _leukocytosis_flag(df)
    if {"has_diabetes", "has_hypertension", "has_heart_disease"}.issubset(cols):
        df = _comorbidity_count(df)
    if "smoking_status" in cols:
        df = _current_smoker(df)
    if "med_adherence_score" in cols:
        df = _low_adherence(df)
    if "length_of_stay_days" in cols:
        df = _long_stay(df)
    if "prev_admissions_12m" in cols:
        df = _frequent_flyer(df)
    if "egfr" in cols:
        df = _ckd_stage(df)
    if "ejection_fraction" in cols:
        df = _reduced_ef(df)
    if "fev1_pct" in cols:
        df = _severe_obstruction(df)
    if "hospitalizations_6m" in cols:
        df = _high_hospitalisation(df)

    # ── Interaction & composite features (always attempted) ───────────────
    df = _lab_abnormality_count(df)
    df = _age_comorbidity_interaction(df)
    df = _metabolic_risk_composite(df)
    df = _readmission_complexity(df)
    df = _adherence_severity_interaction(df)

    return df


# ── Demographic & Anthropometric ──────────────────────────────────────────

def _age_group(df: pd.DataFrame) -> pd.DataFrame:
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 35, 55, 70, 120],
        labels=["Young (<35)", "Middle (35-55)", "Senior (55-70)", "Elderly (>70)"]
    )
    return df


def _bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 35, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese_I", "Obese_II+"]
    )
    return df


def _bp_category(df: pd.DataFrame) -> pd.DataFrame:
    s = df["systolic_bp"]
    d = df["diastolic_bp"]
    conds = [
        (s < 120) & (d < 80),
        (s < 130) & (d < 80),
        (s < 140) | (d < 90),
        (s >= 140) | (d >= 90),
    ]
    cats = ["Normal", "Elevated", "High_Stage1", "High_Stage2"]
    df["bp_category"] = np.select(conds, cats, default="High_Stage2")
    return df


# ── Medication & Utilisation ───────────────────────────────────────────────

def _polypharmacy(df: pd.DataFrame) -> pd.DataFrame:
    df["polypharmacy"] = (df["n_meds"] >= 5).astype(int)
    return df


def _polypharmacy_discharge(df: pd.DataFrame) -> pd.DataFrame:
    df["polypharmacy_discharge"] = (df["n_meds_at_discharge"] >= 10).astype(int)
    return df


def _high_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    df["high_utilisation"] = (df["n_visits"] >= 5).astype(int)
    return df


# ── Lab Flags ─────────────────────────────────────────────────────────────

def _lab_composite(df: pd.DataFrame) -> pd.DataFrame:
    lab_cols = ["avg_glucose", "avg_cholesterol", "avg_hemoglobin"]
    df["lab_composite"] = df[lab_cols].mean(axis=1)
    return df


def _glucose_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["hyperglycaemia"] = (df["avg_glucose"] > 126).astype(int)
    return df


def _hba1c_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["poor_glycaemic_control"] = (df["avg_hba1c"] > 7.0).astype(int)
    return df


def _cholesterol_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["high_cholesterol"] = (df["avg_cholesterol"] > 200).astype(int)
    return df


def _creatinine_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["elevated_creatinine"] = (df["avg_creatinine"] > 1.3).astype(int)
    return df


def _anaemia_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["anaemia"] = (df["avg_hemoglobin"] < 12.0).astype(int)
    return df


def _leukocytosis_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["leukocytosis"] = (df["avg_wbc"] > 11.0).astype(int)
    return df


# ── Comorbidity ────────────────────────────────────────────────────────────

def _comorbidity_count(df: pd.DataFrame) -> pd.DataFrame:
    comorbidity_cols = [c for c in ["has_diabetes", "has_hypertension",
                                     "has_heart_disease", "has_ckd", "has_copd"]
                        if c in df.columns]
    df["comorbidity_count"] = df[comorbidity_cols].sum(axis=1)
    return df


def _current_smoker(df: pd.DataFrame) -> pd.DataFrame:
    df["current_smoker"] = (df["smoking_status"] == "current").astype(int)
    return df


# ── Readmission-specific ───────────────────────────────────────────────────

def _long_stay(df: pd.DataFrame) -> pd.DataFrame:
    df["long_stay"] = (df["length_of_stay_days"] > 7).astype(int)
    return df


def _frequent_flyer(df: pd.DataFrame) -> pd.DataFrame:
    df["frequent_admission"] = (df["prev_admissions_12m"] >= 3).astype(int)
    return df


# ── Disease Progression-specific ──────────────────────────────────────────

def _low_adherence(df: pd.DataFrame) -> pd.DataFrame:
    df["low_adherence"] = (df["med_adherence_score"] < 0.7).astype(int)
    return df


def _ckd_stage(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 15, 29, 44, 59, 89, float("inf")]
    labels = ["Stage5", "Stage4", "Stage3b", "Stage3a", "Stage2", "Stage1"]
    df["ckd_stage"] = pd.cut(df["egfr"], bins=bins, labels=labels)
    return df


def _reduced_ef(df: pd.DataFrame) -> pd.DataFrame:
    df["reduced_ef"] = (df["ejection_fraction"] < 40).astype(int)
    return df


def _severe_obstruction(df: pd.DataFrame) -> pd.DataFrame:
    df["severe_obstruction"] = (df["fev1_pct"] < 50).astype(int)
    return df


def _high_hospitalisation(df: pd.DataFrame) -> pd.DataFrame:
    df["high_hospitalisation"] = (df["hospitalizations_6m"] >= 2).astype(int)
    return df


# ── Interaction & Composite Features ──────────────────────────────────────

def _lab_abnormality_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count of clinically abnormal lab values per patient."""
    flags = []
    if "avg_glucose" in df.columns:
        flags.append((df["avg_glucose"] > 126).astype(int))
    if "avg_hba1c" in df.columns:
        flags.append((df["avg_hba1c"] > 7.0).astype(int))
    if "avg_cholesterol" in df.columns:
        flags.append((df["avg_cholesterol"] > 200).astype(int))
    if "avg_creatinine" in df.columns:
        flags.append((df["avg_creatinine"] > 1.3).astype(int))
    if "avg_hemoglobin" in df.columns:
        flags.append((df["avg_hemoglobin"] < 12.0).astype(int))
    if "avg_wbc" in df.columns:
        flags.append((df["avg_wbc"] > 11.0).astype(int))
    if "egfr" in df.columns:
        flags.append((df["egfr"] < 60).astype(int))
    if "ejection_fraction" in df.columns:
        flags.append((df["ejection_fraction"] < 40).astype(int))
    if "fev1_pct" in df.columns:
        flags.append((df["fev1_pct"] < 70).astype(int))
    if flags:
        df["lab_abnormality_count"] = sum(flags)
    return df


def _age_comorbidity_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Age × comorbidity burden — elderly + multimorbid = highest risk."""
    if "comorbidity_count" in df.columns and "age" in df.columns:
        df["age_comorbidity_risk"] = (
            (df["age"].fillna(50) / 100) * df["comorbidity_count"].fillna(0)
        ).round(3)
    elif "n_comorbidities" in df.columns and "age" in df.columns:
        df["age_comorbidity_risk"] = (
            (df["age"].fillna(50) / 100) * df["n_comorbidities"].fillna(0)
        ).round(3)
    return df


def _metabolic_risk_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted metabolic syndrome score from available markers."""
    score = pd.Series(0.0, index=df.index)
    if "avg_glucose" in df.columns:
        score += (df["avg_glucose"].fillna(df["avg_glucose"].median()) > 100).astype(float) * 1.5
    if "avg_hba1c" in df.columns:
        score += (df["avg_hba1c"].fillna(df["avg_hba1c"].median()) > 6.5).astype(float) * 2.0
    if "bmi" in df.columns:
        score += (df["bmi"].fillna(df["bmi"].median()) > 30).astype(float) * 1.0
    if "avg_cholesterol" in df.columns:
        score += (df["avg_cholesterol"].fillna(df["avg_cholesterol"].median()) > 200).astype(float) * 1.0
    if "systolic_bp" in df.columns:
        score += (df["systolic_bp"].fillna(df["systolic_bp"].median()) > 130).astype(float) * 1.0
    if score.sum() > 0:
        df["metabolic_risk_score"] = score.round(2)
    return df


def _readmission_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Complexity index for readmission: LOS × comorbidities proxy."""
    if "length_of_stay_days" in df.columns and "n_comorbidities" in df.columns:
        df["complexity_index"] = (
            (df["length_of_stay_days"].fillna(df["length_of_stay_days"].median()) *
             (df["n_comorbidities"].fillna(df["n_comorbidities"].median()) + 1))
            / 10.0
        ).round(2)
    if "prev_admissions_12m" in df.columns and "n_comorbidities" in df.columns:
        df["readmission_risk_index"] = (
            df["prev_admissions_12m"].fillna(0) * 2
            + df["n_comorbidities"].fillna(0)
        ).astype(float)
    return df


def _adherence_severity_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Low adherence × high severity = compounded progression risk."""
    if "med_adherence_score" in df.columns and "baseline_severity" in df.columns:
        df["non_adherence_severity"] = (
            (1 - df["med_adherence_score"].fillna(0.5))
            * df["baseline_severity"].fillna(2)
        ).round(3)
    return df
