"""
tests/test_feature_engine_unit.py
Pure unit tests for core/feature_engine.py — no server required.
Run:  pytest tests/test_feature_engine_unit.py -v
"""

import numpy as np
import pandas as pd
import pytest

from core.feature_engine import engineer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(**kwargs):
    """Build a single-row DataFrame from keyword args."""
    return pd.DataFrame([kwargs])


# ── engineer() does not mutate input ─────────────────────────────────────────

def test_engineer_does_not_mutate_input():
    df = _row(age=55.0, bmi=28.0)
    cols_before = set(df.columns)
    engineer(df)
    assert set(df.columns) == cols_before


# ── Column name normalisation ─────────────────────────────────────────────────

def test_column_names_lowercased():
    df = pd.DataFrame([{"AGE": 40.0, "BMI": 27.0}])
    result = engineer(df)
    assert "age" in result.columns
    assert "bmi" in result.columns
    assert "AGE" not in result.columns


# ── Age group ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("age,expected", [
    (25,  "Young (<35)"),
    (45,  "Middle (35-55)"),
    (60,  "Senior (55-70)"),
    (80,  "Elderly (>70)"),
])
def test_age_group(age, expected):
    result = engineer(_row(age=float(age)))
    assert str(result["age_group"].iloc[0]) == expected


# ── BMI category ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bmi,expected", [
    (17.0, "Underweight"),
    (22.0, "Normal"),
    (27.0, "Overweight"),
    (32.0, "Obese_I"),
    (40.0, "Obese_II+"),
])
def test_bmi_category(bmi, expected):
    result = engineer(_row(bmi=bmi))
    assert str(result["bmi_category"].iloc[0]) == expected


# ── Blood pressure category ───────────────────────────────────────────────────

@pytest.mark.parametrize("sbp,dbp,expected", [
    (110, 70, "Normal"),
    (125, 75, "Elevated"),
    (150, 95, "High_Stage2"),
])
def test_bp_category(sbp, dbp, expected):
    result = engineer(_row(systolic_bp=sbp, diastolic_bp=dbp))
    assert result["bp_category"].iloc[0] == expected


# ── Polypharmacy flags ────────────────────────────────────────────────────────

def test_polypharmacy_flag_true():
    result = engineer(_row(n_meds=6))
    assert result["polypharmacy"].iloc[0] == 1


def test_polypharmacy_flag_false():
    result = engineer(_row(n_meds=3))
    assert result["polypharmacy"].iloc[0] == 0


def test_polypharmacy_discharge_flag():
    result = engineer(_row(n_meds_at_discharge=10))
    assert result["polypharmacy_discharge"].iloc[0] == 1


# ── Lab flags ─────────────────────────────────────────────────────────────────

def test_hyperglycaemia_flag():
    result = engineer(_row(avg_glucose=130.0))
    assert result["hyperglycaemia"].iloc[0] == 1


def test_no_hyperglycaemia_flag():
    result = engineer(_row(avg_glucose=100.0))
    assert result["hyperglycaemia"].iloc[0] == 0


def test_high_cholesterol_flag():
    result = engineer(_row(avg_cholesterol=210.0))
    assert result["high_cholesterol"].iloc[0] == 1


def test_anaemia_flag():
    result = engineer(_row(avg_hemoglobin=10.0))
    assert result["anaemia"].iloc[0] == 1


def test_no_anaemia_flag():
    result = engineer(_row(avg_hemoglobin=14.0))
    assert result["anaemia"].iloc[0] == 0


def test_hba1c_flag():
    result = engineer(_row(avg_hba1c=8.0))
    assert result["poor_glycaemic_control"].iloc[0] == 1


def test_creatinine_flag():
    result = engineer(_row(avg_creatinine=1.5))
    assert result["elevated_creatinine"].iloc[0] == 1


def test_leukocytosis_flag():
    result = engineer(_row(avg_wbc=12.0))
    assert result["leukocytosis"].iloc[0] == 1


# ── Lab composite ─────────────────────────────────────────────────────────────

def test_lab_composite_mean():
    result = engineer(_row(avg_glucose=120.0, avg_cholesterol=200.0, avg_hemoglobin=13.0))
    expected = (120 + 200 + 13) / 3
    assert result["lab_composite"].iloc[0] == pytest.approx(expected)


# ── CKD stage ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("egfr,expected", [
    (10,  "Stage5"),
    (20,  "Stage4"),
    (65,  "Stage2"),
    (95,  "Stage1"),
])
def test_ckd_stage(egfr, expected):
    result = engineer(_row(egfr=float(egfr)))
    assert str(result["ckd_stage"].iloc[0]) == expected


# ── Reduced ejection fraction ────────────────────────────────────────────────

def test_reduced_ef_flag():
    result = engineer(_row(ejection_fraction=35.0))
    assert result["reduced_ef"].iloc[0] == 1


def test_normal_ef_flag():
    result = engineer(_row(ejection_fraction=60.0))
    assert result["reduced_ef"].iloc[0] == 0


# ── Severe obstruction ────────────────────────────────────────────────────────

def test_severe_obstruction_flag():
    result = engineer(_row(fev1_pct=40.0))
    assert result["severe_obstruction"].iloc[0] == 1


# ── Readmission / LOS ────────────────────────────────────────────────────────

def test_long_stay_flag():
    result = engineer(_row(length_of_stay_days=10))
    assert result["long_stay"].iloc[0] == 1


def test_frequent_admission_flag():
    result = engineer(_row(prev_admissions_12m=3))
    assert result["frequent_admission"].iloc[0] == 1


# ── Comorbidity count ────────────────────────────────────────────────────────

def test_comorbidity_count():
    result = engineer(_row(has_diabetes=1, has_hypertension=1, has_heart_disease=0))
    assert result["comorbidity_count"].iloc[0] == 2


# ── Interaction features ──────────────────────────────────────────────────────

def test_metabolic_risk_score_created_when_glucose_present():
    result = engineer(_row(avg_glucose=130.0))
    assert "metabolic_risk_score" in result.columns
    assert result["metabolic_risk_score"].iloc[0] > 0


def test_age_comorbidity_risk():
    result = engineer(_row(age=70.0, has_diabetes=1, has_hypertension=1, has_heart_disease=1))
    assert "age_comorbidity_risk" in result.columns
    assert result["age_comorbidity_risk"].iloc[0] > 0


def test_lab_abnormality_count():
    result = engineer(_row(avg_glucose=130.0, avg_hba1c=8.0, avg_cholesterol=210.0))
    assert "lab_abnormality_count" in result.columns
    assert result["lab_abnormality_count"].iloc[0] == 3


# ── No crash on empty-feature DataFrame ──────────────────────────────────────

def test_engineer_minimal_df():
    df = pd.DataFrame({"outcome": [0, 1, 0]})
    result = engineer(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


# ── Current smoker flag ───────────────────────────────────────────────────────

def test_current_smoker_flag():
    result = engineer(_row(smoking_status="current"))
    assert result["current_smoker"].iloc[0] == 1


def test_non_smoker_flag():
    result = engineer(_row(smoking_status="never"))
    assert result["current_smoker"].iloc[0] == 0


# ── Low adherence ─────────────────────────────────────────────────────────────

def test_low_adherence_flag():
    result = engineer(_row(med_adherence_score=0.5))
    assert result["low_adherence"].iloc[0] == 1


def test_high_adherence_no_flag():
    result = engineer(_row(med_adherence_score=0.9))
    assert result["low_adherence"].iloc[0] == 0
