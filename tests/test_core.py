"""
tests/test_core.py
------------------
Unit tests for ClinIQ v2.0: profiler, cleaner, feature_engine,
and end-to-end pipeline for all 5 disease modules.
Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from core.profiler import profile, DataProfile
from core.cleaner import clean
from core import feature_engine
import modules.breast_cancer  as mod_bc
import modules.heart_disease   as mod_hd
import modules.diabetes        as mod_db
import modules.stroke          as mod_sk
import modules.respiratory     as mod_rp
import modules.universal       as mod_u


# ── Synthetic helpers ──────────────────────────────────────────────────────

def make_clinical_df(n=300, seed=42):
    """Generic clinical DataFrame with common columns; binary target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":               rng.integers(20, 85, n),
        "gender":            rng.choice(["M", "F"], n),
        "bmi":               rng.uniform(18, 40, n).round(1),
        "avg_glucose":       rng.uniform(70, 200, n).round(1),
        "avg_hba1c":         rng.uniform(4.5, 12.0, n).round(1),
        "avg_hemoglobin":    rng.uniform(9, 18, n).round(1),
        "avg_creatinine":    rng.uniform(0.5, 3.0, n).round(2),
        "avg_cholesterol":   rng.uniform(130, 280, n).round(1),
        "systolic_bp":       rng.integers(90, 180, n),
        "diastolic_bp":      rng.integers(60, 110, n),
        "n_meds":            rng.integers(0, 12, n),
        "n_visits":          rng.integers(0, 15, n),
        "med_adherence_score": rng.uniform(0.3, 1.0, n).round(2),
        "hospitalizations_6m": rng.integers(0, 5, n),
        "smoking_status":    rng.choice(["never", "former", "current"], n),
        "high_risk":         rng.integers(0, 2, n),
    })
    return df


def make_heart_df(n=300, seed=1):
    """Heart-disease-like DataFrame (UCI columns)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":      rng.integers(29, 77, n),
        "sex":      rng.integers(0, 2, n),
        "cp":       rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 200, n),
        "chol":     rng.integers(126, 564, n),
        "fbs":      rng.integers(0, 2, n),
        "restecg":  rng.integers(0, 3, n),
        "thalach":  rng.integers(71, 202, n),
        "exang":    rng.integers(0, 2, n),
        "oldpeak":  rng.uniform(0, 6.2, n).round(1),
        "slope":    rng.integers(0, 3, n),
        "ca":       rng.integers(0, 4, n),
        "thal":     rng.integers(0, 4, n),
        "target":   rng.integers(0, 2, n),
    })
    return df


def make_stroke_df(n=300, seed=2):
    """Stroke-like DataFrame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":               rng.uniform(10, 82, n).round(1),
        "gender":            rng.choice(["Male", "Female", "Other"], n),
        "hypertension":      rng.integers(0, 2, n),
        "heart_disease":     rng.integers(0, 2, n),
        "ever_married":      rng.choice(["Yes", "No"], n),
        "work_type":         rng.choice(["Private", "Self-employed", "Govt_job", "children"], n),
        "Residence_type":    rng.choice(["Urban", "Rural"], n),
        "avg_glucose_level": rng.uniform(55, 272, n).round(2),
        "bmi":               rng.uniform(10, 48, n).round(1),
        "smoking_status":    rng.choice(["formerly smoked", "never smoked", "smokes", "Unknown"], n),
        "stroke":            rng.integers(0, 2, n),
    })
    return df


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def generic_df():
    return make_clinical_df()

@pytest.fixture(scope="module")
def heart_df():
    return make_heart_df()

@pytest.fixture(scope="module")
def stroke_df():
    return make_stroke_df()

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "age":         [45, 70, 30, 55, -5, np.nan],
        "gender":      ["M", "F", "M", "F", "M", None],
        "avg_glucose": [90.0, 150.0, -50.0, np.nan, 110.0, 95.0],
        "high_risk":   [0, 1, 0, 1, 0, 1],
    })

@pytest.fixture(scope="module")
def bc_df():
    """Real Wisconsin breast cancer CSV (if present), else skip."""
    path = Path(__file__).parent.parent / "sample_data" / "breast_cancer" / "breast_cancer_wisconsin.csv"
    if not path.exists():
        pytest.skip("Breast cancer CSV not present")
    return pd.read_csv(path)


# ── Profiler ───────────────────────────────────────────────────────────────

class TestProfiler:

    def test_returns_dataprofile(self, simple_df):
        assert isinstance(profile(simple_df), DataProfile)

    def test_detects_row_count(self, simple_df):
        assert profile(simple_df).n_rows == len(simple_df)

    def test_detects_column_count(self, simple_df):
        assert profile(simple_df).n_cols == len(simple_df.columns)

    def test_detects_missing_values(self, simple_df):
        p = profile(simple_df)
        assert p.columns["avg_glucose"].missing_count >= 1
        assert p.columns["gender"].missing_count >= 1

    def test_detects_invalid_values(self, simple_df):
        p = profile(simple_df)
        assert p.columns["avg_glucose"].invalid_count >= 1
        assert p.columns["age"].invalid_count >= 1

    def test_quality_score_range(self, simple_df):
        assert 0 <= profile(simple_df).quality_score <= 100

    def test_quality_label_valid(self, simple_df):
        assert profile(simple_df).quality_label in {"Excellent", "Good", "Fair", "Poor"}

    def test_warnings_is_list(self, simple_df):
        assert isinstance(profile(simple_df).warnings, list)

    def test_clean_data_scores_high(self):
        df = pd.DataFrame({"age": [30, 40, 50], "gender": ["M", "F", "M"]})
        assert profile(df).quality_score >= 80

    def test_large_dataset(self, generic_df):
        assert profile(generic_df).n_rows == len(generic_df)


# ── Cleaner ────────────────────────────────────────────────────────────────

class TestCleaner:

    def test_returns_df_and_log(self, simple_df):
        df_out, log = clean(simple_df, profile(simple_df))
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(log, list)

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "age": [30, 30, 40],
            "avg_glucose": [90.0, 90.0, 110.0],
            "high_risk": [0, 0, 1],
        })
        df_out, _ = clean(df, profile(df))
        assert len(df_out) < len(df)

    def test_fixes_negative_values(self):
        df = pd.DataFrame({
            "avg_glucose": [-50.0, 90.0, 120.0, 80.0],
            "high_risk": [0, 1, 0, 1],
        })
        df_out, _ = clean(df, profile(df))
        assert (df_out["avg_glucose"] >= 0).all()

    def test_imputes_missing_numeric(self, simple_df):
        df_out, _ = clean(simple_df, profile(simple_df))
        assert df_out["avg_glucose"].isna().sum() == 0

    def test_imputes_missing_categorical(self, simple_df):
        df_out, _ = clean(simple_df, profile(simple_df))
        assert df_out["gender"].isna().sum() == 0

    def test_does_not_mutate_input(self, simple_df):
        original_len = len(simple_df)
        clean(simple_df, profile(simple_df))
        assert len(simple_df) == original_len

    def test_log_has_required_keys(self, simple_df):
        _, log = clean(simple_df, profile(simple_df))
        for entry in log:
            assert "step" in entry
            assert "rows_affected" in entry


# ── Feature Engine ─────────────────────────────────────────────────────────

class TestFeatureEngine:

    def test_adds_age_group(self, generic_df):
        assert "age_group" in feature_engine.engineer(generic_df).columns

    def test_adds_bmi_category(self, generic_df):
        assert "bmi_category" in feature_engine.engineer(generic_df).columns

    def test_adds_polypharmacy_flag(self, generic_df):
        out = feature_engine.engineer(generic_df)
        assert "polypharmacy" in out.columns
        assert set(out["polypharmacy"].dropna().unique()).issubset({0, 1})

    def test_adds_hyperglycaemia_flag(self, generic_df):
        assert "hyperglycaemia" in feature_engine.engineer(generic_df).columns

    def test_does_not_mutate_input(self, generic_df):
        cols_before = list(generic_df.columns)
        feature_engine.engineer(generic_df)
        assert list(generic_df.columns) == cols_before

    def test_output_has_more_cols_than_input(self, generic_df):
        out = feature_engine.engineer(generic_df)
        assert len(out.columns) >= len(generic_df.columns)


# ── End-to-End Pipeline ────────────────────────────────────────────────────

def _run_module(runner, df, target_col=None):
    """Clean → run module → return ModelResult."""
    prof = profile(df)
    df_clean, _ = clean(df, prof)
    result, *_ = runner(df_clean, target_col=target_col)
    return result


class TestBreastCancerModule:

    def test_runs_on_wisconsin_csv(self, bc_df):
        result, _, _, _, _ = mod_bc.run(bc_df, target_col="diagnosis")
        assert 0 <= result.test_auc <= 1

    def test_predictions_binary(self, bc_df):
        result, *_ = mod_bc.run(bc_df, target_col="diagnosis")
        assert set(result.predictions_df["prediction"].unique()).issubset({0, 1})

    def test_probabilities_in_range(self, bc_df):
        result, *_ = mod_bc.run(bc_df, target_col="diagnosis")
        p = result.predictions_df["risk_probability"]
        assert (p >= 0).all() and (p <= 1).all()

    def test_feature_importances_not_empty(self, bc_df):
        result, *_ = mod_bc.run(bc_df, target_col="diagnosis")
        assert not result.feature_importances.empty

    def test_cv_scores_present(self, bc_df):
        result, *_ = mod_bc.run(bc_df, target_col="diagnosis")
        assert len(result.cv_scores) >= 1


class TestUniversalModule:

    def test_runs_on_generic_df(self, generic_df):
        prof = profile(generic_df)
        df_clean, _ = clean(generic_df, prof)
        result, _, _, _ = mod_u.run(df_clean, target_col="high_risk")
        assert 0 <= result.test_auc <= 1

    def test_predictions_and_probs(self, generic_df):
        prof = profile(generic_df)
        df_clean, _ = clean(generic_df, prof)
        result, *_ = mod_u.run(df_clean, target_col="high_risk")
        assert "prediction" in result.predictions_df.columns
        assert "risk_probability" in result.predictions_df.columns

    def test_rejects_non_binary_target(self):
        df = pd.DataFrame({
            "feat": range(50),
            "multi_target": list(range(10)) * 5,
        })
        prof = profile(df)
        df_clean, _ = clean(df, prof)
        with pytest.raises(ValueError, match="binary"):
            mod_u.run(df_clean, target_col="multi_target")


class TestHeartDiseaseModule:

    def test_runs_on_synthetic_data(self, heart_df):
        prof = profile(heart_df)
        df_clean, _ = clean(heart_df, prof)
        result, *_ = mod_hd.run(df_clean, target_col="target")
        assert 0 <= result.test_auc <= 1

    def test_predictions_binary(self, heart_df):
        prof = profile(heart_df)
        df_clean, _ = clean(heart_df, prof)
        result, *_ = mod_hd.run(df_clean, target_col="target")
        assert set(result.predictions_df["prediction"].unique()).issubset({0, 1})


class TestStrokeModule:

    def test_runs_on_synthetic_data(self, stroke_df):
        prof = profile(stroke_df)
        df_clean, _ = clean(stroke_df, prof)
        result, *_ = mod_sk.run(df_clean, target_col="stroke")
        assert 0 <= result.test_auc <= 1


class TestDiabetesModule:

    def test_runs_on_generic_df(self, generic_df):
        df = generic_df.rename(columns={"high_risk": "Outcome"})
        prof = profile(df)
        df_clean, _ = clean(df, prof)
        result, *_ = mod_db.run(df_clean, target_col="Outcome")
        assert 0 <= result.test_auc <= 1
