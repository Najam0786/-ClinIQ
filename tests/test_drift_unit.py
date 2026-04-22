"""
tests/test_drift_unit.py
Pure unit tests for core/drift_detector.py — no server required.
Run:  pytest tests/test_drift_unit.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.drift_detector as dd


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_drift_dir(tmp_path, monkeypatch):
    """Redirect all drift file I/O to a fresh temp dir per test."""
    monkeypatch.setattr(dd, "DRIFT_DIR", tmp_path)
    return tmp_path


def _baseline_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":    rng.integers(30, 70, n).astype(float),
        "bmi":    rng.normal(27, 4, n),
        "gender": rng.choice(["M", "F"], n),
    })


def _shifted_df(n=200, seed=42):
    """Same schema but clearly different distribution."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":    rng.integers(60, 90, n).astype(float),   # shifted up
        "bmi":    rng.normal(35, 4, n),                    # shifted up
        "gender": rng.choice(["M", "F", "Other"], n),      # new category
    })


# ── save_baseline / load_baseline ─────────────────────────────────────────────

def test_save_and_load_baseline_roundtrip():
    df = _baseline_df()
    dd.save_baseline("diabetes", df, list(df.columns))
    loaded = dd.load_baseline("diabetes")
    assert loaded is not None
    assert set(loaded.columns) == set(df.columns)
    assert len(loaded) == len(df)


def test_load_baseline_returns_none_when_missing():
    result = dd.load_baseline("nonexistent_disease")
    assert result is None


def test_save_baseline_filters_to_feature_cols():
    df = _baseline_df()
    df["target"] = np.random.randint(0, 2, len(df))
    dd.save_baseline("heart_disease", df, ["age", "bmi"])
    loaded = dd.load_baseline("heart_disease")
    assert "target" not in loaded.columns
    assert set(loaded.columns) == {"age", "bmi"}


# ── compute_drift ─────────────────────────────────────────────────────────────

def test_no_drift_on_identical_data():
    df = _baseline_df()
    dd.save_baseline("diabetes", df, list(df.columns))
    report = dd.compute_drift("diabetes", df.copy())
    assert report["drift_ratio"] < 0.5
    assert report["dataset_drift"] is False


def test_drift_detected_on_shifted_data():
    baseline = _baseline_df()
    current  = _shifted_df()
    dd.save_baseline("diabetes", baseline, list(baseline.columns))
    report = dd.compute_drift("diabetes", current)
    assert report["drift_ratio"] > 0
    assert len(report["drifted_features"]) >= 1


def test_alert_fires_at_30pct():
    baseline = _baseline_df()
    current  = _shifted_df()
    dd.save_baseline("stroke", baseline, list(baseline.columns))
    report = dd.compute_drift("stroke", current)
    if report["drift_ratio"] >= 0.3:
        assert report["alert"] is True


def test_report_contains_required_keys():
    df = _baseline_df()
    dd.save_baseline("diabetes", df, list(df.columns))
    report = dd.compute_drift("diabetes", df)
    for key in ("disease", "generated_at", "drift_ratio", "alert",
                "column_reports", "drifted_features", "summary"):
        assert key in report, f"Missing key: {key}"


def test_raises_when_no_baseline():
    with pytest.raises(ValueError, match="No baseline found"):
        dd.compute_drift("unknown_disease", _baseline_df())


def test_raises_when_no_shared_columns():
    baseline = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    current  = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(ValueError, match="no columns"):
        dd.compute_drift("diabetes", current, reference_df=baseline)


# ── _column_drift ─────────────────────────────────────────────────────────────

def test_column_drift_numerical_no_drift():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500), name="feature")
    result = dd._column_drift(s, s.copy())
    assert result["type"] == "numerical"
    assert result["test"] == "Kolmogorov-Smirnov"
    assert result["drift_detected"] is False


def test_column_drift_numerical_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0,  1, 500), name="age")
    cur = pd.Series(rng.normal(10, 1, 500), name="age")
    result = dd._column_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["drift_score"] > 0.5


def test_column_drift_categorical():
    ref = pd.Series(["M"] * 100 + ["F"] * 100, name="gender")
    cur = pd.Series(["M"] * 10  + ["F"] * 190, name="gender")
    result = dd._column_drift(ref, cur)
    assert result["type"] == "categorical"
    assert result["test"] == "Chi-Square"


# ── list_baselines ────────────────────────────────────────────────────────────

def test_list_baselines_empty_initially():
    result = dd.list_baselines()
    assert result == []


def test_list_baselines_after_save():
    df = _baseline_df()
    dd.save_baseline("diabetes",     df, list(df.columns))
    dd.save_baseline("heart_disease",df, list(df.columns))
    result = dd.list_baselines()
    diseases = [r["disease"] for r in result]
    assert "diabetes"      in diseases
    assert "heart_disease" in diseases


# ── Report persistence ────────────────────────────────────────────────────────

def test_report_saved_and_reloaded():
    df = _baseline_df()
    dd.save_baseline("stroke", df, list(df.columns))
    dd.compute_drift("stroke", df)
    report = dd.load_latest_report("stroke")
    assert report is not None
    assert report["disease"] == "stroke"


def test_load_latest_report_returns_none_when_missing():
    result = dd.load_latest_report("no_such_disease")
    assert result is None
