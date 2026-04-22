"""
tests/test_model_selector_unit.py
Pure unit tests for core/model_selector.py — no server required.
Run:  pytest tests/test_model_selector_unit.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from core.model_selector import ModelResult, select_and_train, _adaptive_models


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dataset(n=300, n_cat=1, seed=0):
    """Synthetic binary classification dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":     rng.integers(30, 80, n).astype(float),
        "bmi":     rng.normal(27, 5, n),
        "glucose": rng.normal(110, 25, n),
        "bp":      rng.normal(80, 12, n),
    })
    if n_cat:
        df["gender"] = rng.choice(["M", "F"], n)
    # target correlated with features so AUC > random
    score = df["age"] * 0.02 + df["bmi"] * 0.05 + df["glucose"] * 0.01
    df["target"] = (score > score.median()).astype(int)
    return df


def _split(df):
    target = df["target"]
    feats  = df.drop(columns=["target"])
    split  = int(len(df) * 0.8)
    return feats.iloc[:split], target.iloc[:split], feats.iloc[split:], target.iloc[split:]


# ── ModelResult contract ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def result():
    df = _make_dataset(300)
    X_tr, y_tr, X_te, y_te = _split(df)
    return select_and_train(X_tr, y_tr, X_te, y_te)


def test_returns_model_result(result):
    assert isinstance(result, ModelResult)


def test_best_model_name_is_string(result):
    assert isinstance(result.best_model_name, str)
    assert len(result.best_model_name) > 0


def test_auc_in_valid_range(result):
    assert 0.0 <= result.test_auc <= 1.0


def test_cv_scores_dict_nonempty(result):
    assert isinstance(result.cv_scores, dict)
    assert len(result.cv_scores) >= 1
    for v in result.cv_scores.values():
        assert 0.0 <= v <= 1.0


def test_pipeline_is_sklearn_pipeline(result):
    assert isinstance(result.pipeline, Pipeline)


def test_predictions_df_has_required_columns(result):
    for col in ("prediction", "risk_label", "risk_probability", "actual"):
        assert col in result.predictions_df.columns, f"Missing column: {col}"


def test_risk_labels_only_valid_values(result):
    labels = set(result.predictions_df["risk_label"].unique())
    assert labels <= {"High Risk", "Low Risk"}


def test_risk_probability_in_0_1(result):
    p = result.predictions_df["risk_probability"]
    assert (p >= 0).all() and (p <= 1).all()


def test_feature_importances_columns(result):
    assert "feature" in result.feature_importances.columns
    assert "importance" in result.feature_importances.columns


def test_feature_importances_nonnegative(result):
    assert (result.feature_importances["importance"] >= 0).all()


def test_fpr_tpr_arrays(result):
    assert len(result.fpr) == len(result.tpr)
    assert result.fpr[0] == pytest.approx(0.0, abs=0.01)
    assert result.tpr[-1] == pytest.approx(1.0, abs=0.01)


def test_confusion_matrix_shape(result):
    cm = result.confusion_matrix
    assert cm.shape == (2, 2)
    assert cm.sum() > 0


def test_numeric_categorical_cols_split(result):
    assert "age"    in result.numeric_cols
    assert "gender" in result.categorical_cols


def test_x_train_sample_is_dataframe(result):
    assert isinstance(result.X_train_sample, pd.DataFrame)
    assert len(result.X_train_sample) <= 150


# ── Pipeline predicts correctly ───────────────────────────────────────────────

def test_pipeline_predicts_on_new_data(result):
    new = pd.DataFrame({
        "age": [55.0], "bmi": [30.0], "glucose": [120.0],
        "bp": [85.0], "gender": ["M"],
    })
    proba = result.pipeline.predict_proba(new)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


# ── _adaptive_models selects correct tier ─────────────────────────────────────

def test_adaptive_models_small_dataset():
    models = _adaptive_models(500)
    assert len(models) >= 3


def test_adaptive_models_large_dataset():
    models = _adaptive_models(50_000)
    assert "Hist Gradient Boosting" in models
    rf = models.get("Random Forest")
    if rf is not None:
        assert rf.n_estimators <= 100


# ── Works without categorical columns ────────────────────────────────────────

def test_numeric_only_dataset():
    df = _make_dataset(200, n_cat=0)
    X_tr, y_tr, X_te, y_te = _split(df)
    result = select_and_train(X_tr, y_tr, X_te, y_te)
    assert isinstance(result, ModelResult)
    assert result.categorical_cols == []
