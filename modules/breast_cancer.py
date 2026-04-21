"""
modules/breast_cancer.py
-------------------------
Breast Cancer Module — ClinIQ Flagship Disease Module

Accepts ANY breast cancer CSV and auto-detects the problem type:
  • Diagnosis   — Malignant vs Benign  (Wisconsin morphology features)
  • Staging     — Early vs Late stage  (clinical staging features)
  • Survival    — Survived vs Deceased
  • Recurrence  — Recurred vs No recurrence

Returns ModelResult + problem_type string for rich dashboard display.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from core import feature_engine
from core.model_selector import select_and_train, ModelResult
from core.target_detector import detect_target, select_features_auto


# ── Columns always excluded ─────────────────────────────────────────────────
_DROP_ALWAYS = {
    "id", "patient_id", "sample_id",
    "unnamed: 32", "unnamed:_32", "unnamed:32",
}

# ── Positive-class label mapping ────────────────────────────────────────────
_POSITIVE_LABELS = {
    # numeric
    "1", "1.0",
    # generic
    "true", "yes", "y", "positive", "present",
    # oncology diagnosis
    "m", "malignant", "cancer", "tumor",
    # staging
    "late", "advanced", "high", "iii", "iv",
    # survival
    "died", "dead", "deceased",
    # recurrence
    "recurrence-events", "recurred", "relapse", "metastasis",
}

# ── Wisconsin morphology feature groups ─────────────────────────────────────
_MORPHOLOGY_BASE = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
]

# ── Problem-type detection keywords ─────────────────────────────────────────
_SURVIVAL_KW    = {"survival", "vital_status", "death", "died", "deceased",
                   "alive", "os_status", "overall_survival"}
_RECURRENCE_KW  = {"recurrence", "recur", "relapse", "metastasis", "distant"}
_STAGING_KW     = {"stage", "staging", "tnm", "lymph_node", "node_positive"}


def _detect_problem_type(target_col: str, y: pd.Series) -> str:
    col = target_col.lower()
    vals = {str(v).strip().lower() for v in y.dropna().unique()}

    if any(kw in col for kw in _RECURRENCE_KW):
        return "recurrence"
    if any(kw in col for kw in _SURVIVAL_KW):
        return "survival"
    if any(kw in col for kw in _STAGING_KW):
        return "staging"
    if vals & {"recurrence-events", "no-recurrence-events", "recurred"}:
        return "recurrence"
    if vals & {"alive", "dead", "deceased", "survived", "died"}:
        return "survival"
    return "diagnosis"


def _engineer_bc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Breast-cancer-specific composite features."""
    df = df.copy()

    # ── Size composite (radius × area / perimeter) ─────────────────────────
    if all(c in df.columns for c in ["radius_mean", "area_mean", "perimeter_mean"]):
        df["bc_size_score"] = (
            df["radius_mean"] * df["area_mean"]
            / (df["perimeter_mean"].abs() + 1e-9)
        )

    # ── Worst-to-mean ratios for key morphology features ──────────────────
    for feat in ["radius", "texture", "area", "concavity", "compactness"]:
        m, w = f"{feat}_mean", f"{feat}_worst"
        if m in df.columns and w in df.columns:
            df[f"bc_{feat}_worst_ratio"] = df[w] / (df[m].abs() + 1e-9)

    # ── Malignancy index (compactness × concavity × concave_points) ───────
    needed = ["compactness_mean", "concavity_mean", "concave_points_mean"]
    if all(c in df.columns for c in needed):
        df["bc_malignancy_index"] = (
            df["compactness_mean"]
            * df["concavity_mean"]
            * df["concave_points_mean"]
        )

    # ── Symmetry irregularity (worst − mean) ─────────────────────────────
    if "symmetry_mean" in df.columns and "symmetry_worst" in df.columns:
        df["bc_symmetry_irregularity"] = (
            df["symmetry_worst"] - df["symmetry_mean"]
        )

    # ── Texture heterogeneity (SE / mean) ────────────────────────────────
    if "texture_mean" in df.columns and "texture_se" in df.columns:
        df["bc_texture_heterogeneity"] = (
            df["texture_se"] / (df["texture_mean"].abs() + 1e-9)
        )

    # ── Concavity severity (worst / SE as instability ratio) ─────────────
    if "concavity_worst" in df.columns and "concavity_se" in df.columns:
        df["bc_concavity_severity"] = (
            df["concavity_worst"] / (df["concavity_se"].abs() + 1e-9)
        )

    # ── Mean morphology composite score ───────────────────────────────────
    mean_cols = [f"{b}_mean" for b in _MORPHOLOGY_BASE if f"{b}_mean" in df.columns]
    if len(mean_cols) >= 3:
        df["bc_mean_composite"] = df[mean_cols].mean(axis=1)

    return df


def _map_target(y: pd.Series) -> pd.Series:
    """Convert any breast cancer binary label to integer 0/1."""
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
        # Fallback: alphabetically higher string → 1
        sorted_vals = sorted(uniq_norm)
        return y.map(lambda v: 1 if str(v).strip().lower() == sorted_vals[1] else 0)

    return pd.to_numeric(y, errors="coerce")


def run(
    df: pd.DataFrame,
    target_col=None,
):
    """
    Parameters
    ----------
    df         : raw breast cancer DataFrame — any schema
    target_col : override column; if None, auto-detected

    Returns
    -------
    result       : ModelResult (best trained model metrics)
    target_used  : str   — column used as target
    problem_type : str   — 'diagnosis' | 'staging' | 'survival' | 'recurrence'
    confidence   : float — detection confidence (1.0 if user-supplied)
    candidates   : list[(col, score)]
    """
    # ── Drop always-excluded columns ──────────────────────────────────────
    drop = [c for c in df.columns if c.lower().strip() in _DROP_ALWAYS]
    df = df.drop(columns=drop, errors="ignore")

    # ── Step 1: Target detection ──────────────────────────────────────────
    if target_col is not None:
        best_target = target_col
        confidence  = 1.0
        candidates  = [(target_col, 1.0)]
    else:
        best_target, confidence, candidates = detect_target(df)
        if best_target is None:
            raise ValueError(
                "Could not auto-detect a binary target column in your breast "
                "cancer dataset. Please use the Override dropdown to select "
                "your target column (e.g. 'diagnosis', 'stage', 'survived', "
                "'recurrence')."
            )

    # ── Step 2: Identify problem type ─────────────────────────────────────
    y_raw        = df[best_target].copy()
    problem_type = _detect_problem_type(best_target, y_raw)

    # ── Step 3: Feature selection ─────────────────────────────────────────
    feature_cols = select_features_auto(df, target_col=best_target)
    if len(feature_cols) == 0:
        raise ValueError(
            "No usable feature columns found after filtering. "
            "Please check that your CSV contains numeric predictor columns."
        )

    # ── Step 4: Feature engineering ───────────────────────────────────────
    X_df = _engineer_bc_features(df[feature_cols].copy())
    X_df = feature_engine.engineer(X_df)

    # ── Step 5: Map target → 0 / 1 ───────────────────────────────────────
    y          = _map_target(y_raw)
    valid_mask = y.notna()
    X_df       = X_df.loc[valid_mask].reset_index(drop=True)
    y          = y.loc[valid_mask].astype(int).reset_index(drop=True)

    n_classes = y.nunique()
    uniq_display = list(df[best_target].dropna().unique()[:6])
    if n_classes < 2:
        raise ValueError(
            f"Target column '{best_target}' has only one class after "
            f"preprocessing (raw values: {uniq_display}). "
            "Please select a different target column."
        )

    # ── Step 6: Train ──────────────────────────────────────────────────────
    _stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=_stratify
    )

    result = select_and_train(X_train, y_train, X_test, y_test)

    return result, best_target, problem_type, confidence, candidates
