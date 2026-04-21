"""
modules/universal.py
---------------------
Universal Module — accepts ANY clinical/health dataset.

Pipeline:
  1. Auto-detect the binary target column (or accept override)
  2. Auto-select useful feature columns (drop IDs, sparse, constant)
  3. Apply domain feature engineering where columns match known patterns
  4. Train the 4-model ensemble and return the best result

This means a user can upload a completely unknown CSV and ClinIQ will
figure out what to predict, which features to use, and return a
production-quality ML result — no configuration required.
"""

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from core import feature_engine
from core.model_selector import select_and_train, ModelResult
from core.target_detector import detect_target, select_features_auto


def run(
    df: pd.DataFrame,
    target_col=None,
):
    """
    Parameters
    ----------
    df         : raw (or cleaned) DataFrame — any schema
    target_col : if provided, skip auto-detection and use this column

    Returns
    -------
    result      : ModelResult from the best trained model
    target_used : the column that was used as the target
    confidence  : auto-detection confidence (1.0 if user-provided)
    candidates  : list of (col, score) candidates from auto-detection
    """
    # ── Step 1: Target detection ───────────────────────────────────────────
    if target_col is not None:
        # User-supplied — skip detection
        best_target  = target_col
        confidence   = 1.0
        candidates   = [(target_col, 1.0)]
    else:
        best_target, confidence, candidates = detect_target(df)
        if best_target is None:
            raise ValueError(
                "ClinIQ could not automatically detect a binary target column in "
                "your dataset. Please use the dropdown to specify the column you "
                "want to predict."
            )

    # ── Step 2: Feature selection ──────────────────────────────────────────
    feature_cols = select_features_auto(df, target_col=best_target)
    if len(feature_cols) == 0:
        raise ValueError(
            "No usable feature columns found after filtering. "
            "Please check that your CSV has numeric or categorical predictor columns."
        )

    # ── Step 3: Feature engineering (pattern-based, safe for unknown schemas) ─
    X_df = feature_engine.engineer(df[feature_cols].copy())

    # Re-align feature_cols after engineering (new columns added)
    feature_cols_engineered = [c for c in X_df.columns]

    y = df[best_target].copy()

    # ── Map target to 0/1 ─────────────────────────────────────────────────
    # Mapping: known string → 1 (positive class)
    _POSITIVE_LABELS = {
        "1", "1.0", "true", "yes", "y", "positive", "present",
        "readmitted", "worsened", "died", "dead", "deceased",
        "abnormal", "high", "poor", "unstable", "adverse",
        "female", "f",
        # Oncology
        "m", "malignant", "cancer", "tumor",
        # Kidney / chronic disease
        "ckd",
        # Stroke / cardiovascular
        "stroke",
    }
    uniq_raw  = y.dropna().unique()
    uniq_norm = {str(v).strip().lower() for v in uniq_raw}

    if len(uniq_norm) == 2:
        # Try numeric first
        try:
            num_uniq = {float(v) for v in uniq_norm}
            if num_uniq <= {0.0, 1.0}:
                y = pd.to_numeric(y, errors="coerce")
            else:
                raise ValueError("not 0/1")
        except (ValueError, TypeError):
            # Try known label mapping
            if uniq_norm & _POSITIVE_LABELS:
                pos_val = (uniq_norm & _POSITIVE_LABELS).pop()
                y = y.map(lambda v: 1 if str(v).strip().lower() == pos_val else 0)
            else:
                # Unknown pair: map alphabetically (higher string → 1)
                sorted_vals = sorted(uniq_norm)
                y = y.map(lambda v: 1 if str(v).strip().lower() == sorted_vals[1] else 0)
    elif len(uniq_norm) > 2:
        # More than 2 values — try numeric coercion (may still be 0/1 with NaN)
        y = pd.to_numeric(y, errors="coerce")
    else:
        y = pd.to_numeric(y, errors="coerce")

    # Drop rows where target is NaN after conversion
    valid_mask = y.notna()
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y    = y.loc[valid_mask].astype(int).reset_index(drop=True)

    n_classes    = y.nunique()
    uniq_display = list(df[best_target].dropna().unique()[:6])
    if n_classes < 2:
        raise ValueError(
            f"Target column '{best_target}' has only 1 distinct value after "
            f"preprocessing (raw values: {uniq_display}). "
            "Please select a different target column from the Override dropdown."
        )
    if n_classes != 2:
        raise ValueError(
            f"Target column '{best_target}' has {n_classes} distinct classes "
            f"(sample values: {uniq_display}). "
            "ClinIQ performs binary classification only (exactly 2 classes: e.g. 0/1, Yes/No, disease/no disease). "
            "This dataset may be a surveillance or regression dataset rather than a patient-level clinical dataset. "
            "Please select a binary outcome column from the Override dropdown, or use a different dataset."
        )

    # ── Step 4: Train ──────────────────────────────────────────────────────
    _stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=_stratify
    )

    result = select_and_train(X_train, y_train, X_test, y_test)

    return result, best_target, confidence, candidates
