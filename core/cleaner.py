"""
core/cleaner.py
---------------
Auto-cleans any incoming DataFrame:
  1. Remove duplicate rows
  2. Replace medically impossible values (negatives) with NaN
  3. Impute missing values (median for numeric, mode for categorical)
Logs every action taken so the dashboard can show a transparent clean report.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple
from core.profiler import DataProfile, MEDICAL_NON_NEGATIVE


CleanLog = List[dict]   # list of {"step", "column", "detail", "rows_affected"}


def clean(df: pd.DataFrame, profile: DataProfile) -> Tuple[pd.DataFrame, CleanLog]:
    df = df.copy()
    log: CleanLog = []

    # ── Step 1: Remove duplicates ──────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = n_before - len(df)
    if removed > 0:
        log.append({
            "step": "Duplicate Removal",
            "column": "ALL",
            "detail": f"Removed {removed} exact duplicate rows.",
            "rows_affected": removed,
        })

    # ── Step 2: Replace invalid (medically impossible) values with NaN ─────
    for col in df.columns:
        if col.lower() not in MEDICAL_NON_NEGATIVE:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        invalid_mask = df[col] < 0
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            df.loc[invalid_mask, col] = np.nan
            log.append({
                "step": "Invalid Value Fix",
                "column": col,
                "detail": f"Replaced {n_invalid} negative (impossible) values with NaN.",
                "rows_affected": n_invalid,
            })

    # ── Step 3: Impute missing values ──────────────────────────────────────
    for col in df.columns:
        col_profile = profile.columns.get(col)
        if col_profile is None:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        dtype = col_profile.dtype_detected

        if dtype in ("numeric", "boolean"):
            fill_value = df[col].median()
            if pd.isna(fill_value):
                fill_value = 0
            df[col] = df[col].fillna(fill_value)
            log.append({
                "step": "Median Imputation",
                "column": col,
                "detail": f"Filled {n_missing} missing values with median ({round(float(fill_value), 3)}).",
                "rows_affected": n_missing,
            })

        elif dtype == "categorical":
            mode_vals = df[col].mode()
            if len(mode_vals) == 0:
                continue
            fill_value = mode_vals[0]
            df[col] = df[col].fillna(fill_value)
            log.append({
                "step": "Mode Imputation",
                "column": col,
                "detail": f"Filled {n_missing} missing values with mode ('{fill_value}').",
                "rows_affected": n_missing,
            })

    total_changes = sum(entry["rows_affected"] for entry in log)
    if total_changes == 0:
        log.append({
            "step": "No Action Needed",
            "column": "ALL",
            "detail": "Dataset was already clean. No changes made.",
            "rows_affected": 0,
        })

    return df, log


def cleaning_summary(log: CleanLog) -> dict:
    return {
        "total_steps": len(log),
        "total_cells_fixed": sum(e["rows_affected"] for e in log),
        "steps": log,
    }
