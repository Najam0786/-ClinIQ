"""
core/profiler.py
----------------
Auto-detects data schema, column types, quality issues, and computes
an overall data quality score (0-100) for any uploaded dataset.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


# Columns where negative values are medically impossible
MEDICAL_NON_NEGATIVE = {
    "age", "bmi", "weight_kg", "height_cm",
    "systolic_bp", "diastolic_bp",
    "avg_glucose", "avg_cholesterol", "avg_hemoglobin",
    "avg_creatinine", "avg_wbc", "avg_platelets", "avg_hba1c",
    "hba1c", "egfr", "fev1_pct", "ejection_fraction",
    "length_of_stay_days", "n_comorbidities", "n_procedures",
    "n_meds", "n_meds_at_discharge", "n_visits", "n_hospitalizations",
    "hospitalizations_6m", "er_visits_6m", "prev_admissions_12m",
    "med_adherence_score",
}


@dataclass
class ColumnProfile:
    name: str
    dtype_detected: str          # numeric | categorical | datetime | id | boolean
    missing_count: int
    missing_pct: float
    invalid_count: int           # values that are medically impossible
    n_unique: int
    sample_values: list
    stats: dict = field(default_factory=dict)


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    n_duplicates: int
    duplicate_pct: float
    columns: Dict[str, ColumnProfile]
    quality_score: float         # 0 – 100
    quality_label: str           # Excellent / Good / Fair / Poor
    warnings: List[str]


def _detect_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 2 and set(series.dropna().unique()).issubset({0, 1}):
            return "boolean"
        if series.nunique() <= 15 and series.nunique() / max(len(series), 1) < 0.05:
            return "categorical"
        name_lower = series.name.lower()
        if any(k in name_lower for k in ("id", "code", "key", "num")):
            if series.nunique() / max(len(series), 1) > 0.8:
                return "id"
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    try:
        pd.to_datetime(series.dropna().head(20))
        return "datetime"
    except Exception:
        pass
    n_unique = series.nunique()
    if n_unique / max(len(series), 1) > 0.9:
        return "id"
    return "categorical"


def _count_invalid(series: pd.Series, col_name: str) -> int:
    if col_name.lower() not in MEDICAL_NON_NEGATIVE:
        return 0
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    return int((series.dropna() < 0).sum())


def profile(df: pd.DataFrame) -> DataProfile:
    n_rows = len(df)
    n_duplicates = int(df.duplicated().sum())
    duplicate_pct = round(n_duplicates / max(n_rows, 1) * 100, 2)
    warnings: List[str] = []

    columns: Dict[str, ColumnProfile] = {}
    total_missing = 0
    total_invalid = 0
    total_cells = n_rows * len(df.columns)

    for col in df.columns:
        s = df[col]
        dtype = _detect_dtype(s)
        missing_count = int(s.isna().sum())
        missing_pct   = round(missing_count / max(n_rows, 1) * 100, 2)
        invalid_count = _count_invalid(s, col)
        n_unique      = int(s.nunique(dropna=True))

        sample = s.dropna().head(5).tolist()

        stats = {}
        if dtype == "numeric":
            stats = {
                "mean":   round(float(s.mean()), 3) if not s.isna().all() else None,
                "median": round(float(s.median()), 3) if not s.isna().all() else None,
                "std":    round(float(s.std()), 3) if not s.isna().all() else None,
                "min":    round(float(s.min()), 3) if not s.isna().all() else None,
                "max":    round(float(s.max()), 3) if not s.isna().all() else None,
            }
        elif dtype == "categorical":
            vc = s.value_counts(dropna=True)
            stats = {"top_values": vc.head(5).to_dict()}

        total_missing += missing_count
        total_invalid += invalid_count

        columns[col] = ColumnProfile(
            name=col,
            dtype_detected=dtype,
            missing_count=missing_count,
            missing_pct=missing_pct,
            invalid_count=invalid_count,
            n_unique=n_unique,
            sample_values=sample,
            stats=stats,
        )

        if missing_pct > 30:
            warnings.append(f"'{col}' has {missing_pct:.1f}% missing values.")
        if invalid_count > 0:
            warnings.append(f"'{col}' has {invalid_count} invalid (negative) values.")

    if duplicate_pct > 5:
        warnings.append(f"{n_duplicates} duplicate rows detected ({duplicate_pct:.1f}%).")

    # Quality score: start at 100, deduct for issues
    missing_penalty  = min(40, (total_missing / max(total_cells, 1)) * 200)
    invalid_penalty  = min(20, (total_invalid / max(total_cells, 1)) * 300)
    duplicate_penalty = min(20, duplicate_pct * 2)
    quality_score = max(0.0, 100 - missing_penalty - invalid_penalty - duplicate_penalty)
    quality_score = round(quality_score, 1)

    if quality_score >= 85:
        quality_label = "Excellent"
    elif quality_score >= 70:
        quality_label = "Good"
    elif quality_score >= 50:
        quality_label = "Fair"
    else:
        quality_label = "Poor"

    return DataProfile(
        n_rows=n_rows,
        n_cols=len(df.columns),
        n_duplicates=n_duplicates,
        duplicate_pct=duplicate_pct,
        columns=columns,
        quality_score=quality_score,
        quality_label=quality_label,
        warnings=warnings,
    )
