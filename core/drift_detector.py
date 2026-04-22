"""
core/drift_detector.py
-----------------------
Data drift detection using Evidently AI (v0.7.x).

Workflow
--------
1. On every /analyse run, the training dataframe is saved as the
   baseline snapshot (core.drift_detector.save_baseline).
2. On /drift/detect, incoming data is compared against the baseline;
   a JSON drift report is returned and stored.
3. /drift/report/{disease} returns the latest stored report.

Storage layout (all relative to project root)
----------------------------------------------
drift_reports/
  {disease}/
    baseline.parquet   <- column-filtered training data
    latest_report.json <- most recent drift result
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DRIFT_DIR = Path(os.environ.get("CLINIQ_DRIFT_DIR", "drift_reports"))


# ── Baseline helpers ──────────────────────────────────────────────────────────

def _disease_dir(disease: str) -> Path:
    d = DRIFT_DIR / disease
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_baseline(disease: str, df: pd.DataFrame, feature_cols: List[str]) -> Path:
    """
    Persist a column-filtered copy of the training data as the drift baseline.
    Called automatically by the /analyse pipeline after training.
    """
    out_dir  = _disease_dir(disease)
    baseline = df[feature_cols].copy()
    path     = out_dir / "baseline.parquet"
    baseline.to_parquet(path, index=False)
    return path


def load_baseline(disease: str) -> Optional[pd.DataFrame]:
    path = _disease_dir(disease) / "baseline.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ── Drift computation ─────────────────────────────────────────────────────────

def _column_drift(reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
    """Compute per-column drift statistics manually (no Evidently dependency at call site)."""
    is_numeric = pd.api.types.is_numeric_dtype(reference)

    result: Dict[str, Any] = {
        "column":     reference.name,
        "type":       "numerical" if is_numeric else "categorical",
        "drift_detected": False,
        "drift_score":    0.0,
    }

    try:
        if is_numeric:
            from scipy.stats import ks_2samp
            stat, p_value = ks_2samp(
                reference.dropna().values,
                current.dropna().values,
            )
            result["test"]         = "Kolmogorov-Smirnov"
            result["statistic"]    = round(float(stat), 4)
            result["p_value"]      = round(float(p_value), 4)
            result["drift_score"]  = round(float(stat), 4)
            result["drift_detected"] = bool(p_value < 0.05)
            result["reference_mean"] = round(float(reference.mean()), 4)
            result["current_mean"]   = round(float(current.mean()), 4)
        else:
            from scipy.stats import chi2_contingency
            ref_counts = reference.value_counts(normalize=True)
            cur_counts = current.value_counts(normalize=True)
            all_cats   = ref_counts.index.union(cur_counts.index)
            ref_freq   = ref_counts.reindex(all_cats, fill_value=0).values
            cur_freq   = cur_counts.reindex(all_cats, fill_value=0).values
            # Chi-square needs counts, not proportions
            n_ref = len(reference.dropna())
            n_cur = len(current.dropna())
            contingency = pd.DataFrame(
                [ref_freq * n_ref, cur_freq * n_cur],
                columns=all_cats,
            )
            chi2, p_value, _, _ = chi2_contingency(contingency.values + 1e-8)
            psi = float(sum(
                (c - r) * (max(c, 1e-10) / max(r, 1e-10))
                for r, c in zip(ref_freq, cur_freq)
            ))
            result["test"]           = "Chi-Square"
            result["statistic"]      = round(float(chi2), 4)
            result["p_value"]        = round(float(p_value), 4)
            result["drift_score"]    = round(abs(psi), 4)
            result["drift_detected"] = bool(p_value < 0.05)
    except Exception as exc:
        result["error"] = str(exc)

    return result


def compute_drift(
    disease:   str,
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compare current_df against the stored (or supplied) baseline.
    Returns a serialisable dict that is also saved to disk.
    """
    if reference_df is None:
        reference_df = load_baseline(disease)
    if reference_df is None:
        raise ValueError(
            f"No baseline found for '{disease}'. "
            "Upload a training CSV via POST /api/v1/analyse first."
        )

    shared_cols = [c for c in reference_df.columns if c in current_df.columns]
    if not shared_cols:
        raise ValueError("Current data shares no columns with the baseline.")

    ref = reference_df[shared_cols]
    cur = current_df[shared_cols]

    column_results = [_column_drift(ref[c], cur[c]) for c in shared_cols]
    drifted_cols   = [r for r in column_results if r.get("drift_detected")]
    drift_ratio    = len(drifted_cols) / len(shared_cols) if shared_cols else 0.0

    report: Dict[str, Any] = {
        "disease":           disease,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "reference_rows":    int(len(ref)),
        "current_rows":      int(len(cur)),
        "columns_checked":   len(shared_cols),
        "columns_drifted":   len(drifted_cols),
        "drift_ratio":       round(drift_ratio, 4),
        "dataset_drift":     drift_ratio >= 0.5,
        "alert":             drift_ratio >= 0.3,
        "column_reports":    column_results,
        "drifted_features":  [r["column"] for r in drifted_cols],
        "summary":           _build_summary(disease, drift_ratio, drifted_cols),
    }

    _save_report(disease, report)
    return report


def _build_summary(disease: str, drift_ratio: float, drifted: List[Dict]) -> str:
    pct = round(drift_ratio * 100, 1)
    if not drifted:
        return (
            f"No significant drift detected in {disease.replace('_', ' ')} data. "
            "Input distribution matches the training baseline."
        )
    feat_list = ", ".join(r["column"] for r in drifted[:5])
    severity  = "⚠️ WARNING" if drift_ratio < 0.5 else "🚨 CRITICAL"
    return (
        f"{severity}: {pct}% of features show significant drift in "
        f"{disease.replace('_', ' ')} data. "
        f"Drifted features: {feat_list}. "
        "Consider retraining the model with recent data."
    )


# ── Report storage ────────────────────────────────────────────────────────────

def _save_report(disease: str, report: Dict[str, Any]) -> None:
    path = _disease_dir(disease) / "latest_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def load_latest_report(disease: str) -> Optional[Dict[str, Any]]:
    path = _disease_dir(disease) / "latest_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_baselines() -> List[Dict[str, Any]]:
    """Return a list of diseases that have a stored baseline."""
    if not DRIFT_DIR.exists():
        return []
    result = []
    for d in sorted(DRIFT_DIR.iterdir()):
        if d.is_dir() and (d / "baseline.parquet").exists():
            report = load_latest_report(d.name)
            result.append({
                "disease":          d.name,
                "baseline_exists":  True,
                "last_report_at":   report.get("generated_at") if report else None,
                "last_drift_ratio": report.get("drift_ratio")  if report else None,
                "last_alert":       report.get("alert")        if report else None,
            })
    return result
