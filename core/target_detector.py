"""
core/target_detector.py
------------------------
Automatically infers the most likely binary classification target column
from any unknown dataset.

Strategy (scored heuristics, best score wins):
  1. Cardinality   — binary columns (exactly 2 unique non-null values: 0/1, yes/no, etc.)
  2. Name keywords — column name contains clinical/ML outcome keywords
  3. Position      — last column gets a bonus (common data science convention)
  4. Completeness  — columns with fewer missing values score higher
  5. Value balance — classes that are 10-60 % positive score higher (realistic prevalence)

Also runs a smart column-selector that drops ID columns, near-constant columns,
and columns with excessive missingness — so any messy upload becomes model-ready.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# ── Keywords associated with clinical/ML outcome columns ────────────────────
_TARGET_KEYWORDS = [
    "target", "label", "outcome", "output", "class", "y$",
    "risk", "high_risk", "low_risk",
    "readmit", "readmission", "admit",
    "worsen", "worsened", "progression", "deteriorat",
    "died", "death", "mortality", "deceased", "survive", "survival",
    "positive", "negative", "flag", "event", "incident",
    "sepsis", "complication", "adverse",
    "rehospitali", "icu", "intubat",
    "diagnos", "prognos",
]

# Columns whose names suggest IDs / metadata / geography — never a target
_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^patient|^admission|^encounter|^subject|^record|^index$|^row"
    r"|^city|^state|^country|^region|^zip|^postal|^province|^county"
    r"|^address|^street|^location|^phone|^email|^url"
    r"|^first.?name|^last.?name|^full.?name|^surname|^forename"
    r"|^date|^time|^year|^month|^day|^timestamp|^created|^updated"
    r"|^code$|^icd|^cpt|^npi|^ssn)",
    re.IGNORECASE,
)

# Value sets that are recognised as valid binary encodings
_KNOWN_BINARY_SETS = [
    {"0", "1"},
    {"0.0", "1.0"},
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"positive", "negative"},
    {"present", "absent"},
    {"male", "female"},
    {"m", "f"},
    {"alive", "dead"},
    {"survived", "died"},
    {"normal", "abnormal"},
    {"high", "low"},
    {"good", "poor"},
    {"stable", "unstable"},
    {"home", "not home"},
    {"discharged", "readmitted"},
    {"worsened", "stable"},
    {"improved", "worsened"},
    # Oncology / pathology
    {"m", "b"},
    {"malignant", "benign"},
    {"cancer", "no cancer"},
    {"tumor", "no tumor"},
    # Kidney disease
    {"ckd", "notckd"},
    {"ckd", "not ckd"},
    # Stroke / heart
    {"stroke", "no stroke"},
    {"heart disease", "no heart disease"},
    # Generic outcome labels seen on Kaggle
    {"1", "2"},
    {"0", "2"},
]


def _is_binary_like(series: pd.Series) -> bool:
    """
    True only if the non-null values are exactly 2 distinct values AND they
    come from a recognised binary encoding (numeric 0/1, yes/no, true/false,
    or a known clinical pair).  Arbitrary string pairs (e.g. city names) are
    rejected to avoid false positives.
    """
    vals = series.dropna()
    if len(vals) == 0:
        return False
    uniq = set(str(v).strip().lower() for v in vals.unique())
    if len(uniq) != 2:
        return False
    # Always accept pure numeric 0/1 series
    try:
        num_uniq = {float(v) for v in uniq}
        if num_uniq <= {0.0, 1.0}:
            return True
    except (ValueError, TypeError):
        pass
    # Accept known string binary sets
    for bset in _KNOWN_BINARY_SETS:
        if uniq <= bset:
            return True
    return False


def _binary_balance_score(series: pd.Series) -> float:
    """
    Returns 0-1 score for how 'clinically realistic' the positive-class rate is.
    Columns with 10-60 % positive rate score best.  Near-0 or near-100 score low.
    """
    vals = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(vals) == 0:
        return 0.0
    pos_rate = vals.mean()
    if 0.10 <= pos_rate <= 0.60:
        return 1.0
    elif 0.05 <= pos_rate < 0.10 or 0.60 < pos_rate <= 0.80:
        return 0.5
    return 0.1


def _name_keyword_score(col: str) -> float:
    """Score a column name against outcome-related keywords."""
    col_lower = col.lower()
    score = 0.0
    for kw in _TARGET_KEYWORDS:
        if re.search(kw, col_lower):
            score += 1.0
    return min(score, 3.0)   # cap to avoid one column dominating solely by name


def detect_target(
    df: pd.DataFrame,
) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
    """
    Identify the most likely binary classification target column.

    Returns
    -------
    best_col   : str | None — winning column name (None if no candidate found)
    confidence : float 0-1  — how confident we are in the top pick
    candidates : list[(col, score)] sorted best-first
    """
    scores: dict[str, float] = {}
    last_col = df.columns[-1]

    for col in df.columns:
        # Immediately exclude obvious ID / metadata columns
        if _ID_PATTERNS.search(col):
            continue

        series = df[col]

        # Must be binary-like for primary consideration
        if not _is_binary_like(series):
            # Fallback: only allow low-cardinality NUMERIC columns (2-4 unique values).
            # String columns with arbitrary labels (e.g. D1/D2/D3, city names) are
            # never valid classification targets.
            if not pd.api.types.is_numeric_dtype(series):
                continue
            try:
                n_uniq = series.dropna().nunique()
                if not (2 <= n_uniq <= 4):
                    continue
            except Exception:
                continue
            penalty = 0.4
        else:
            penalty = 1.0

        s = 0.0
        # 1 — name keyword match
        s += _name_keyword_score(col) * 1.5
        # 2 — position bonus
        if col == last_col:
            s += 1.0
        # 3 — completeness (fewer missing = higher score)
        missing_frac = series.isna().mean()
        s += (1.0 - missing_frac) * 0.5
        # 4 — balance score
        s += _binary_balance_score(series) * 1.2
        # Apply binary penalty if not strictly binary
        s *= penalty

        scores[col] = round(s, 4)

    if not scores:
        return None, 0.0, []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_col, best_score = ranked[0]

    # Confidence: how much better is the top pick vs the runner-up?
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best_score - second_score
    # Normalize confidence to 0-1 range
    confidence = min(round((best_score / 6.0) * (1 + gap / (best_score + 1e-9)), 2), 1.0)
    confidence = max(confidence, 0.05)

    return best_col, confidence, ranked


# ── Smart feature selector for any dataset ──────────────────────────────────

def select_features_auto(
    df: pd.DataFrame,
    target_col: str,
    max_missing_frac: float = 0.60,
) -> List[str]:
    """
    Returns a clean list of feature columns for any unknown dataset.

    Drops:
      - The target column itself
      - Obvious ID / metadata columns (high-cardinality strings)
      - Columns with > max_missing_frac missing values
      - Constant or near-constant columns (≤ 1 unique value)
      - Free-text columns (object dtype with > 95 % unique values)
    """
    drop = {target_col}
    n = len(df)

    for col in df.columns:
        if col in drop:
            continue

        # Explicit ID-like name pattern
        if _ID_PATTERNS.search(col):
            drop.add(col)
            continue

        series = df[col]

        # Excessive missingness
        if series.isna().mean() > max_missing_frac:
            drop.add(col)
            continue

        # Constant column
        n_uniq = series.dropna().nunique()
        if n_uniq <= 1:
            drop.add(col)
            continue

        # Free-text / high-cardinality string
        if series.dtype == object and n_uniq > 0.90 * n:
            drop.add(col)
            continue

    return [c for c in df.columns if c not in drop]
