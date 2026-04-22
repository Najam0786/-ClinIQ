"""
core/file_joiner.py
--------------------
Auto-join multiple uploaded DataFrames on a common key column.

Strategy
--------
1. Find columns shared across ALL uploaded files (candidate join keys).
2. Prefer low-cardinality, ID-like columns (patient_id, id, subject_id, etc.).
3. If a candidate has close-to-unique values across files → use it as the join key.
4. Fall back to a simple pd.concat if no join key can be identified.

Returns
-------
JoinResult dataclass with:
  - merged:    the combined DataFrame
  - join_key:  column used to join (None if concat was used)
  - strategy:  "merge" | "concat"
  - files:     list of file names that were joined
  - row_counts: rows per file before merging
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

_ID_KEYWORDS = re.compile(
    r"(patient|subject|person|case|record|encounter|id|uid|uuid|key|mrn|nhs|ref)",
    re.IGNORECASE,
)


@dataclass
class JoinResult:
    merged:     pd.DataFrame
    join_key:   Optional[str]
    strategy:   str
    files:      List[str]
    row_counts: Dict[str, int]
    columns_added: List[str] = field(default_factory=list)


def _score_key_column(col: str, series: pd.Series) -> float:
    """Score how likely a column is to be a good join key (higher = better)."""
    score = 0.0
    if _ID_KEYWORDS.search(col):
        score += 3.0
    n_unique  = series.nunique()
    n_total   = len(series)
    if n_total == 0:
        return 0.0
    uniqueness = n_unique / n_total
    if uniqueness > 0.8:
        score += 2.0
    elif uniqueness > 0.5:
        score += 1.0
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_string_dtype(series):
        score += 0.5
    return score


def _find_join_key(
    frames: List[Tuple[str, pd.DataFrame]],
) -> Optional[str]:
    """Return the best join key column name, or None if none found."""
    if len(frames) < 2:
        return None

    common_cols = set(frames[0][1].columns)
    for _, df in frames[1:]:
        common_cols &= set(df.columns)

    if not common_cols:
        return None

    candidates: Dict[str, float] = {}
    for col in common_cols:
        scores = [_score_key_column(col, df[col]) for _, df in frames]
        avg_score = sum(scores) / len(scores)
        if avg_score > 0:
            candidates[col] = avg_score

    if not candidates:
        return None

    best = max(candidates, key=candidates.__getitem__)
    if candidates[best] < 1.0:
        return None
    return best


def join_files(
    named_frames: List[Tuple[str, pd.DataFrame]],
    explicit_key: Optional[str] = None,
) -> JoinResult:
    """
    Merge a list of (filename, DataFrame) pairs into one DataFrame.

    Parameters
    ----------
    named_frames : list of (name, DataFrame)
    explicit_key : if provided, forces this column as the join key

    Returns
    -------
    JoinResult
    """
    if not named_frames:
        raise ValueError("No files provided.")

    names      = [n for n, _ in named_frames]
    row_counts = {n: len(df) for n, df in named_frames}

    if len(named_frames) == 1:
        return JoinResult(
            merged=named_frames[0][1].copy(),
            join_key=None,
            strategy="single",
            files=names,
            row_counts=row_counts,
        )

    join_key = explicit_key or _find_join_key(named_frames)

    if join_key:
        merged = named_frames[0][1].copy()
        cols_before = set(merged.columns)
        for name, df in named_frames[1:]:
            suffix = f"_{name.split('.')[0]}"
            merged = merged.merge(
                df,
                on=join_key,
                how="outer",
                suffixes=("", suffix),
            )
        cols_added = [c for c in merged.columns if c not in cols_before]
        return JoinResult(
            merged=merged,
            join_key=join_key,
            strategy="merge",
            files=names,
            row_counts=row_counts,
            columns_added=cols_added,
        )

    merged = pd.concat([df for _, df in named_frames], ignore_index=True)
    return JoinResult(
        merged=merged,
        join_key=None,
        strategy="concat",
        files=names,
        row_counts=row_counts,
    )
