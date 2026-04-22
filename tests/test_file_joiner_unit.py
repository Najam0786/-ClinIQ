"""
tests/test_file_joiner_unit.py
Pure unit tests for core/file_joiner.py — no server required.
Run:  pytest tests/test_file_joiner_unit.py -v
"""

import numpy as np
import pandas as pd
import pytest

from core.file_joiner import join_files, JoinResult


def _df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age":        rng.integers(30, 80, n),
        "bmi":        rng.normal(28, 5, n).round(2),
    })


# ── Single file → strategy "single" ──────────────────────────────────────────

def test_single_file_passthrough():
    df = _df()
    result = join_files([("only.csv", df)])
    assert isinstance(result, JoinResult)
    assert result.strategy == "single"
    assert result.join_key is None
    assert len(result.merged) == len(df)


# ── Two files with shared patient_id → merge ─────────────────────────────────

def test_two_file_merge_on_id():
    n = 100
    df1 = pd.DataFrame({
        "patient_id": range(n),
        "age":        np.random.randint(30, 80, n),
    })
    df2 = pd.DataFrame({
        "patient_id": range(n),
        "glucose":    np.random.normal(110, 20, n).round(1),
        "outcome":    np.random.randint(0, 2, n),
    })
    result = join_files([("demo.csv", df1), ("labs.csv", df2)])
    assert result.strategy == "merge"
    assert result.join_key == "patient_id"
    assert len(result.merged) == n
    assert {"patient_id", "age", "glucose", "outcome"} <= set(result.merged.columns)


# ── Explicit join key overrides auto-detection ────────────────────────────────

def test_explicit_join_key():
    df1 = pd.DataFrame({"record_no": [1, 2, 3], "height": [170, 165, 180]})
    df2 = pd.DataFrame({"record_no": [1, 2, 3], "weight": [70, 60, 85]})
    result = join_files([("a.csv", df1), ("b.csv", df2)], explicit_key="record_no")
    assert result.join_key == "record_no"
    assert result.strategy == "merge"
    assert "height" in result.merged.columns
    assert "weight" in result.merged.columns


# ── No common columns → concat fallback ──────────────────────────────────────

def test_concat_fallback_no_common_key():
    df1 = pd.DataFrame({"age": [55, 60], "bmi": [28, 30], "outcome": [0, 1]})
    df2 = pd.DataFrame({"sbp": [130, 140], "dbp": [85, 90], "outcome": [1, 0]})
    result = join_files([("batch1.csv", df1), ("batch2.csv", df2)])
    assert result.strategy in ("merge", "concat")
    assert len(result.merged) >= 2


# ── Three-way merge ───────────────────────────────────────────────────────────

def test_three_way_merge():
    n = 50
    ids = [f"X{i}" for i in range(n)]
    df1 = pd.DataFrame({"patient_id": ids, "age": range(n)})
    df2 = pd.DataFrame({"patient_id": ids, "bmi": np.random.normal(28, 4, n).round(1)})
    df3 = pd.DataFrame({"patient_id": ids, "outcome": np.random.randint(0, 2, n)})
    result = join_files([("d1.csv", df1), ("d2.csv", df2), ("d3.csv", df3)])
    assert result.strategy == "merge"
    assert result.join_key == "patient_id"
    assert len(result.merged) == n
    assert set(result.files) == {"d1.csv", "d2.csv", "d3.csv"}


# ── Empty input raises ────────────────────────────────────────────────────────

def test_empty_input_raises():
    with pytest.raises(ValueError, match="No files"):
        join_files([])


# ── Row counts are tracked ────────────────────────────────────────────────────

def test_row_counts_tracked():
    df1 = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    df2 = pd.DataFrame({"id": [1, 2, 3], "b": [40, 50, 60]})
    result = join_files([("f1.csv", df1), ("f2.csv", df2)])
    assert result.row_counts == {"f1.csv": 3, "f2.csv": 3}
