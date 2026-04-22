"""
tests/test_cleaner_profiler_unit.py
Pure unit tests for core/profiler.py and core/cleaner.py — no server required.
Run:  pytest tests/test_cleaner_profiler_unit.py -v
"""

import numpy as np
import pandas as pd
import pytest

from core.profiler import profile, DataProfile, ColumnProfile, _detect_dtype
from core.cleaner import clean, cleaning_summary


# ═══════════════════════════════════════════════════════════════════
# PROFILER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestProfilerBasics:

    def test_returns_data_profile(self):
        df = pd.DataFrame({"age": [30, 40, 50], "outcome": [0, 1, 0]})
        result = profile(df)
        assert isinstance(result, DataProfile)

    def test_row_col_counts(self):
        df = pd.DataFrame({"a": range(10), "b": range(10), "c": range(10)})
        p = profile(df)
        assert p.n_rows == 10
        assert p.n_cols == 3

    def test_duplicate_detection(self):
        df = pd.DataFrame({"age": [30, 30, 40], "bmi": [25, 25, 28]})
        p = profile(df)
        assert p.n_duplicates == 1
        assert p.duplicate_pct > 0

    def test_no_duplicates(self):
        df = pd.DataFrame({"age": [30, 40, 50]})
        p = profile(df)
        assert p.n_duplicates == 0

    def test_column_profiles_created(self):
        df = pd.DataFrame({"age": [30, 40], "gender": ["M", "F"]})
        p = profile(df)
        assert "age"    in p.columns
        assert "gender" in p.columns

    def test_quality_score_clean_data(self):
        df = pd.DataFrame({
            "age":  [30, 40, 50, 60, 70],
            "bmi":  [25, 27, 28, 30, 32],
            "out":  [0, 1, 0, 1, 0],
        })
        p = profile(df)
        assert p.quality_score >= 80
        assert p.quality_label in ("Excellent", "Good")

    def test_quality_penalised_by_missing(self):
        df = pd.DataFrame({"age": [30, None, None, None, None]})
        p = profile(df)
        assert p.quality_score < 80

    def test_warnings_for_high_missing(self):
        df = pd.DataFrame({"age": [None] * 70 + [30] * 30})
        p = profile(df)
        assert any("age" in w for w in p.warnings)

    def test_invalid_negative_detected(self):
        df = pd.DataFrame({"age": [30, -5, 40]})
        p = profile(df)
        assert p.columns["age"].invalid_count == 1
        assert any("invalid" in w.lower() or "negative" in w.lower() for w in p.warnings)

    def test_numeric_stats_computed(self):
        df = pd.DataFrame({"age": [20.0, 40.0, 60.0]})
        p = profile(df)
        col = p.columns["age"]
        assert col.stats.get("mean") == pytest.approx(40.0)
        assert col.stats.get("min")  == pytest.approx(20.0)
        assert col.stats.get("max")  == pytest.approx(60.0)

    def test_categorical_top_values(self):
        df = pd.DataFrame({"gender": ["M", "M", "F", "M", "F"]})
        p = profile(df)
        col = p.columns["gender"]
        assert "top_values" in col.stats
        assert "M" in col.stats["top_values"]


class TestDetectDtype:

    def test_numeric_column(self):
        s = pd.Series([1.0, 2.0, 3.0], name="bmi")
        assert _detect_dtype(s) == "numeric"

    def test_boolean_column(self):
        s = pd.Series([0, 1, 0, 1], name="flag")
        assert _detect_dtype(s) == "boolean"

    def test_categorical_string(self):
        s = pd.Series(["M", "F", "M", "F"] * 10, name="gender")
        assert _detect_dtype(s) == "categorical"

    def test_id_column_skipped(self):
        s = pd.Series([f"P{i:04d}" for i in range(100)], name="patient_id")
        assert _detect_dtype(s) == "id"


# ═══════════════════════════════════════════════════════════════════
# CLEANER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestCleaner:

    def _df_with_profile(self, df):
        p = profile(df)
        cleaned, log = clean(df, p)
        return cleaned, log

    def test_clean_returns_dataframe_and_log(self):
        df = pd.DataFrame({"age": [30.0, 40.0, 50.0]})
        cleaned, log = self._df_with_profile(df)
        assert isinstance(cleaned, pd.DataFrame)
        assert isinstance(log, list)

    def test_removes_duplicate_rows(self):
        df = pd.DataFrame({"age": [30, 30, 40], "bmi": [25, 25, 28]})
        cleaned, log = self._df_with_profile(df)
        assert len(cleaned) == 2
        assert any("Duplicate" in e["step"] for e in log)

    def test_no_duplicates_no_log_entry(self):
        df = pd.DataFrame({"age": [30, 40, 50]})
        _, log = self._df_with_profile(df)
        assert not any("Duplicate" in e["step"] for e in log)

    def test_negative_medical_values_replaced_with_nan(self):
        df = pd.DataFrame({"age": [30.0, -5.0, 40.0]})
        p = profile(df)
        cleaned, log = clean(df, p)
        # cleaner: negative → NaN, then median-imputed → final value is NOT -5
        assert cleaned["age"].iloc[1] != -5.0
        assert cleaned["age"].isna().sum() == 0
        assert any("Invalid" in e["step"] for e in log)

    def test_non_medical_negatives_kept(self):
        df = pd.DataFrame({"score": [-1.0, -2.0, 3.0]})
        p = profile(df)
        cleaned, log = clean(df, p)
        assert cleaned["score"].tolist() == [-1.0, -2.0, 3.0]

    def test_numeric_missing_imputed_with_median(self):
        df = pd.DataFrame({"age": [30.0, None, 50.0]})
        p = profile(df)
        cleaned, log = clean(df, p)
        assert cleaned["age"].isna().sum() == 0
        assert any("Median" in e["step"] for e in log)

    def test_categorical_missing_imputed_with_mode(self):
        df = pd.DataFrame({"gender": ["M", "M", None, "F", "M"]})
        p = profile(df)
        cleaned, log = clean(df, p)
        # drop_duplicates re-indexes; just verify no nulls remain and mode step ran
        assert cleaned["gender"].isna().sum() == 0
        assert any("Mode" in e["step"] for e in log)

    def test_already_clean_data_gets_no_action_entry(self):
        df = pd.DataFrame({"age": [30.0, 40.0], "bmi": [25.0, 27.0]})
        _, log = self._df_with_profile(df)
        assert any("No Action" in e["step"] for e in log)

    def test_cleaning_summary_totals(self):
        df = pd.DataFrame({"age": [30, 30, 40], "bmi": [25, 25, 28]})
        p = profile(df)
        _, log = clean(df, p)
        summary = cleaning_summary(log)
        assert "total_steps" in summary
        assert "total_cells_fixed" in summary
        assert summary["total_steps"] == len(log)

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"age": [30.0, None, 50.0]})
        original_len = len(df)
        original_nulls = df["age"].isna().sum()
        p = profile(df)
        clean(df, p)
        assert len(df) == original_len
        assert df["age"].isna().sum() == original_nulls
