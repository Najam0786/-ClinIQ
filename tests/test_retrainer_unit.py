"""
tests/test_retrainer_unit.py
Pure unit tests for core/retrainer.py state machine — no server required.
Run:  pytest tests/test_retrainer_unit.py -v
"""

import json
from unittest.mock import MagicMock, patch
import pytest

import core.retrainer as retrainer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_retrain_dir(tmp_path, monkeypatch):
    """Redirect all retrainer file I/O to a fresh temp dir per test."""
    monkeypatch.setattr(retrainer, "RETRAIN_STATE_DIR", tmp_path)
    # DRIFT_DIR is imported *by value* into retrainer's namespace, so patch both.
    monkeypatch.setattr(retrainer, "DRIFT_DIR", tmp_path)
    import core.drift_detector as dd
    monkeypatch.setattr(dd, "DRIFT_DIR", tmp_path)
    return tmp_path


# ── get_retrain_status ────────────────────────────────────────────────────────

def test_status_returns_none_when_no_job():
    result = retrainer.get_retrain_status("diabetes")
    assert result is None


def test_status_roundtrip(tmp_path):
    state = {
        "job_id": "abc123",
        "disease": "diabetes",
        "status": "completed",
        "triggered_by": "manual",
        "drift_ratio": 0.6,
        "started_at": "2025-01-01T00:00:00+00:00",
        "finished_at": "2025-01-01T00:01:00+00:00",
        "model_id": "diabetes_20250101",
        "test_auc": 0.85,
        "error": None,
    }
    retrainer._write_state("diabetes", state)
    loaded = retrainer.get_retrain_status("diabetes")
    assert loaded["job_id"] == "abc123"
    assert loaded["status"] == "completed"
    assert loaded["test_auc"] == 0.85


# ── list_retrain_jobs ─────────────────────────────────────────────────────────

def test_list_jobs_empty_initially():
    result = retrainer.list_retrain_jobs()
    assert result == []


def test_list_jobs_after_write():
    for disease in ("diabetes", "stroke"):
        retrainer._write_state(disease, {
            "job_id": "x", "disease": disease,
            "status": "completed", "triggered_by": "manual",
        })
    jobs = retrainer.list_retrain_jobs()
    diseases = [j["disease"] for j in jobs]
    assert "diabetes" in diseases
    assert "stroke" in diseases


# ── _write_state overwrites previous state ────────────────────────────────────

def test_write_state_overwrites():
    retrainer._write_state("heart_disease", {"status": "running", "disease": "heart_disease"})
    retrainer._write_state("heart_disease", {"status": "completed", "disease": "heart_disease"})
    loaded = retrainer.get_retrain_status("heart_disease")
    assert loaded["status"] == "completed"


# ── maybe_trigger_retrain ─────────────────────────────────────────────────────

def test_maybe_trigger_skips_below_threshold():
    bg = MagicMock()
    triggered = retrainer.maybe_trigger_retrain("diabetes", drift_ratio=0.3, background_tasks=bg)
    assert triggered is False
    bg.add_task.assert_not_called()


def test_maybe_trigger_enqueues_at_threshold():
    bg = MagicMock()
    triggered = retrainer.maybe_trigger_retrain("diabetes", drift_ratio=0.5, background_tasks=bg)
    assert triggered is True
    bg.add_task.assert_called_once()
    call_kwargs = bg.add_task.call_args
    assert call_kwargs[0][0] == retrainer.run_retrain_job
    assert call_kwargs[1]["disease"] == "diabetes"
    assert call_kwargs[1]["triggered_by"] == "auto_drift"


def test_maybe_trigger_skips_when_already_running():
    retrainer._write_state("diabetes", {
        "status": "running", "disease": "diabetes",
        "job_id": "xyz", "triggered_by": "manual",
    })
    bg = MagicMock()
    triggered = retrainer.maybe_trigger_retrain("diabetes", drift_ratio=0.9, background_tasks=bg)
    assert triggered is False
    bg.add_task.assert_not_called()


def test_maybe_trigger_fires_after_previous_completed():
    retrainer._write_state("diabetes", {
        "status": "completed", "disease": "diabetes",
        "job_id": "xyz", "triggered_by": "manual",
    })
    bg = MagicMock()
    triggered = retrainer.maybe_trigger_retrain("diabetes", drift_ratio=0.8, background_tasks=bg)
    assert triggered is True
    bg.add_task.assert_called_once()


# ── run_retrain_job error path ────────────────────────────────────────────────

def test_run_retrain_job_writes_failed_state_on_missing_data():
    """No baseline → FileNotFoundError → status=failed written, exception re-raised."""
    with pytest.raises(FileNotFoundError):
        retrainer.run_retrain_job("diabetes", data_path=None, triggered_by="test")

    status = retrainer.get_retrain_status("diabetes")
    assert status is not None
    assert status["status"] == "failed"
    assert "FileNotFoundError" in status["error"]


def test_run_retrain_job_writes_failed_state_on_unknown_disease(tmp_path):
    """Unknown disease module → ValueError → status=failed written."""
    import pandas as pd
    # Write a tiny fake CSV as data_path
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": range(60), "b": range(60), "outcome": [0,1]*30}).to_csv(csv_path, index=False)

    with pytest.raises((ValueError, Exception)):
        retrainer.run_retrain_job("unknown_disease_xyz", data_path=str(csv_path))

    status = retrainer.get_retrain_status("unknown_disease_xyz")
    assert status is not None
    assert status["status"] == "failed"


def test_run_retrain_job_writes_failed_state_on_too_few_rows(tmp_path):
    """< 50 rows → ValueError: Not enough rows."""
    import pandas as pd
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": range(10), "outcome": [0,1]*5}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Not enough rows"):
        retrainer.run_retrain_job("diabetes", data_path=str(csv_path))

    status = retrainer.get_retrain_status("diabetes")
    assert status["status"] == "failed"
