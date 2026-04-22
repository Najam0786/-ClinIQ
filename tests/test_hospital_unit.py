"""
tests/test_hospital_unit.py
Pure unit tests for hospital stats helper logic — no server required.
Run:  pytest tests/test_hospital_unit.py -v
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from api.routes.hospital import _hospital_stats, HospitalStats


# ── Helpers to build lightweight mock ORM objects ────────────────────────────

def _user(role="doctor", is_active=True):
    u = MagicMock()
    u.role.value = role
    u.is_active = is_active
    u.created_at = datetime.now(timezone.utc)
    return u


def _pred(disease="diabetes", risk_label="High Risk", hospital="City"):
    p = MagicMock()
    p.disease = disease
    p.risk_label = risk_label
    p.hospital = hospital
    p.created_at = datetime.now(timezone.utc)
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_stats_counts_users_and_predictions():
    db = MagicMock()
    users = [_user("doctor"), _user("admin"), _user("viewer", is_active=False)]
    preds = [_pred("diabetes", "High Risk"), _pred("heart_disease", "Low Risk")]

    db.query.return_value.filter.return_value.all.side_effect = [users, preds]

    result = _hospital_stats("City", db)

    assert result.hospital == "City"
    assert result.total_users == 3
    assert result.active_users == 2
    assert result.total_predictions == 2


def test_stats_high_low_risk_split():
    db = MagicMock()
    users = [_user()]
    preds = [
        _pred("diabetes", "High Risk"),
        _pred("diabetes", "High Risk"),
        _pred("stroke",   "Low Risk"),
    ]
    db.query.return_value.filter.return_value.all.side_effect = [users, preds]

    result = _hospital_stats("General", db)

    assert result.high_risk_count == 2
    assert result.low_risk_count == 1


def test_stats_diseases_sorted():
    db = MagicMock()
    users = [_user()]
    preds = [
        _pred("stroke",       "Low Risk"),
        _pred("breast_cancer","High Risk"),
        _pred("diabetes",     "Low Risk"),
        _pred("diabetes",     "High Risk"),
    ]
    db.query.return_value.filter.return_value.all.side_effect = [users, preds]

    result = _hospital_stats("Metro", db)

    assert result.diseases_covered == sorted(["stroke", "breast_cancer", "diabetes"])


def test_stats_role_breakdown():
    db = MagicMock()
    users = [_user("doctor"), _user("doctor"), _user("admin")]
    preds = []
    db.query.return_value.filter.return_value.all.side_effect = [users, preds]

    result = _hospital_stats("RoyalInf", db)

    assert result.roles == {"doctor": 2, "admin": 1}


def test_stats_raises_404_when_no_users():
    from fastapi import HTTPException
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []

    with pytest.raises(HTTPException) as exc:
        _hospital_stats("UnknownHosp", db)

    assert exc.value.status_code == 404


def test_stats_empty_predictions_ok():
    db = MagicMock()
    users = [_user()]
    preds = []
    db.query.return_value.filter.return_value.all.side_effect = [users, preds]

    result = _hospital_stats("SmallClinic", db)

    assert result.total_predictions == 0
    assert result.high_risk_count == 0
    assert result.diseases_covered == []
