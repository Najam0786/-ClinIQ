"""
tests/test_auth_unit.py
Pure unit tests for auth utility functions — no server required.
Run:  pytest tests/test_auth_unit.py -v
"""

import pytest
from api.auth_utils import hash_password, verify_password, create_access_token


# ── Password hashing ──────────────────────────────────────────────────────────

def test_hash_is_not_plaintext():
    h = hash_password("secret123")
    assert h != "secret123"
    assert len(h) > 20


def test_verify_correct_password():
    h = hash_password("ClinIQ@2024")
    assert verify_password("ClinIQ@2024", h) is True


def test_verify_wrong_password():
    h = hash_password("ClinIQ@2024")
    assert verify_password("WrongPass!", h) is False


def test_two_hashes_differ():
    h1 = hash_password("same_password")
    h2 = hash_password("same_password")
    assert h1 != h2


# ── JWT token creation ────────────────────────────────────────────────────────

def test_token_is_string():
    token = create_access_token(subject="user-uuid-123", role="doctor")
    assert isinstance(token, str)
    assert len(token) > 40


def test_token_has_three_parts():
    token = create_access_token(subject="user-uuid-123", role="admin")
    parts = token.split(".")
    assert len(parts) == 3


def test_different_subjects_produce_different_tokens():
    t1 = create_access_token(subject="user-1", role="doctor")
    t2 = create_access_token(subject="user-2", role="doctor")
    assert t1 != t2


def test_extra_claims_accepted():
    token = create_access_token(
        subject="user-uuid-123",
        role="doctor",
        extra={"email": "doc@cliniq.com", "hospital": "City Hospital"},
    )
    assert isinstance(token, str)
