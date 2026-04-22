"""
tests/test_drift.py
--------------------
Integration test for drift detection endpoints.
Run with a live server:  CLINIQ_API_URL=http://localhost:8001 python tests/test_drift.py
"""

import io
import json
import os
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd

BASE = os.environ.get("CLINIQ_API_URL", "http://localhost:8000") + "/api/v1"


def api(url, method="GET", data=None, headers=None, form=False, raw_body=None, content_type=None):
    h = headers or {}
    if raw_body is not None:
        payload = raw_body
        h["Content-Type"] = content_type or "application/octet-stream"
    elif data and not form:
        payload = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    elif data and form:
        payload = urllib.parse.urlencode(data).encode()
        h["Content-Type"] = "application/x-www-form-urlencoded"
    else:
        payload = None
    r = urllib.request.Request(url, data=payload, headers=h, method=method)
    try:
        resp = urllib.request.urlopen(r)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return e.code, body


def _csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _multipart(filename, csv_bytes):
    boundary = b"ClinIQBoundary1234"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: text/csv\r\n\r\n" +
        csv_bytes +
        b"\r\n--" + boundary + b"--\r\n"
    )
    return body, b"multipart/form-data; boundary=" + boundary


# ── Login first ───────────────────────────────────────────────────────────────
s, b = api(
    BASE + "/auth/login", "POST",
    {"username": "dr.nazmul@cliniq.com", "password": "ClinIQ@2024"},
    form=True,
)
assert s == 200, f"Login failed: {b}"
TOKEN = b["access_token"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}
print(f"LOGIN      [200]: token={TOKEN[:35]}...")

# ── Upload baseline ───────────────────────────────────────────────────────────
np.random.seed(42)
ref_df = pd.DataFrame({
    "age":    np.random.normal(55, 10, 300).round(1),
    "bmi":    np.random.normal(28, 5, 300).round(2),
    "gender": np.random.choice(["M", "F"], 300),
    "bp":     np.random.normal(120, 15, 300).round(1),
})
csv_bytes = _csv_bytes(ref_df)
body, ct = _multipart("baseline.csv", csv_bytes)
s, b = api(
    BASE + "/drift/baseline/heart_disease", "POST",
    headers=AUTH,
    raw_body=body,
    content_type=ct.decode(),
)
assert s == 201, f"Baseline upload failed [{s}]: {b}"
print(f"BASELINE   [{s}]: rows={b['rows']} cols={b['columns']}")

# ── Drift status ──────────────────────────────────────────────────────────────
s, b = api(BASE + "/drift/status", headers=AUTH)
assert s == 200 and any(d["disease"] == "heart_disease" for d in b), f"Status failed: {b}"
print(f"STATUS     [{s}]: {[d['disease'] for d in b]}")

# ── Detect drift (drifted data) ───────────────────────────────────────────────
cur_df = pd.DataFrame({
    "age":    np.random.normal(70, 15, 100).round(1),
    "bmi":    np.random.normal(35, 8, 100).round(2),
    "gender": np.random.choice(["M", "F"], 100),
    "bp":     np.random.normal(145, 20, 100).round(1),
})
body2, ct2 = _multipart("current.csv", _csv_bytes(cur_df))
s, b = api(
    BASE + "/drift/detect/heart_disease", "POST",
    headers=AUTH,
    raw_body=body2,
    content_type=ct2.decode(),
)
assert s == 200, f"Drift detect failed [{s}]: {b}"
print(f"DETECT     [{s}]: drift_ratio={b['drift_ratio']} alert={b['alert']} drifted={b['drifted_features']}")
assert b["alert"], "Expected alert=True for clearly drifted data"

# ── Get stored report ─────────────────────────────────────────────────────────
s, b = api(BASE + "/drift/report/heart_disease", headers=AUTH)
assert s == 200 and b["disease"] == "heart_disease"
print(f"REPORT     [{s}]: {b['summary'][:70]}...")

print("\nALL DRIFT TESTS PASSED")
