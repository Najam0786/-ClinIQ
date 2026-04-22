"""
tests/test_multifile.py
Run:  CLINIQ_API_URL=http://localhost:8002 python tests/test_multifile.py
"""
import io, json, os, urllib.request, urllib.parse
import numpy as np
import pandas as pd

BASE = os.environ.get("CLINIQ_API_URL", "http://localhost:8000") + "/api/v1"


def api(url, method="GET", data=None, headers=None, form=False, raw_body=None, ct=None):
    h = headers or {}
    if raw_body is not None:
        payload = raw_body
        h["Content-Type"] = ct or "application/octet-stream"
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
        try: body = json.loads(e.read())
        except: body = {}
        return e.code, body


def _multipart_multi(files_dict, form_fields):
    """Build multipart body with multiple file fields + form fields."""
    boundary = b"ClinIQMultiBoundary"
    parts = []
    for fname, csv_bytes in files_dict.items():
        parts.append(
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="files"; filename="' + fname.encode() + b'"\r\n'
            b"Content-Type: text/csv\r\n\r\n" + csv_bytes + b"\r\n"
        )
    for key, val in form_fields.items():
        parts.append(
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="' + key.encode() + b'"\r\n\r\n'
            + str(val).encode() + b"\r\n"
        )
    body = b"".join(parts) + b"--" + boundary + b"--\r\n"
    ct = "multipart/form-data; boundary=" + boundary.decode()
    return body, ct


def _csv(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ── Build two relational CSVs ─────────────────────────────────────────────────
np.random.seed(99)
n = 200
patient_ids = [f"P{i:04d}" for i in range(n)]

demographics = pd.DataFrame({
    "patient_id": patient_ids,
    "age":        np.random.randint(30, 80, n),
    "gender":     np.random.choice(["M", "F"], n),
    "hospital":   np.random.choice(["St.Mary", "General", "Royal"], n),
})
labs = pd.DataFrame({
    "patient_id":   patient_ids,
    "bmi":          np.random.normal(28, 5, n).round(2),
    "glucose":      np.random.normal(110, 25, n).round(1),
    "bp_systolic":  np.random.normal(130, 20, n).round(1),
    "outcome":      np.random.randint(0, 2, n),
})

body, ct = _multipart_multi(
    {"demographics.csv": _csv(demographics), "labs.csv": _csv(labs)},
    {"disease": "universal"},
)

s, b = api(BASE + "/analyse", "POST", raw_body=body, ct=ct)
assert s == 200, f"Multi-file analyse failed [{s}]: {json.dumps(b)[:300]}"
assert b.get("files_joined") == 2,  f"Expected files_joined=2, got {b.get('files_joined')}"
assert b.get("join_key") == "patient_id", f"Expected join_key=patient_id, got {b.get('join_key')}"
assert b.get("join_strategy") == "merge",  f"Expected strategy=merge, got {b.get('join_strategy')}"
print(f"MULTI-FILE [200]: joined={b['files_joined']} key='{b['join_key']}' "
      f"strategy={b['join_strategy']} auc={b['test_auc']:.3f} "
      f"patients={b['total_patients']}")

# ── Single file still works ───────────────────────────────────────────────────
single_df = pd.DataFrame({
    "age": np.random.randint(30,80,150), "bmi": np.random.normal(28,5,150).round(2),
    "bp":  np.random.normal(120,15,150).round(1), "outcome": np.random.randint(0,2,150),
})
body2, ct2 = _multipart_multi({"single.csv": _csv(single_df)}, {"disease": "universal"})
s2, b2 = api(BASE + "/analyse", "POST", raw_body=body2, ct=ct2)
assert s2 == 200, f"Single-file failed [{s2}]: {b2}"
assert b2["files_joined"] == 1
print(f"SINGLE-FILE [200]: strategy={b2['join_strategy']} auc={b2['test_auc']:.3f}")

print("\nALL MULTI-FILE TESTS PASSED")
