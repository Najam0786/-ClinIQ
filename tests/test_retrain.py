"""
tests/test_retrain.py
Run:  CLINIQ_API_URL=http://localhost:8002 python tests/test_retrain.py
"""
import io, json, os, time, urllib.request, urllib.parse
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


def _csv_mp(filename, df):
    buf = io.BytesIO(); df.to_csv(buf, index=False); csv = buf.getvalue()
    boundary = b"ClinIQBoundary"
    body = (b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\""
            + filename.encode() + b"\"\r\nContent-Type: text/csv\r\n\r\n" + csv
            + b"\r\n--" + boundary + b"--\r\n")
    return body, ("multipart/form-data; boundary=" + boundary.decode())


# ── Login ─────────────────────────────────────────────────────────────────────
s, b = api(BASE + "/auth/login", "POST", {"username": "dr.nazmul@cliniq.com", "password": "ClinIQ@2024"}, form=True)
assert s == 200, f"Login failed: {b}"
AUTH = {"Authorization": f"Bearer {b['access_token']}"}
print(f"LOGIN [200]")

# ── Save a baseline so retrain has data to work with ─────────────────────────
np.random.seed(7)
baseline = pd.DataFrame({
    "age":    np.random.normal(55,10,200).round(1),
    "bmi":    np.random.normal(28,5,200).round(2),
    "bp":     np.random.normal(120,15,200).round(1),
    "gender": np.random.choice(["M","F"],200),
})
body, ct = _csv_mp("baseline.csv", baseline)
s, b = api(BASE + "/drift/baseline/diabetes", "POST", headers=AUTH, raw_body=body, ct=ct)
assert s == 201, f"Baseline failed [{s}]: {b}"
print(f"BASELINE [201]: {b['rows']} rows")

# ── Trigger manual retrain ────────────────────────────────────────────────────
s, b = api(BASE + "/retrain/diabetes", "POST", headers=AUTH)
assert s == 202, f"Retrain trigger failed [{s}]: {b}"
print(f"RETRAIN QUEUED [202]: {b['message'][:60]}...")

# ── Poll status (up to 60s) ───────────────────────────────────────────────────
for i in range(24):
    time.sleep(2.5)
    s, b = api(BASE + "/retrain/diabetes/status", headers=AUTH)
    assert s == 200, f"Status failed [{s}]: {b}"
    status = b.get("status")
    print(f"  poll {i+1:02d}: {status}")
    if status in ("completed", "failed"):
        break

assert status == "completed", f"Retrain did not complete: {b.get('error','')[:200]}"
print(f"RETRAIN COMPLETED: model_id={b.get('model_id')} auc={b.get('test_auc')}")

# ── All jobs list ─────────────────────────────────────────────────────────────
s, b = api(BASE + "/retrain/jobs", headers=AUTH)
assert s == 200 and any(j["disease"] == "diabetes" for j in b)
print(f"JOBS [200]: {[j['disease'] for j in b]}")

print("\nALL RETRAIN TESTS PASSED")
