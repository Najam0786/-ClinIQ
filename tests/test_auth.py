import sys, urllib.request, json, urllib.parse

import os
BASE = os.environ.get("CLINIQ_API_URL", "http://localhost:8000") + "/api/v1"


def api(url, method="GET", data=None, headers=None, form=False):
    h = headers or {}
    if data and not form:
        payload = json.dumps(data).encode()
        h = {**h, "Content-Type": "application/json"}
    elif data and form:
        payload = urllib.parse.urlencode(data).encode()
        h = {**h, "Content-Type": "application/x-www-form-urlencoded"}
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


# ── 1. Register ───────────────────────────────────────────────────────────────
s, b = api(
    BASE + "/auth/register", "POST",
    {"email": "dr.nazmul@cliniq.com", "full_name": "Dr. Nazmul Farooquee",
     "password": "ClinIQ@2024", "role": "doctor"},
)
email = b.get("email", b.get("detail", ""))
role  = b.get("role", "-")
print(f"REGISTER  [{s}]: {email} | role={role}")
assert s in (201, 409), f"Unexpected status {s}: {b}"

# ── 2. Login ──────────────────────────────────────────────────────────────────
s, b = api(
    BASE + "/auth/login", "POST",
    {"username": "dr.nazmul@cliniq.com", "password": "ClinIQ@2024"},
    form=True,
)
token = b.get("access_token", "")
print(f"LOGIN     [{s}]: {token[:40]}...")
assert s == 200 and token, f"Login failed: {b}"

# ── 3. GET /me ────────────────────────────────────────────────────────────────
s, b = api(BASE + "/auth/me", headers={"Authorization": f"Bearer {token}"})
print(f"GET /me   [{s}]: {b.get('full_name')} | {b.get('role')} | active={b.get('is_active')}")
assert s == 200

# ── 4. Prediction history (empty) ─────────────────────────────────────────────
s, b = api(BASE + "/audit/me/predictions", headers={"Authorization": f"Bearer {token}"})
print(f"HISTORY   [{s}]: {b}")
assert s == 200 and isinstance(b, list)

# ── 5. Wrong password ─────────────────────────────────────────────────────────
s, b = api(
    BASE + "/auth/login", "POST",
    {"username": "dr.nazmul@cliniq.com", "password": "wrongpass"},
    form=True,
)
print(f"BAD LOGIN [{s}]: {b.get('detail')}")
assert s in (401, 429), f"Expected 401 or 429, got {s}"

print("\nALL AUTH TESTS PASSED")
