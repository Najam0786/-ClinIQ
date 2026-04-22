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
        raw = resp.read()
        body = json.loads(raw) if raw else {}
        return resp.status, body
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

# ── 6. Password change ────────────────────────────────────────────────────────
AUTH = {"Authorization": f"Bearer {token}"}
s, b = api(BASE + "/auth/me/password", "POST",
           {"current_password": "wrongpass", "new_password": "newpass999"},
           headers=AUTH)
print(f"PW WRONG  [{s}]: {b.get('detail')}")
assert s == 400, f"Expected 400 for wrong current password, got {s}"

s, b = api(BASE + "/auth/me/password", "POST",
           {"current_password": "ClinIQ@2024", "new_password": "ClinIQ@2024"},
           headers=AUTH)
print(f"PW CHANGE [{s}]: OK (204)")
assert s == 204, f"Expected 204 for password change, got {s}"

# ── 7. Admin register + deactivate/reactivate ─────────────────────────────────
s, b = api(
    BASE + "/auth/register", "POST",
    {"email": "admin@cliniq.com", "full_name": "ClinIQ Admin",
     "password": "Admin@2024!", "role": "admin"},
)
print(f"ADMIN REG [{s}]: {b.get('email', b.get('detail'))}")
assert s in (201, 409)

s, b = api(BASE + "/auth/login", "POST",
           {"username": "admin@cliniq.com", "password": "Admin@2024!"}, form=True)
print(f"ADMIN LOGIN [{s}]: {str(b.get('access_token',''))[:30]}...")
assert s == 200, f"Admin login failed: {b}"
admin_token = b["access_token"]
ADMIN = {"Authorization": f"Bearer {admin_token}"}

# get the doctor user id
s, b = api(BASE + "/auth/users", headers=ADMIN)
assert s == 200
doctor = next((u for u in b if u["email"] == "dr.nazmul@cliniq.com"), None)
assert doctor, "Doctor user not found in /auth/users"
doctor_id = doctor["id"]

# deactivate
s, b = api(BASE + f"/auth/users/{doctor_id}/active", "PUT",
           {"is_active": False}, headers=ADMIN)
print(f"DEACTIVATE [{s}]: active={b.get('is_active')}")
assert s == 200 and b.get("is_active") is False

# reactivate
s, b = api(BASE + f"/auth/users/{doctor_id}/active", "PUT",
           {"is_active": True}, headers=ADMIN)
print(f"REACTIVATE [{s}]: active={b.get('is_active')}")
assert s == 200 and b.get("is_active") is True

print("\nALL AUTH TESTS PASSED")
