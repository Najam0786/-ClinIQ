# 🔄 ClinIQ — Rollback & Recovery Guide

Every push to `main` automatically creates a timestamped release tag (e.g. `v20240501-143200`).
This gives you a **full version history** you can instantly revert to without losing any audit trail.

---

## CI/CD Pipeline Overview

```
Developer pushes code
        │
        ▼
┌─────────────────────────────────┐
│  CI — Quality Gate (ci.yml)     │  ← runs on EVERY push & PR
│  • Syntax check all .py files   │
│  • pytest tests/ (Py 3.10–3.12) │
│  • Blocks merge if tests fail   │
└────────────┬────────────────────┘
             │  (only if CI passes)
             ▼
┌─────────────────────────────────┐
│  CD — Auto-Tag Release (cd.yml) │  ← runs on push to main only
│  • Creates tag v{date}-{time}   │
│  • Creates GitHub Release entry │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Streamlit Community Cloud      │  ← watches main branch
│  • Auto-redeploys in ~2 min     │
│  • No extra config needed       │
└─────────────────────────────────┘
```

---

## How to Roll Back

### ✅ Option 1 — GitHub Actions (Recommended — zero command line)

1. Go to **GitHub → Actions → Rollback — Revert Main to Previous Release**
2. Click **Run workflow** (top-right)
3. Fill in:
   - **Tag**: the version to restore (e.g. `v20240501-143200`) — find tags in **Releases**
   - **Reason**: brief description (logged permanently in git history)
4. Click **Run workflow**
5. Wait ~30 seconds for the revert commit to land on `main`
6. Streamlit Cloud **auto-redeploys** within ~2 minutes ✅

> This creates a **safe revert commit** — no force pushes, full audit trail preserved.

---

### 🖥️ Option 2 — Git CLI (local)

```bash
# 1. See available tags (newest first)
git fetch --tags
git tag --sort=-creatordate | head -20

# 2. Revert main to a specific tag (SAFE — creates a new commit)
git checkout main
git pull origin main
git revert --no-commit <TAG>..HEAD
git commit -m "chore(rollback): revert to <TAG> — <your reason>"
git push origin main
```

Streamlit Cloud picks up the push and redeploys automatically.

---

### 🚨 Option 3 — Emergency Hard Reset (last resort)

Only use if the revert approach fails due to merge conflicts:

```bash
git fetch --tags
git checkout <TAG>
git checkout -b hotfix/emergency-rollback
git push origin hotfix/emergency-rollback
# Then open a PR into main and merge immediately
```

---

## Finding the Right Tag to Roll Back To

| Where | How |
|---|---|
| **GitHub UI** | Repository → **Releases** (right sidebar) |
| **Git CLI** | `git tag --sort=-creatordate \| head -20` |
| **GitHub Actions log** | Each CD run shows the tag it created |

Tags follow the format: **`v{YYYYMMDD}-{HHMMSS}`**
Example: `v20240501-143200` = deployed on May 1st 2024 at 14:32:00 UTC

---

## Streamlit Community Cloud Deployment

| Action | How |
|---|---|
| **Auto-deploy** | Happens automatically on every `main` push (~2 min) |
| **Force redeploy** | Streamlit Cloud dashboard → your app → **Reboot app** |
| **Pin to a commit** | Streamlit Cloud dashboard → **Edit app** → change branch/tag |
| **App URL** | Set during initial Streamlit Cloud setup |

### Initial Streamlit Cloud Setup (one-time)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo: `YOUR-USERNAME/ClinIQ`
4. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy** — done. All future `main` pushes auto-deploy.

---

## Rollback Decision Guide

| Situation | Action |
|---|---|
| Bad code merged to main | **Option 1** — GitHub Actions rollback |
| App crashes after deploy | **Option 1** — pick the last working tag |
| Need to audit what changed | Check **GitHub → Releases** for the commit diff |
| Merge conflicts during revert | **Option 3** — emergency hotfix branch |
| Just need to restart app | Streamlit Cloud → **Reboot app** (no code change) |

---

## Workflow Files Reference

| File | Trigger | Purpose |
|---|---|---|
| `.github/workflows/ci.yml` | Every push + PR | Syntax check + pytest quality gate |
| `.github/workflows/cd.yml` | Push to `main` | Auto-tag + GitHub Release |
| `.github/workflows/rollback.yml` | Manual dispatch | Safe revert to any tag |
