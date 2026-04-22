"""
api/main.py
-----------
ClinIQ REST API — FastAPI entry point.

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, analyse, predict, models_info, fhir, auth, audit, drift


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db.database import init_db
    init_db()
    yield


app = FastAPI(
    title="ClinIQ Clinical Intelligence API",
    description=(
        "Production REST API for the ClinIQ Clinical ML Platform.\n\n"
        "## Authentication\n"
        "Register → Login → copy the `access_token` → click **Authorize** (top right) → paste token.\n\n"
        "## Features\n"
        "- **Auth**: JWT login, role-based access (admin / doctor / viewer)\n"
        "- **Analyse**: Upload a medical dataset → full ML pipeline → trained model\n"
        "- **Predict**: Single-patient real-time risk prediction (logged to DB)\n"
        "- **Models**: Browse and inspect all trained models\n"
        "- **FHIR**: FHIR R4 patient bundle → RiskAssessment (NHS/EMR ready)\n"
        "- **Audit**: Full prediction + action audit trail\n"
        "- **MLflow**: Every training run automatically logged\n\n"
        "## Disease Modules\n"
        "`breast_cancer` | `heart_disease` | `diabetes` | `stroke` | `respiratory` | `universal`\n\n"
        "## Roles\n"
        "| Role | Permissions |\n"
        "|---|---|\n"
        "| `admin` | All endpoints + user management + full audit log |\n"
        "| `doctor` | Analyse, predict, view own history |\n"
        "| `viewer` | Predict only, view own history |"
    ),
    version="2.1.0",
    contact={
        "name":  "Nazmul Farooquee",
        "url":   "https://www.linkedin.com/in/nazmul-farooquee-mba-0b433b1b/",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

app.include_router(health.router,       prefix=API_PREFIX)
app.include_router(auth.router,         prefix=API_PREFIX)
app.include_router(audit.router,        prefix=API_PREFIX)
app.include_router(analyse.router,      prefix=API_PREFIX)
app.include_router(predict.router,      prefix=API_PREFIX)
app.include_router(models_info.router,  prefix=API_PREFIX)
app.include_router(fhir.router,         prefix=API_PREFIX)
app.include_router(drift.router,        prefix=API_PREFIX)


@app.get("/", tags=["System"])
def root():
    return {
        "service": "ClinIQ Clinical Intelligence API",
        "version": "2.1.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "auth":    "/api/v1/auth/login",
    }
