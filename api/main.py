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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, analyse, predict, models_info, fhir

app = FastAPI(
    title="ClinIQ Clinical Intelligence API",
    description=(
        "Production REST API for the ClinIQ Clinical ML Platform.\n\n"
        "## Features\n"
        "- **Analyse**: Upload a medical dataset → full ML pipeline → trained model\n"
        "- **Predict**: Single-patient real-time risk prediction\n"
        "- **Models**: Browse and inspect all trained models\n"
        "- **FHIR**: FHIR R4 patient bundle → RiskAssessment (NHS/EMR ready)\n"
        "- **MLflow**: Every training run automatically logged\n\n"
        "## Disease Modules\n"
        "`breast_cancer` | `heart_disease` | `diabetes` | `stroke` | `respiratory` | `universal`"
    ),
    version="2.0.0",
    contact={
        "name":  "Nazmul Farooquee",
        "url":   "https://www.linkedin.com/in/nazmul-farooquee-mba-0b433b1b/",
    },
    license_info={"name": "MIT"},
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
app.include_router(analyse.router,      prefix=API_PREFIX)
app.include_router(predict.router,      prefix=API_PREFIX)
app.include_router(models_info.router,  prefix=API_PREFIX)
app.include_router(fhir.router,         prefix=API_PREFIX)


@app.get("/", tags=["System"])
def root():
    return {
        "service": "ClinIQ Clinical Intelligence API",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }
