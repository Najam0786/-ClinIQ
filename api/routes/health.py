"""api/routes/health.py — GET /api/v1/health"""

from fastapi import APIRouter
from api.schemas import HealthResponse
from api.dependencies import list_all_models

router = APIRouter()

API_VERSION = "2.1.0"


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API status and count of trained models available."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        trained_models_available=len(list_all_models()),
    )
