"""api/routes/models_info.py — GET /api/v1/models"""

from fastapi import APIRouter, HTTPException
from api.schemas import ModelsListResponse, ModelInfo
from api.dependencies import list_all_models, get_model_info

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse, tags=["Models"])
def list_models():
    """List all trained models saved in the registry."""
    models = list_all_models()
    return ModelsListResponse(
        count=len(models),
        models=[ModelInfo(**m) for m in models],
    )


@router.get("/models/{model_id}", response_model=ModelInfo, tags=["Models"])
def get_model(model_id: str):
    """Get metadata for a specific trained model."""
    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelInfo(**info)
