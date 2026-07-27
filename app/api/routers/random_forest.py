from fastapi import APIRouter, Depends

from app.api.deps import get_random_forest_service
from app.schemas.phase_prediction_requests import (
    BatchPhasePredictionRequest,
    PhasePredictionRequest,
)
from app.schemas.phase_prediction_responses import (
    RandomForestBatchPredictionResponse,
    RandomForestPredictionResponse,
)
from app.services.random_forest_service import RandomForestService

router = APIRouter(prefix="/random-forest", tags=["Random Forest"])

# predict the next rice growth phase for a specific segment and subsegment using the Random Forest model
@router.post("/predict", response_model=RandomForestPredictionResponse)
def predict_phase_horizons(
    request: PhasePredictionRequest,
    service: RandomForestService = Depends(get_random_forest_service),
) -> RandomForestPredictionResponse:
    """Predict rice growth phase for h+1, h+2, and h+3 months for a single subsegment."""
    return service.predict_horizons(request)

# predict the next rice growth phase for a batch of segments and subsegments using the Random Forest model
@router.post("/predict/batch", response_model=RandomForestBatchPredictionResponse)
def predict_phase_horizons_batch(
    request: BatchPhasePredictionRequest,
    service: RandomForestService = Depends(get_random_forest_service),
) -> RandomForestBatchPredictionResponse:
    return service.predict_batch(request)