from fastapi import APIRouter, Depends

from app.api.deps import get_markov_service
from app.schemas.phase_prediction_requests import (
    BatchPhasePredictionRequest,
    PhasePredictionRequest,
)
from app.schemas.phase_prediction_responses import (
    MarkovBatchPredictionResponse,
    MarkovPredictionResponse,
)
from app.services.markov_service import MarkovService

router = APIRouter(prefix="/markov", tags=["Markov Chain"])


# predict the next rice growth phase for a specific segment and subsegment using the Markov Chain model
@router.post("/predict", response_model=MarkovPredictionResponse)
def predict_next_phase(
    request: PhasePredictionRequest,
    service: MarkovService = Depends(get_markov_service),
) -> MarkovPredictionResponse:
    return service.predict_next_phase(request)

# predict the next rice growth phase for a batch of segments and subsegments using the Markov Chain model
@router.post("/predict/batch", response_model=MarkovBatchPredictionResponse)
def predict_next_phase_batch(
    request: BatchPhasePredictionRequest,
    service: MarkovService = Depends(get_markov_service),
) -> MarkovBatchPredictionResponse:
    return service.predict_batch(request)