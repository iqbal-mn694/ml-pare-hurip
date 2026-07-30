from fastapi import APIRouter, Depends

from app.api.deps import get_naive_price_service
from app.schemas.price_prediction_requests import (
    BatchPricePredictionRequest,
    PricePredictionRequest,
)
from app.schemas.price_prediction_responses import (
    NaiveBatchPredictionResponse,
    NaivePricePredictionResponse,
)
from app.services.naive_price_service import NaivePriceService

router = APIRouter(prefix="/naive-price", tags=["Naive Price"])


# predict the next 30 days of rice price for a specific rice type using the naive (last-known-price) model
@router.post("/predict", response_model=NaivePricePredictionResponse)
def predict_next_prices(
    request: PricePredictionRequest,
    service: NaivePriceService = Depends(get_naive_price_service),
) -> NaivePricePredictionResponse:
    return service.predict_next_prices(request)

# predict the next 30 days of rice price for a batch of rice types using the naive (last-known-price) model
@router.post("/predict/batch", response_model=NaiveBatchPredictionResponse)
def predict_next_prices_batch(
    request: BatchPricePredictionRequest,
    service: NaivePriceService = Depends(get_naive_price_service),
) -> NaiveBatchPredictionResponse:
    return service.predict_batch(request)