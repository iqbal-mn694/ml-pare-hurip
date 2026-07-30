from fastapi import APIRouter, Depends

from app.api.deps import get_lstm_hybrid_price_service
from app.schemas.price_prediction_requests import (
    BatchPricePredictionRequest,
    PricePredictionRequest,
)
from app.schemas.price_prediction_responses import (
    LSTMHybridBatchPredictionResponse,
    LSTMHybridPricePredictionResponse,
)
from app.services.lstm_hybrid_price_service import LSTMHybridPriceService

router = APIRouter(prefix="/lstm-hybrid-price", tags=["LSTM Hybrid Price"])


# predict the next 30 days of rice price for a specific rice type using the LSTM delta model blended with a naive baseline
@router.post("/predict", response_model=LSTMHybridPricePredictionResponse)
def predict_next_prices(
    request: PricePredictionRequest,
    service: LSTMHybridPriceService = Depends(get_lstm_hybrid_price_service),
) -> LSTMHybridPricePredictionResponse:
    return service.predict_next_prices(request)

# predict the next 30 days of rice price for a batch of rice types using the LSTM delta model blended with a naive baseline
@router.post("/predict/batch", response_model=LSTMHybridBatchPredictionResponse)
def predict_next_prices_batch(
    request: BatchPricePredictionRequest,
    service: LSTMHybridPriceService = Depends(get_lstm_hybrid_price_service),
) -> LSTMHybridBatchPredictionResponse:
    return service.predict_batch(request)