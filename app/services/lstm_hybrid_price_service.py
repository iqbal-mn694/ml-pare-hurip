from app.ml.lstm_price_model import LSTMHybridPriceModel
from app.schemas.price_prediction_requests import (
    BatchPricePredictionRequest,
    PricePredictionRequest,
)
from app.schemas.price_prediction_responses import (
    LSTMHybridBatchPredictionResponse,
    LSTMHybridPricePredictionResponse,
)
from app.services.prediction_formatters import (
    to_lstm_hybrid_batch_result_item,
    to_lstm_hybrid_prediction_response,
)

class LSTMHybridPriceService:
  # initializes the LSTMHybridPriceService with an LSTMHybridPriceModel instance, which is used to predict future rice prices by blending an LSTM delta model with a naive baseline
  def __init__(self, model: LSTMHybridPriceModel) -> None:
    self._model = model

  # predict the next 30 days of price for a single rice type given a PricePredictionRequest, returning an LSTMHybridPricePredictionResponse
  def predict_next_prices(self, request: PricePredictionRequest) -> LSTMHybridPricePredictionResponse:
    raw_result = self._model.predict(request.rice_type, request.last_prices)
    return to_lstm_hybrid_prediction_response(request, raw_result)

  # predict the next 30 days of price for a batch of rice types given a BatchPricePredictionRequest, returning an LSTMHybridBatchPredictionResponse
  def predict_batch(self, request: BatchPricePredictionRequest) -> LSTMHybridBatchPredictionResponse:
    raw_results = self._model.predict_batch(
      [{"rice_type": item.rice_type, "last_prices": item.last_prices} for item in request.items]
    )
    results = [
      to_lstm_hybrid_batch_result_item(item, raw_result)
      for item, raw_result in zip(request.items, raw_results)
    ]
    return LSTMHybridBatchPredictionResponse(results=results)