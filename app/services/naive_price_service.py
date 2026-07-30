from app.ml.naive_price_model import NaivePriceModel
from app.schemas.price_prediction_requests import (
    BatchPricePredictionRequest,
    PricePredictionRequest,
)
from app.schemas.price_prediction_responses import (
    NaiveBatchPredictionResponse,
    NaivePricePredictionResponse,
)
from app.services.prediction_formatters import (
    to_naive_batch_result_item,
    to_naive_prediction_response,
)

class NaivePriceService:
  # initializes the NaivePriceService with a NaivePriceModel instance, which is used to predict future rice prices by repeating the last known price
  def __init__(self, model: NaivePriceModel) -> None:
    self._model = model

  # predict the next 30 days of price for a single rice type given a PricePredictionRequest, returning a NaivePricePredictionResponse
  def predict_next_prices(self, request: PricePredictionRequest) -> NaivePricePredictionResponse:
    raw_result = self._model.predict(request.rice_type, request.last_prices)
    return to_naive_prediction_response(request, raw_result)

  # predict the next 30 days of price for a batch of rice types given a BatchPricePredictionRequest, returning a NaiveBatchPredictionResponse
  def predict_batch(self, request: BatchPricePredictionRequest) -> NaiveBatchPredictionResponse:
    raw_results = self._model.predict_batch(
      [{"rice_type": item.rice_type, "last_prices": item.last_prices} for item in request.items]
    )
    results = [
      to_naive_batch_result_item(item, raw_result)
      for item, raw_result in zip(request.items, raw_results)
    ]
    return NaiveBatchPredictionResponse(results=results)