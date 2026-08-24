from fastapi import HTTPException

from app.ml.markov_model import MarkovChainModel
from app.schemas.phase_prediction_requests import (
    BatchPhasePredictionRequest,
    PhasePredictionRequest,
)
from app.schemas.phase_prediction_responses import (
    MarkovBatchPredictionResponse,
    MarkovPredictionResponse,
)
from app.services.prediction_formatters import (
    to_markov_batch_result_item,
    to_markov_prediction_response,
)

class MarkovService:
  # initializes the MarkovService with a MarkovChainModel instance, which is used to predict future rice growth phase(s) for every horizon based on the current phase
  def __init__(self, model: MarkovChainModel) -> None:
    self._model = model

  # predict future phases for all horizons given a PhasePredictionRequest, returning a MarkovPredictionResponse with the predicted phase and its transition probability per horizon
  def predict_next_phase(self, request: PhasePredictionRequest) -> MarkovPredictionResponse:
    raw_predictions = self._predict_with_validation(request.current_phase)
    return to_markov_prediction_response(request, raw_predictions)

  # predict future phases for many items at once given a BatchPhasePredictionRequest, returning a MarkovBatchPredictionResponse
  def predict_batch(self, request: BatchPhasePredictionRequest) -> MarkovBatchPredictionResponse:
    current_phases = [item.current_phase for item in request.items]
    try:
      raw_results = self._model.predict_batch(current_phases)
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc))

    results = [
      to_markov_batch_result_item(item, raw_predictions)
      for item, raw_predictions in zip(request.items, raw_results)
    ]

    return MarkovBatchPredictionResponse(results=results)

  # runs a single prediction and translates unknown-phase errors into a 422 response
  def _predict_with_validation(self, current_phase: str) -> dict[int, tuple[str, float]]:
    try:
      return self._model.predict(current_phase)
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc))
