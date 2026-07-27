from app.ml.markov_model import MarkovChainModel
from app.schemas.phase_prediction_requests import (
    BatchPhasePredictionRequest,
    PhasePredictionRequest,
)
from app.schemas.phase_prediction_responses import (
    MarkovBatchPredictionResponse,
    MarkovPredictionResponse,
)
from app.services.prediction_formatters import to_markov_batch_result_item

class MarkovService:
  # initializes the MarkovService with a MarkovChainModel instance, which is used to predict the next rice growth phase(s) based on the current phase
  def __init__(self, model: MarkovChainModel) -> None:
    self._model = model

  # predict single phase given a PhasePredictionRequest, returning a MarkovPredictionResponse with the predicted phase and its transition probability
  def predict_next_phase(self, request: PhasePredictionRequest) -> MarkovPredictionResponse:
    predicted_phase, probability = self._model.predict(request.current_phase)
    return MarkovPredictionResponse(
      predicted_phase=predicted_phase,
      transition_probability=probability
    )

  # predict many phases at once given a BatchPhasePredictionRequest, returning a MarkovBatchPredictionResponse with the predicted phases and their transition probabilities
  def predict_batch(self, request: BatchPhasePredictionRequest) -> MarkovBatchPredictionResponse:
    current_phases = [item.current_phase for item in request.items]
    results = [
      to_markov_batch_result_item(item, predicted_phase, probability)
      for item, (predicted_phase, probability) in zip(
        request.items,
        self._model.predict_batch(current_phases),
      )
    ]

    return MarkovBatchPredictionResponse(results=results)