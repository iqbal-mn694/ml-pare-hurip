from app.ml.random_forest_model import RandomForestPhaseModel
from app.schemas.phase_prediction_requests import (
    BatchPhasePredictionRequest,
    PhasePredictionRequest,
)
from app.schemas.phase_prediction_responses import (
    RandomForestBatchPredictionResponse,
    RandomForestPredictionResponse,
)
from app.services.prediction_formatters import (
  to_random_forest_batch_result_item,
  to_random_forest_prediction_response,
)


class RandomForestService:
  # initializes the RandomForestService with a RandomForestPhaseModel and provides a method to predict the next rice growth phase(s) for a given horizon
  def __init__(self, model: RandomForestPhaseModel):
    self._model = model

  def _build_prediction_rows(
      self,
      request: BatchPhasePredictionRequest,
  ) -> list[dict[str, str | int]]:
    return [item.model_dump(exclude={"segment_id"}) for item in request.items]

  # predict the next rice growth phase(s) for a given horizon using the RandomForestPhaseModel
  def predict_horizons(self, request: PhasePredictionRequest) -> RandomForestPredictionResponse:
    return to_random_forest_prediction_response(
      request,
      self._model.predict(
        current_phase=request.current_phase,
        previous_phase=request.previous_phase,
        district_code=request.district_code,
        subsegment=request.subsegment,
        month=request.month,
      ),
    )

  # predict batch of rice growth phase(s) for multiple segments and subsegments using the RandomForestPhaseModel
  def predict_batch(self, request: BatchPhasePredictionRequest) -> RandomForestBatchPredictionResponse:
    raw_results = self._model.predict_batch(self._build_prediction_rows(request))

    results = [
      to_random_forest_batch_result_item(item, raw_prediction)
      for item, raw_prediction in zip(request.items, raw_results)
    ]
    return RandomForestBatchPredictionResponse(results=results)
