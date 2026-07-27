from app.schemas.phase_prediction_requests import PhasePredictionInput
from app.schemas.phase_prediction_responses import (
    MarkovHorizonPrediction,
    HorizonPrediction,
    MarkovBatchResultItem,
    RandomForestBatchResultItem,
    RandomForestPredictionResponse,
)


def _add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    zero_based_month = month - 1 + offset
    new_year = year + zero_based_month // 12
    new_month = zero_based_month % 12 + 1
    return new_year, new_month

def to_markov_horizon_predictions(raw_predictions: dict[int, str]) -> list[MarkovHorizonPrediction]:
    return [
        MarkovHorizonPrediction(horizon_months=horizon, predicted_phase=phase)
        for horizon, phase in sorted(raw_predictions.items())
    ]

def to_random_forest_horizon_predictions(
    year: int,
    month: int,
    raw_predictions: dict[int, tuple[str, float]],
) -> list[HorizonPrediction]:
    return [
        HorizonPrediction(
            horizon_months=horizon,
            target_year=target_year,
            target_month=target_month,
            predicted_phase=predicted_phase,
            confidence=confidence,
        )
        for horizon, (predicted_phase, confidence) in sorted(raw_predictions.items())
        for target_year, target_month in [_add_months(year, month, horizon)]
    ]

def to_markov_batch_result_item(
    item: PhasePredictionInput,
    predicted_phase: str,
    probability: float,
) -> MarkovBatchResultItem:
    return MarkovBatchResultItem(
        segment_id=item.segment_id,
        subsegment=item.subsegment,
        predicted_phase=predicted_phase,
        transition_probability=probability,
    )

def to_random_forest_batch_result_item(
    item: PhasePredictionInput,
    raw_predictions: dict[int, tuple[str, float]],
) -> RandomForestBatchResultItem:
    return RandomForestBatchResultItem(
        segment_id=item.segment_id,
        subsegment=item.subsegment,
        district_code=item.district_code,
        last_known_phase=item.current_phase,
        last_known_year=item.year,
        last_known_month=item.month,
        predictions=to_random_forest_horizon_predictions(item.year, item.month, raw_predictions),
    )

def to_random_forest_prediction_response(
    item: PhasePredictionInput,
    raw_predictions: dict[int, tuple[str, float]],
) -> RandomForestPredictionResponse:
    return RandomForestPredictionResponse(
        segment_id=item.segment_id,
        subsegment=item.subsegment,
        district_code=item.district_code,
        last_known_phase=item.current_phase,
        last_known_year=item.year,
        last_known_month=item.month,
        predictions=to_random_forest_horizon_predictions(item.year, item.month, raw_predictions),
    )