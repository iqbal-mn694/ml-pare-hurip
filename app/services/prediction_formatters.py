from datetime import date, timedelta
from app.schemas.phase_prediction_requests import PhasePredictionInput
from app.schemas.price_prediction_requests import PricePredictionInput
from app.schemas.phase_prediction_responses import (
    MarkovPredictionResponse,
    MarkovHorizonPrediction,
    HorizonPrediction,
    MarkovBatchResultItem,
    RandomForestBatchResultItem,
    RandomForestPredictionResponse,
)
from app.schemas.price_prediction_responses import (
    DailyPricePrediction,
    LSTMHybridBatchResultItem,
    LSTMHybridPricePredictionResponse,
    NaiveBatchResultItem,
    NaivePricePredictionResponse,
)



def _add_months(year: int, month: int, offset: int) -> tuple[int, int]:
  zero_based_month = month - 1 + offset
  new_year = year + zero_based_month // 12
  new_month = zero_based_month % 12 + 1
  return new_year, new_month

def to_markov_horizon_predictions(
    raw_predictions: dict[int, tuple[str, float]],
) -> list[MarkovHorizonPrediction]:
    return [
        MarkovHorizonPrediction(
            horizon_months=horizon,
            predicted_phase=predicted_phase,
            transition_probability=probability,
        )
        for horizon, (predicted_phase, probability) in sorted(raw_predictions.items())
    ]

def to_markov_prediction_response(
    item: PhasePredictionInput,
    raw_predictions: dict[int, tuple[str, float]],
) -> MarkovPredictionResponse:
    return MarkovPredictionResponse(
        segment_id=item.segment_id,
        subsegment=item.subsegment,
        current_phase=item.current_phase,
        predictions=to_markov_horizon_predictions(raw_predictions),
    )

def to_markov_batch_result_item(
    item: PhasePredictionInput,
    raw_predictions: dict[int, tuple[str, float]],
) -> MarkovBatchResultItem:
    return MarkovBatchResultItem(
        segment_id=item.segment_id,
        subsegment=item.subsegment,
        current_phase=item.current_phase,
        predictions=to_markov_horizon_predictions(raw_predictions),
    )

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

# generates a sequence of daily predictions starting the day after last_price_date, pairing each predicted price with its corresponding calendar date
def to_daily_price_predictions(last_price_date: date, predicted_prices: list[float]) -> list[DailyPricePrediction]:
    return [
        DailyPricePrediction(target_date=last_price_date + timedelta(days=offset + 1), predicted_price=price)
        for offset, price in enumerate(predicted_prices)
    ]

def to_naive_prediction_response(item: PricePredictionInput, raw_result: dict) -> NaivePricePredictionResponse:
    return NaivePricePredictionResponse(
        rice_type=item.rice_type,
        last_known_price=raw_result["last_known_price"],
        last_known_date=item.last_price_date,
        predictions=to_daily_price_predictions(item.last_price_date, raw_result["predicted_prices"]),
    )

def to_naive_batch_result_item(item: PricePredictionInput, raw_result: dict) -> NaiveBatchResultItem:
    return NaiveBatchResultItem(
        rice_type=item.rice_type,
        last_known_price=raw_result["last_known_price"],
        last_known_date=item.last_price_date,
        predictions=to_daily_price_predictions(item.last_price_date, raw_result["predicted_prices"]),
    )

def to_lstm_hybrid_prediction_response(item: PricePredictionInput, raw_result: dict) -> LSTMHybridPricePredictionResponse:
    return LSTMHybridPricePredictionResponse(
        rice_type=item.rice_type,
        last_known_price=raw_result["last_known_price"],
        last_known_date=item.last_price_date,
        lstm_weight=raw_result["lstm_weight"],
        relative_volatility=raw_result["relative_volatility"],
        predictions=to_daily_price_predictions(item.last_price_date, raw_result["predicted_prices"]),
    )

def to_lstm_hybrid_batch_result_item(item: PricePredictionInput, raw_result: dict) -> LSTMHybridBatchResultItem:
    return LSTMHybridBatchResultItem(
        rice_type=item.rice_type,
        last_known_price=raw_result["last_known_price"],
        last_known_date=item.last_price_date,
        lstm_weight=raw_result["lstm_weight"],
        relative_volatility=raw_result["relative_volatility"],
        predictions=to_daily_price_predictions(item.last_price_date, raw_result["predicted_prices"]),
    )