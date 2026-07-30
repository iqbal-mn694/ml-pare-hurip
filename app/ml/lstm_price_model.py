import joblib
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

WINDOW_INPUT = 60
WINDOW_TARGET = 30
VOLATILITY_DIVISOR = 0.02
MAX_CHANGE_RATIO = 0.15

class LSTMHybridPriceModel:
  # loads the trained LSTM delta model, delta scaler, and rice type encoder from the given artifact directory
  def __init__(self, artifacts_dir: Path) -> None:
    self._model = load_model(artifacts_dir / "lstm_delta_price_model.keras")
    self._delta_scaler = joblib.load(artifacts_dir / "delta_scaler.joblib")
    self._rice_type_encoder: dict[str, int] = joblib.load(artifacts_dir / "rice_type_encoder.joblib")

  # converts a sequence of raw prices into daily deltas
  def _compute_deltas(self, prices: np.ndarray) -> np.ndarray:
    return np.diff(prices)
  
  # scales, reshapes, and runs the LSTM model to predict the next window_target deltas
  def _predict_raw_deltas(self, deltas: np.ndarray, rice_type: str) -> np.ndarray:
    deltas_scaled = self._delta_scaler.transform(deltas.reshape(-1, 1)).flatten()
    delta_input = deltas_scaled[-WINDOW_INPUT:].reshape(1, WINDOW_INPUT, 1).astype(np.float32)

    type_id = self._rice_type_encoder[rice_type]
    type_input = np.array([[type_id]], dtype=np.int32)

    predicted_scaled = self._model.predict([delta_input, type_input], verbose=0)
    predicted_deltas = self._delta_scaler.inverse_transform(
      predicted_scaled.reshape(-1, 1)
    ).flatten()
    return predicted_deltas

  # reconstructs a price sequence from a starting price and a sequence of deltas
  def _reconstruct_prices(self, last_known_price: float, predicted_deltas: np.ndarray) -> np.ndarray:
    return last_known_price + np.cumsum(predicted_deltas)

  # clips the predicted price sequence to a maximum allowed change from the last known price, as a safety net against runaway accumulation
  def _clip_predictions(self, predictions: np.ndarray, last_known_price: float) -> np.ndarray:
    max_price = last_known_price * (1 + MAX_CHANGE_RATIO)
    min_price = last_known_price * (1 - MAX_CHANGE_RATIO)
    return np.clip(predictions, min_price, max_price)

  # computes relative volatility (coefficient of variation) over the input window, used to weight the LSTM prediction against the naive baseline
  def _compute_relative_volatility(self, prices: np.ndarray) -> float:
    recent_prices = prices[-WINDOW_INPUT:]
    return float(np.std(recent_prices) / np.mean(recent_prices))

  # blends the LSTM prediction with a naive (last-known-price) baseline, weighted by recent relative volatility
  def _blend_with_naive(
      self,
      lstm_predictions: np.ndarray,
      last_known_price: float,
      relative_volatility: float,
  ) -> tuple[np.ndarray, float]:
    lstm_weight = min(relative_volatility / VOLATILITY_DIVISOR, 1.0)
    naive_weight = 1 - lstm_weight
    blended = lstm_weight * lstm_predictions + naive_weight * last_known_price
    return blended, lstm_weight

  # predicts the next window_target days of price for a given rice type, given the last window_input + 1 known prices
  def predict(self, rice_type: str, last_prices: list[float]) -> dict:
    prices = np.array(last_prices, dtype=np.float64)
    last_known_price = float(prices[-1])

    deltas = self._compute_deltas(prices)
    predicted_deltas = self._predict_raw_deltas(deltas, rice_type)

    lstm_predictions = self._reconstruct_prices(last_known_price, predicted_deltas)
    lstm_predictions = self._clip_predictions(lstm_predictions, last_known_price)

    relative_volatility = self._compute_relative_volatility(prices)
    hybrid_predictions, lstm_weight = self._blend_with_naive(
      lstm_predictions, last_known_price, relative_volatility
    )

    return {
      "predicted_prices": hybrid_predictions.tolist(),
      "lstm_weight": lstm_weight,
      "relative_volatility": relative_volatility,
      "last_known_price": last_known_price,
    }

  # predicts for multiple rice types at once
  def predict_batch(self, rows: list[dict]) -> list[dict]:
    return [self.predict(row["rice_type"], row["last_prices"]) for row in rows]