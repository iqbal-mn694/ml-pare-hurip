import numpy as np

WINDOW_TARGET = 30

class NaivePriceModel:
  # predicts the next window_target days of price for a given rice type by repeating the last known price, assuming no change from the most recent observation
  def predict(self, rice_type: str, last_prices: list[float]) -> dict:
    last_known_price = float(last_prices[-1])
    predicted_prices = np.full(WINDOW_TARGET, last_known_price).tolist()

    return {
      "predicted_prices": predicted_prices,
      "last_known_price": last_known_price,
    }

  # predicts for multiple rice types at once
  def predict_batch(self, rows: list[dict]) -> list[dict]:
    return [self.predict(row["rice_type"], row["last_prices"]) for row in rows] 
