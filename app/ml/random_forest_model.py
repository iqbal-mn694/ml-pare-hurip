import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

class RandomForestPhaseModel:
  # initializes a dictionary of random forest models for different prediction horizons, and provides methods to predict the next phase given a current phase and a prediction horizon 
  def __init__(self, artifacts_dir: Path) -> None:
    self._models: dict[int, RandomForestClassifier] = joblib.load(
      artifacts_dir / "rf_all_horizons.joblib"
    )

    encoders = joblib.load(artifacts_dir / "encoders.joblib")
    self._phase_encoder = encoders["phase_encoder"]
    self._district_encoder = encoders["district_encoder"]
    self._subsegment_encoder = encoders["subsegment_encoder"]

  # raises ValueError listing the accepted values when any categorical input was never seen during training
  def validate(
      self,
      current_phase: str,
      previous_phase: str,
      district_code: str,
      subsegment: str,
  ) -> None:
    valid_phases = list(self._phase_encoder.classes_)
    if current_phase not in valid_phases:
      raise ValueError(
        f"Current phase '{current_phase}' is invalid. Supported phases: {', '.join(valid_phases)}"
      )
    if previous_phase not in valid_phases:
      raise ValueError(
        f"Previous phase '{previous_phase}' is invalid. Supported phases: {', '.join(valid_phases)}"
      )
    valid_districts = list(self._district_encoder.classes_)
    if district_code not in valid_districts:
      raise ValueError(
        f"District code '{district_code}' is invalid. Supported codes: {', '.join(valid_districts)}"
      )
    valid_subsegments = list(self._subsegment_encoder.classes_)
    if subsegment not in valid_subsegments:
      raise ValueError(
        f"Subsegment '{subsegment}' is invalid. Supported subsegments: {', '.join(valid_subsegments)}"
      )

  # builds a feature matrix from a list of input rows, encoding categorical features and adding sine and cosine transformations of the month feature
  def _build_feature_matrix(self, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    month_sin = np.sin(2 * np.pi * df["month"] / 12)
    month_cos = np.cos(2 * np.pi * df["month"] / 12)

    return pd.DataFrame({
      "phase_enc": self._phase_encoder.transform(df["current_phase"]),
      "phase_lag1_enc": self._phase_encoder.transform(df["previous_phase"]),
      "district_enc": self._district_encoder.transform(df["district_code"]),
      "subsegment_enc": self._subsegment_encoder.transform(df["subsegment"]),
      "month_sin": month_sin,
      "month_cos": month_cos,
    })

  # predicts the next phase for a given current phase, previous phase, district code, subsegment, and month, returning a dictionary of predicted phases for each horizon
  def predict(
      self,
      current_phase: str,
      previous_phase: str,
      district_code: str,
      subsegment: str,
      month: int,
  ) -> dict[int, tuple[str, float]]:
    features = self._build_feature_matrix([{
      "current_phase": current_phase,
      "previous_phase": previous_phase,
      "district_code": district_code,
      "subsegment": subsegment,
      "month": month,
    }])

    results = {}
    for horizon, model in self._models.items():
      predicted_class = model.predict(features)[0]
      probabilities = model.predict_proba(features)[0]

      # get the index of the predicted class in the model's classes_ attribute to retrieve the corresponding probability
      class_index = list(model.classes_).index(predicted_class)
      confidence = float(probabilities[class_index])
      results[horizon] = (predicted_class, confidence)
    return results

  # predicts many subsegments at once, returning a list of dictionaries of predicted phases for each horizon
  def predict_batch(self, rows: list[dict]) -> list[dict[int, tuple[str, float]]]:
    features = self._build_feature_matrix(rows)

    per_horizon_results = {}
    for horizon, model in self._models.items():
      predicted_classes = model.predict(features)
      probabilities = model.predict_proba(features)
      class_list = list(model.classes_)

      per_row = []
      for i, predicted_class in enumerate(predicted_classes):
        class_index = class_list.index(predicted_class)
        confidence = float(probabilities[i][class_index])
        per_row.append((predicted_class, confidence))
      per_horizon_results[horizon] = per_row


    # reshape
    return [
      {h: per_horizon_results[h][i] for h in self._models}
      for i in range(len(rows))
    ]