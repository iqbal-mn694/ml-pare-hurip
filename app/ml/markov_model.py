import joblib
import pandas as pd
from pathlib import Path

# wraps a transition matrix with probability distributions for phase transitions, and provides methods to predict the next phase given a current phase --
class MarkovChainModel:
  # load the transition matrix from a joblib file in the given artifact directory
  def __init__(self, artifact_dir: Path) -> None:
    self._transition_matrix: pd.DataFrame = joblib.load(artifact_dir)

  # predict the next phase given a current phase, returning the predicted phase and its probability; if the current phase is not in the transition matrix, return the current phase with a probability of 0.0
  def predict(self, current_phase: str) -> tuple[str, float]:
    if current_phase not in self._transition_matrix.index:
      return current_phase, 0.0
    
    row = self._transition_matrix.loc[current_phase]
    predicted_phase = row.idxmax()
    probability = float(row.max())
    return predicted_phase, probability

  # predict many current phases at once
  def predict_batch(self, current_phases: list[str]) -> list[tuple[str, float]]:
    return [self.predict(phase) for phase in current_phases]