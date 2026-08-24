import joblib
import pandas as pd
from pathlib import Path

# wraps per-horizon transition matrices with probability distributions for phase transitions, and provides methods to predict future phases given a current phase --
class MarkovChainModel:
  # load the combined multi-horizon transition matrix parquet and the valid phase list from the given artifact directory
  def __init__(self, artifact_dir: Path) -> None:
    df = pd.read_parquet(artifact_dir / "markov_transition_matrix.parquet")
    self._transition_matrices: dict[int, pd.DataFrame] = {
      int(horizon): group.drop(columns="horizon").set_index("phase_from")
      for horizon, group in df.groupby("horizon")
    }
    self._valid_phases: list[str] = joblib.load(artifact_dir / "valid_phases.joblib")

  # raises ValueError when the given phase was never seen during training
  def validate_phase(self, current_phase: str) -> None:
    if current_phase not in self._valid_phases:
      raise ValueError(
        f"Invalid phase '{current_phase}'. Supported phases: {', '.join(self._valid_phases)}"
      )

  # predict future phases for every horizon given a current phase, returning a dictionary of (predicted phase, transition probability) per horizon; if the current phase is missing from a horizon's matrix, fall back to the current phase itself with probability 0.0 -- same behavior as the training notebook
  def predict(self, current_phase: str) -> dict[int, tuple[str, float]]:
    self.validate_phase(current_phase)

    results = {}
    for horizon, matrix in self._transition_matrices.items():
      if current_phase not in matrix.index:
        results[horizon] = (current_phase, 0.0)
        continue

      row = matrix.loc[current_phase]
      predicted_phase = row.idxmax()
      probability = float(row.max())
      results[horizon] = (predicted_phase, probability)
    return results

  # predict many current phases at once, returning a list of dictionaries of predicted phases per horizon
  def predict_batch(self, current_phases: list[str]) -> list[dict[int, tuple[str, float]]]:
    return [self.predict(phase) for phase in current_phases]
