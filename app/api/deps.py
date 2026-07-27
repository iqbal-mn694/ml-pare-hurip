from fastapi import Depends

from app.core.config import get_settings
from app.ml.markov_model import MarkovChainModel
from app.ml.random_forest_model import RandomForestPhaseModel
from app.services.markov_service import MarkovService
from app.services.random_forest_service import RandomForestService

# initialize the models to None; they will be loaded on first access
_markov_model: MarkovChainModel | None = None
_random_forest_model: RandomForestPhaseModel | None = None

# get the markov model, loading it if it hasn't been loaded yet
def get_markov_model() -> MarkovChainModel:
    global _markov_model
    if _markov_model is None:
      settings = get_settings()
      path = settings.ml_artifacts_dir / "markov_transition_matrix.parquet"
      _markov_model = MarkovChainModel(path)
    return _markov_model

# get the random forest model, loading it if it hasn't been loaded yet
def get_random_forest_model() -> RandomForestPhaseModel:
    global _random_forest_model
    if _random_forest_model is None:
      settings = get_settings()
      _random_forest_model = RandomForestPhaseModel(settings.ml_artifacts_dir)
    return _random_forest_model

# load the models at startup to fail fast if any model is missing or broken
def preload_models() -> None:
    get_markov_model()
    get_random_forest_model()


# get the markov service, which depends on the markov model
def get_markov_service(model: MarkovChainModel = Depends(get_markov_model)) -> MarkovService:
   return MarkovService(model)

# get the random forest service, which depends on the random forest model
def get_random_forest_service(
      model: RandomForestPhaseModel = Depends(get_random_forest_model),
) -> RandomForestService:
   return RandomForestService(model)

