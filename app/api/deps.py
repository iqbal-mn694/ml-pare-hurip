from fastapi import Depends

from app.core.config import get_settings
from app.ml.markov_model import MarkovChainModel
from app.ml.random_forest_model import RandomForestPhaseModel
from app.ml.naive_price_model import NaivePriceModel
from app.ml.lstm_price_model import LSTMHybridPriceModel
from app.services.markov_service import MarkovService
from app.services.random_forest_service import RandomForestService
from app.services.naive_price_service import NaivePriceService
from app.services.lstm_hybrid_price_service import LSTMHybridPriceService

# initialize the models to None; they will be loaded on first access
_markov_model: MarkovChainModel | None = None
_random_forest_model: RandomForestPhaseModel | None = None
_naive_price_model: NaivePriceModel | None = None
_lstm_hybrid_price_model: LSTMHybridPriceModel | None = None

# get the markov model, loading it if it hasn't been loaded yet
def get_markov_model() -> MarkovChainModel:
    global _markov_model
    if _markov_model is None:
      settings = get_settings()
      _markov_model = MarkovChainModel(settings.ml_artifacts_dir)
    return _markov_model

# get the random forest model, loading it if it hasn't been loaded yet
def get_random_forest_model() -> RandomForestPhaseModel:
    global _random_forest_model
    if _random_forest_model is None:
      settings = get_settings()
      _random_forest_model = RandomForestPhaseModel(settings.ml_artifacts_dir)
    return _random_forest_model

# get the naive price model, loading it if it hasn't been loaded yet
def get_naive_price_model() -> NaivePriceModel:
    global _naive_price_model
    if _naive_price_model is None:
      _naive_price_model = NaivePriceModel()
    return _naive_price_model

# get the lstm hybrid price model, loading it if it hasn't been loaded yet
def get_lstm_hybrid_price_model() -> LSTMHybridPriceModel:
    global _lstm_hybrid_price_model
    if _lstm_hybrid_price_model is None:
      settings = get_settings()
      _lstm_hybrid_price_model = LSTMHybridPriceModel(settings.ml_artifacts_dir)
    return _lstm_hybrid_price_model

# load the models at startup to fail fast if any model is missing or broken
def preload_models() -> None:
    get_markov_model()
    get_random_forest_model()
    get_naive_price_model()
    get_lstm_hybrid_price_model()


# get the markov service, which depends on the markov model
def get_markov_service(model: MarkovChainModel = Depends(get_markov_model)) -> MarkovService:
   return MarkovService(model)

# get the random forest service, which depends on the random forest model
def get_random_forest_service(
      model: RandomForestPhaseModel = Depends(get_random_forest_model),
) -> RandomForestService:
   return RandomForestService(model)

# get the naive price service, which depends on the naive price model
def get_naive_price_service(model: NaivePriceModel = Depends(get_naive_price_model)) -> NaivePriceService:
   return NaivePriceService(model)

# get the lstm hybrid price service, which depends on the lstm hybrid price model
def get_lstm_hybrid_price_service(
      model: LSTMHybridPriceModel = Depends(get_lstm_hybrid_price_model),
) -> LSTMHybridPriceService:
   return LSTMHybridPriceService(model)