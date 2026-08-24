from datetime import date

from pydantic import BaseModel, Field

# daily prediction schema, shared by both the naive and LSTM hybrid response models
class DailyPricePrediction(BaseModel):
    target_date: date
    predicted_price: float


# naive prediction response schemas
class NaivePricePredictionResponse(BaseModel):
    rice_type: str
    last_known_price: float
    last_known_date: date
    predictions: list[DailyPricePrediction]


# naive batch prediction schemas
class NaiveBatchResultItem(BaseModel):
    rice_type: str
    last_known_price: float
    last_known_date: date
    predictions: list[DailyPricePrediction]


# response schema for batch predictions using the naive model
class NaiveBatchPredictionResponse(BaseModel):
    results: list[NaiveBatchResultItem]


# LSTM hybrid prediction response schemas
class LSTMHybridPricePredictionResponse(BaseModel):
    rice_type: str
    last_known_price: float
    last_known_date: date
    lstm_weight: float = Field(..., ge=0.0, le=1.0, description="Weight given to the LSTM prediction versus the naive baseline")
    relative_volatility: float = Field(..., description="Coefficient of variation of the last 60 known prices")
    predictions: list[DailyPricePrediction]


# LSTM hybrid batch prediction schemas
class LSTMHybridBatchResultItem(BaseModel):
    rice_type: str
    last_known_price: float
    last_known_date: date
    lstm_weight: float = Field(..., ge=0.0, le=1.0)
    relative_volatility: float
    predictions: list[DailyPricePrediction]


# response schema for batch predictions using the LSTM hybrid model
class LSTMHybridBatchPredictionResponse(BaseModel):
    results: list[LSTMHybridBatchResultItem]