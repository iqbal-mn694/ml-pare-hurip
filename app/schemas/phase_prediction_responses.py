from typing import Optional

from pydantic import BaseModel, Field

# horizon prediction schema for the Markov Chain model
class MarkovHorizonPrediction(BaseModel):
    horizon_months: int
    predicted_phase: str
    transition_probability: float = Field(..., ge=0.0, le=1.0, description="Transition probability of the predicted phase (0-1)")


# response schema for predictions using the Markov Chain model
class MarkovPredictionResponse(BaseModel):
    segment_id: Optional[str] = None
    subsegment: str
    current_phase: str
    predictions: list[MarkovHorizonPrediction]


# batch result schemas for the Markov Chain model
class MarkovBatchResultItem(BaseModel):
    segment_id: Optional[str] = None
    subsegment: str
    current_phase: str
    predictions: list[MarkovHorizonPrediction]


# response schema for batch predictions using the Markov Chain model
class MarkovBatchPredictionResponse(BaseModel):
    results: list[MarkovBatchResultItem]


class HorizonPrediction(BaseModel):
    horizon_months: int
    target_year: int
    target_month: int
    predicted_phase: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Tingkat keyakinan model terhadap prediksi ini (0-1)")


# response schema for predictions using the Random Forest model
class RandomForestPredictionResponse(BaseModel):
    segment_id: Optional[str] = None
    subsegment: str
    district_code: str
    last_known_phase: str
    last_known_year: int
    last_known_month: int
    predictions: list[HorizonPrediction]


# batch prediction schemas for the Random Forest model
class RandomForestBatchResultItem(BaseModel):
    segment_id: Optional[str] = None
    subsegment: str
    district_code: str
    last_known_phase: str
    last_known_year: int
    last_known_month: int
    predictions: list[HorizonPrediction]


# response schema for batch predictions using the Random Forest model
class RandomForestBatchPredictionResponse(BaseModel):
    results: list[RandomForestBatchResultItem]
