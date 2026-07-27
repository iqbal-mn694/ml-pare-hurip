from typing import Optional

from pydantic import BaseModel, Field

# markov prediction response schemas
class MarkovPredictionResponse(BaseModel):
    predicted_phase: str
    transition_probability: float


# markov batch prediction schemas
class MarkovBatchResultItem(BaseModel):
    segment_id: Optional[str] = None
    subsegment: str
    predicted_phase: str
    transition_probability: float


# response schema for batch predictions using the Markov Chain model
class MarkovBatchPredictionResponse(BaseModel):
    results: list[MarkovBatchResultItem]


# horizon prediction schemas
class MarkovHorizonPrediction(BaseModel):
    horizon_months: int
    predicted_phase: str


class HorizonPrediction(BaseModel):
    horizon_months: int
    target_year: int
    target_month: int
    predicted_phase: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in this prediction (0-1)")


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
