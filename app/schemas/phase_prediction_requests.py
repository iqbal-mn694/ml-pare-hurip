from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

# This file defines the request schemas for phase prediction endpoints. It includes models for individual and batch prediction requests, ensuring that the necessary fields are provided and validated.

class PhasePredictionInput(BaseModel):
    segment_id: Optional[str] = Field(default=None, examples=["327801004"])
    subsegment: str = Field(..., examples=["A1"], description="Subsegment identifier")
    current_phase: str = Field(..., examples=["4.0"], description="Phase code observed this month")
    previous_phase: str = Field(..., examples=["3.1"], description="Phase code observed the month before")
    district_code: str = Field(..., examples=["327801"], description="6-digit district code")
    month: int = Field(..., ge=1, le=12, description="Current calendar month (1-12)")
    year: int = Field(
        default_factory=lambda: date.today().year,
        ge=2020,
        le=2100,
        description="Current calendar year, defaults to the current year when omitted",
    )


# Backward-compatible alias used across services and routers.
PhasePredictionItem = PhasePredictionInput


class PhasePredictionRequest(PhasePredictionInput):
    pass


class BatchPhasePredictionRequest(BaseModel):
    items: list[PhasePredictionInput]
