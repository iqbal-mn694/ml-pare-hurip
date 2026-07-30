from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

# This file defines the request schemas for rice price prediction endpoints. It includes models for individual and batch prediction requests, ensuring that the necessary fields are provided and validated.

# 61 raw prices are required to compute 60 daily deltas (window_input = 60)
REQUIRED_PRICE_COUNT = 61

class PricePredictionInput(BaseModel):
    rice_type: str = Field(..., examples=["Beras Kualitas Super I"], description="Rice quality type")
    last_prices: list[float] = Field(
        ...,
        min_length=REQUIRED_PRICE_COUNT,
        max_length=REQUIRED_PRICE_COUNT,
        description="Last 61 consecutive daily prices, chronologically ordered, used to compute 60 daily deltas",
    )
    last_price_date: date = Field(
        default_factory=date.today,
        description="Calendar date corresponding to the last entry in last_prices",
    )


# Backward-compatible alias used across services and routers.
PricePredictionItem = PricePredictionInput


class PricePredictionRequest(PricePredictionInput):
    pass


class BatchPricePredictionRequest(BaseModel):
    items: list[PricePredictionInput]