from fastapi import APIRouter

from app.api.routers import markov, random_forest, naive_price, lstm_hybrid_price
from app.core.config import get_settings

# routes
api_router = APIRouter(prefix=get_settings().api_v1_prefix)
api_router.include_router(markov.router)
api_router.include_router(random_forest.router)
api_router.include_router(naive_price.router)
api_router.include_router(lstm_hybrid_price.router)