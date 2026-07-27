from fastapi import APIRouter

from app.api.routers import health, markov, random_forest
from app.core.config import get_settings

# routes
api_router = APIRouter(prefix=get_settings().api_v1_prefix)
api_router.include_router(health.router)
api_router.include_router(markov.router)
api_router.include_router(random_forest.router)
