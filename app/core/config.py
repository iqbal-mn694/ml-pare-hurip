from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    app_name: str = "KSA Phase Prediction API"
    app_description: str = "API for KSA Phase Prediction"
    environment: Literal["local", "test", "production"] = "local"
    debug: bool = False
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    # ML model configuration 
    api_v1_prefix: str = "/api/v1"
    ml_artifacts_dir: Path = BASE_DIR / "ml" / "artifacts"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
