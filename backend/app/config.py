"""
Application Configuration
Uses Pydantic Settings for type-safe environment variables
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment"""

    # Database
    NEON_DATABASE_URL: str
    DB_POOL_SIZE: int = 20

    # LLM API
    PERPLEXITY_API_KEY: str
    PERPLEXITY_MODEL: str = "llama-3.1-sonar-small-128k-online"

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Application
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Performance
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
