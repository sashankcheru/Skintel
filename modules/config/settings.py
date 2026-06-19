import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # --- MongoDB ---
    MONGODB_USER:     str
    MONGODB_PASSWORD: str
    MONGODB_DB_NAME:  str
    MONGODB_URL:      str

    # --- MinIO ---
    MINIO_ENDPOINT:         str
    MINIO_ROOT_USER:        str
    MINIO_ROOT_PASSWORD:    str
    MINIO_SECURE:           bool = False
    MINIO_BUCKET_RAW:       str
    MINIO_BUCKET_PROCESSED: str
    MINIO_BUCKET_MODELS:    str

    # --- Redis / Celery ---
    REDIS_URL:              str = "redis://redis:6379/0"
    CELERY_BROKER_URL:      str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND:  str = "redis://redis:6379/1"

    # --- Application ---
    APP_NAME:    str = "Skintel"
    APP_VERSION: str = "1.0.0"
    APP_ENV:     str = "development"
    DEBUG:       bool = True
    LOG_LEVEL:   str = "INFO"

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # --- Security ---
    SECRET_KEY:                  str
    ALGORITHM:                   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_file       = ".env"
        case_sensitive = True
        extra          = "ignore"


settings = Settings()