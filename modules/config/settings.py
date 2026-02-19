import os
from pydantic_settings import BaseSettings
from typing import List
import json

class Settings(BaseSettings):
    # --- MongoDB Configuration ---
    MONGODB_USER: str
    MONGODB_PASSWORD: str
    MONGODB_DB_NAME: str
    MONGODB_URL: str

    # --- MinIO Configuration ---
    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_SECURE: bool = False
    
    MINIO_BUCKET_RAW: str
    MINIO_BUCKET_PROCESSED: str
    MINIO_BUCKET_MODELS: str

    # --- Application Settings ---
    APP_NAME: str = "Skintel"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Process CORS_ORIGINS as a list
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # --- Security ---
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        # This tells Pydantic to look for the .env file in the root directory
        env_file = ".env"
        case_sensitive = True
        # IMPORTANT: Allow extra fields so PYTHONPATH and other env vars don't cause errors
        extra = "ignore"

# Create a single instance of settings to be imported elsewhere
settings = Settings()