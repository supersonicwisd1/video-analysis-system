"""Configuration settings for Video RAG API"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    app_name: str = "Video RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Keys
    gemini_api_key: str
    openai_api_key: str = ""  # Optional
    
    # Database
    database_url: str = "sqlite:///./video_rag.db"
    
    # Storage
    storage_path: str = "./storage"
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    
    # Processing
    max_video_duration: int = 600  # 10 minutes max
    frames_per_video: int = 8
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # CORS
    allowed_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()