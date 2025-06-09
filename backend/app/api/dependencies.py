"""API dependencies for FastAPI endpoints"""
from typing import Annotated
from fastapi import Depends, Request
from app.core.video_processor import VideoRAGProcessor
from app.core.cache import VideoCache, get_cache

async def get_video_processor(request: Request) -> VideoRAGProcessor:
    """Get the global video processor instance"""
    return request.app.state.processor

async def get_cache(request: Request) -> VideoCache:
    """Get the global cache instance"""
    return request.app.state.processor.cache if request.app.state.processor else None 