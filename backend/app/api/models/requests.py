"""Pydantic request models"""
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Literal
from datetime import datetime

class ProcessingOptions(BaseModel):
    """Options for video processing"""
    quality: Literal['auto', 'high', 'medium', 'low'] = 'auto'
    parallel_processing: bool = True
    preload_segments: bool = True
    max_parallel_tasks: int = Field(default=3, ge=1, le=10)
    extract_frames: bool = True
    max_duration: int = Field(default=300, description="Maximum video duration to process (seconds)")

class ProcessVideoRequest(BaseModel):
    """Request model for processing a single video"""
    youtube_url: HttpUrl = Field(..., description="YouTube video URL")
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        """Validate that the URL is a YouTube URL"""
        if not str(v).startswith(('https://www.youtube.com/', 'https://youtu.be/')):
            raise ValueError('URL must be a valid YouTube URL')
        return v

class BatchProcessRequest(BaseModel):
    """Request model for batch processing multiple videos"""
    youtube_urls: List[HttpUrl]
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    @validator('youtube_urls')
    def validate_youtube_urls(cls, v):
        """Validate that all URLs are YouTube URLs"""
        for url in v:
            if not str(url).startswith(('https://www.youtube.com/', 'https://youtu.be/')):
                raise ValueError(f'URL {url} must be a valid YouTube URL')
        return v

class SearchRequest(BaseModel):
    """Request model for video search"""
    video_id: str = Field(..., description="Video ID to search in")
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to return")
    search_type: Literal['transcript', 'visual', 'hybrid'] = 'hybrid'

class ChatRequest(BaseModel):
    """Request model for video chat"""
    video_id: str = Field(..., description="Video ID to chat with")
    message: str = Field(..., min_length=1, max_length=500, description="Chat message")
    conversation_id: Optional[str] = None

class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages"""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ProcessingUpdate(WebSocketMessage):
    """Model for processing status updates"""
    video_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    current_step: Optional[str] = None
    quality: Optional[str] = None