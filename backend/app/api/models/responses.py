"""Pydantic response models"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoInfo(BaseModel):
    video_id: str
    title: str
    duration: int
    youtube_url: str
    processed_at: datetime
    status: str

class SearchSource(BaseModel):
    content: str
    metadata: Dict[str, Any]
    timestamp_range: Optional[str] = None
    youtube_link: Optional[str] = None
    source_type: str = "transcript"

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchSource]
    video_id: str
    response_time: Optional[float] = None

class ProcessVideoResponse(BaseModel):
    video_id: str
    status: str
    message: str
    video_info: Optional[VideoInfo] = None

class ChatResponse(BaseModel):
    message: str
    response: str
    video_id: str
    conversation_id: str
    sources: List[SearchSource]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)