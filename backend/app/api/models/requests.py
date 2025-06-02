"""Pydantic request models"""
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class ProcessVideoRequest(BaseModel):
    youtube_url: HttpUrl = Field(..., description="YouTube video URL")
    extract_frames: bool = Field(default=True, description="Whether to extract video frames")
    max_duration: int = Field(default=300, description="Maximum video duration to process (seconds)")

class SearchRequest(BaseModel):
    video_id: str = Field(..., description="Video ID to search in")
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to return")

class ChatRequest(BaseModel):
    video_id: str = Field(..., description="Video ID for chat context")
    message: str = Field(..., min_length=1, max_length=1000, description="Chat message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn chat")