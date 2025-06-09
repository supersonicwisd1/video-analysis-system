"""Response models for the video processing API"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoSegment(BaseModel):
    """Video segment with timestamp and content"""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    content: str = Field(..., description="Segment content")
    segment_type: str = Field(..., description="Type of segment (transcript, visual, metadata)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional segment metadata")

class VideoMetadata(BaseModel):
    """Video metadata and processing information"""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    duration: float = Field(..., description="Video duration in seconds")
    youtube_url: str = Field(..., description="Original YouTube URL")
    processed_at: datetime = Field(..., description="Processing timestamp")
    status: str = Field(..., description="Processing status")
    quality: str = Field("auto", description="Video quality setting")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Video sections")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Video segments")
    frame_count: Optional[int] = Field(None, description="Number of frames extracted")
    error: Optional[str] = Field(None, description="Processing error if any")

class VideoProcessingResponse(BaseModel):
    """Response for video processing request"""
    video_id: str = Field(..., description="YouTube video ID")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    metadata: Optional[VideoMetadata] = Field(None, description="Video metadata if available")

class BatchProcessingResponse(BaseModel):
    """Response for batch processing request"""
    batch_id: str = Field(..., description="Batch processing ID")
    status: str = Field(..., description="Batch processing status")
    message: str = Field(..., description="Status message")
    total_videos: int = Field(..., description="Total number of videos in batch")
    completed_videos: Optional[int] = Field(None, description="Number of completed videos")
    failed_videos: Optional[int] = Field(None, description="Number of failed videos")
    videos: Optional[Dict[str, str]] = Field(None, description="Status of individual videos")

class ProcessingStatusResponse(BaseModel):
    """Response for processing status request"""
    video_id: str = Field(..., description="YouTube video ID")
    status: str = Field(..., description="Processing status")
    progress: Optional[float] = Field(None, description="Processing progress (0-1)")
    metadata: Optional[VideoMetadata] = Field(None, description="Video metadata if available")
    error: Optional[str] = Field(None, description="Processing error if any")

class VideoInfo(BaseModel):
    """Video information model matching frontend interface"""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    duration: float = Field(..., description="Video duration in seconds")
    youtube_url: str = Field(..., description="Original YouTube URL")
    processed_at: datetime = Field(..., description="Processing timestamp")
    status: str = Field(..., description="Processing status")

class VideoInfoResponse(BaseModel):
    """Response for video information request"""
    video_id: str = Field(..., description="YouTube video ID")
    status: str = Field(..., description="Video status")
    metadata: Optional[VideoInfo] = Field(None, description="Video information")

class SearchResult(BaseModel):
    """Individual search result"""
    text: str = Field(..., description="Result text or answer")
    timestamp: float = Field(..., description="Timestamp in seconds")
    youtube_link: str = Field(..., description="YouTube link with timestamp")
    confidence: float = Field(..., description="Confidence score")

class SearchResponse(BaseModel):
    """Response for video search request"""
    query: str = Field(..., description="Search query")
    search_type: str = Field(..., description="Type of search (transcript/visual)")
    answer: Optional[str] = Field(None, description="AI-generated answer to the query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    sources: Optional[List[SearchResult]] = Field(None, description="Source segments used for the answer")
    response_time: float = Field(..., description="Search processing time in seconds")

class ChatResponse(BaseModel):
    """Response for video chat request"""
    video_id: str = Field(..., description="YouTube video ID")
    conversation_id: str = Field(..., description="Conversation ID")
    message: str = Field(..., description="Chat message")
    response: str = Field(..., description="AI response")
    references: List[VideoSegment] = Field(default_factory=list, description="Referenced video segments")
    processing_time: float = Field(..., description="Chat processing time in seconds")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")

class ProcessVideoResponse(BaseModel):
    """Response for video processing request (frontend interface)"""
    video_id: str = Field(..., description="YouTube video ID")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    video_info: Optional[VideoMetadata] = Field(None, description="Video metadata if available")

class HealthResponse(BaseModel):
    """Response for health check endpoint"""
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Status of individual components")