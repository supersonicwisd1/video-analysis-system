from pydantic import BaseModel, Field
from typing import Optional

class SearchResult(BaseModel):
    """Search result with timestamp and content"""
    text: str = Field(..., description="Result text content")
    timestamp: float = Field(..., description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")
    youtube_link: str = Field(..., description="YouTube link with timestamp")
    confidence: float = Field(..., description="Result confidence score")
    source_type: str = Field(..., description="Type of source (transcript/visual)")
    timestamp_formatted: Optional[str] = Field(None, description="Formatted timestamp (MM:SS)") 