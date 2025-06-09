"""Video processing and management endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
import uuid
from typing import List, Optional, Dict, Any
from app.api.models.requests import ProcessVideoRequest, BatchProcessRequest, ProcessingOptions
from app.core.video_processor import VideoRAGProcessor
from app.core.cache import VideoCache
from app.core.config import settings
from app.api.dependencies import get_video_processor, get_cache
from app.api.models.responses import (
    VideoProcessingResponse,
    BatchProcessingResponse,
    VideoInfoResponse,
    ProcessingStatusResponse,
    VideoMetadata
)
import re
from datetime import datetime

router = APIRouter()

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(url))
        if match:
            return match.group(1)
    
    raise ValueError('Could not extract video ID from URL')

@router.post("/process", response_model=VideoProcessingResponse)
async def process_video(
    video_request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    processor: VideoRAGProcessor = Depends(get_video_processor),
    cache: VideoCache = Depends(get_cache)
):
    """Process a video with specified options"""
    try:
        # Extract video ID from URL
        video_id = extract_video_id(str(video_request.youtube_url))
        
        # Check if already processed
        cached_metadata = await cache.get_video_metadata(video_id)
        if cached_metadata:
            # Convert cached metadata to VideoMetadata model
            metadata = VideoMetadata(
                video_id=video_id,
                title=cached_metadata.get("title", "Unknown"),
                duration=float(cached_metadata.get("duration", 0)),
                youtube_url=str(video_request.youtube_url),
                processed_at=cached_metadata.get("processed_at", datetime.now()),
                status="completed",
                quality=cached_metadata.get("quality", "auto"),
                sections=cached_metadata.get("sections", []),
                segments=cached_metadata.get("segments", []),
                frame_count=cached_metadata.get("frame_count"),
                error=cached_metadata.get("error")
            )
            return VideoProcessingResponse(
                video_id=video_id,
                status="completed",
                message="Video already processed",
                metadata=metadata
            )

        # Start processing in background
        background_tasks.add_task(
            processor.process_video,
            str(video_request.youtube_url),  # Convert HttpUrl to str
            options=video_request.options.dict() if video_request.options else None
        )

        # For new processing, return minimal metadata
        metadata = VideoMetadata(
            video_id=video_id,
            title="Processing...",
            duration=0.0,
            youtube_url=str(video_request.youtube_url),
            processed_at=datetime.now(),
            status="processing",
            quality=video_request.options.quality if video_request.options else "auto",
            sections=[],
            segments=[],
            frame_count=None,
            error=None
        )

        return VideoProcessingResponse(
            video_id=video_id,
            status="processing",
            message="Video processing started",
            metadata=metadata
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchProcessingResponse)
async def batch_process_videos(
    batch_request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    processor: VideoRAGProcessor = Depends(get_video_processor),
    cache: VideoCache = Depends(get_cache)
):
    """Process multiple videos in batch"""
    try:
        batch_id = str(uuid.uuid4())
        videos = batch_request.youtube_urls
        options = batch_request.options.dict() if batch_request.options else None

        # Initialize batch status
        await cache.set_batch_status(batch_id, {
            "status": "processing",
            "videos": {v.video_id: "pending" for v in videos}
        })

        # Start processing each video
        for video in videos:
            background_tasks.add_task(
                processor.process_video,
                video.youtube_url,
                options=options,
                batch_id=batch_id
            )

        return BatchProcessingResponse(
            batch_id=batch_id,
            status="processing",
            message=f"Processing {len(videos)} videos",
            total_videos=len(videos)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{video_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    video_id: str,
    cache: VideoCache = Depends(get_cache)
):
    """Get video processing status"""
    try:
        status = await cache.get_processing_status(video_id)
        if not status:
            raise HTTPException(status_code=404, detail="Video not found")

        metadata = await cache.get_video_metadata(video_id)
        return ProcessingStatusResponse(
            video_id=video_id,
            status=status,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/{batch_id}", response_model=BatchProcessingResponse)
async def get_batch_status(
    batch_id: str,
    cache: VideoCache = Depends(get_cache)
):
    """Get batch processing status"""
    try:
        status = await cache.get_batch_status(batch_id)
        if not status:
            raise HTTPException(status_code=404, detail="Batch not found")

        return BatchProcessingResponse(
            batch_id=batch_id,
            status=status["status"],
            message=f"Batch processing status: {status['status']}",
            total_videos=len(status["videos"]),
            completed_videos=sum(1 for v in status["videos"].values() if v == "completed"),
            failed_videos=sum(1 for v in status["videos"].values() if v == "failed")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info/{video_id}", response_model=VideoInfoResponse)
async def get_video_info(
    video_id: str,
    processor: VideoRAGProcessor = Depends(get_video_processor),
    cache: VideoCache = Depends(get_cache)
):
    """Get video information and processing status"""
    try:
        # Try cache first
        metadata = await cache.get_video_metadata(video_id)
        if metadata:
            return VideoInfoResponse(
                video_id=video_id,
                status="completed",
                metadata=metadata
            )

        # Get processing status
        status = await cache.get_processing_status(video_id)
        if not status:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get basic video info
        video_info = await processor.get_video_info(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail="Video info not found")

        return VideoInfoResponse(
            video_id=video_id,
            status=status,
            metadata={
                "video_id": video_id,
                "title": video_info.get("title", "Unknown"),
                "duration": video_info.get("duration", 0),
                "status": status
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    processor: VideoRAGProcessor = Depends(get_video_processor),
    cache: VideoCache = Depends(get_cache)
):
    """Delete a processed video and its data"""
    try:
        # Delete from storage
        success = await processor.delete_video(video_id)
        if not success:
            raise HTTPException(status_code=404, detail="Video not found")

        # Clear cache
        await cache.delete_video_metadata(video_id)
        await cache.delete_processing_status(video_id)

        return {"message": "Video deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segments/{video_id}")
async def get_video_segments(
    video_id: str,
    start_time: Optional[float] = Query(None, description="Start time in seconds"),
    limit: Optional[int] = Query(10, description="Number of segments to return"),
    quality: Optional[str] = Query("auto", description="Video quality preference"),
    cache: VideoCache = Depends(get_cache)
):
    """Get video segments with optional time range and quality settings"""
    try:
        # Try cache first
        segments = await cache.get_video_segments(video_id, start_time, limit)
        if segments:
            return {
                "video_id": video_id,
                "segments": segments,
                "quality": quality,
                "cached": True
            }

        # If not in cache, return empty list
        return {
            "video_id": video_id,
            "segments": [],
            "quality": quality,
            "cached": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))