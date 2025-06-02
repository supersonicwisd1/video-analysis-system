"""Video processing endpoints"""
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Request
from typing import List

from app.api.models.requests import ProcessVideoRequest
from app.api.models.responses import ProcessVideoResponse, VideoInfo
from app.core.video_processor import VideoRAGProcessor

router = APIRouter()

def get_processor(request: Request) -> VideoRAGProcessor:
    """Get processor from app state"""
    return request.app.state.processor

@router.post("/process", response_model=ProcessVideoResponse)
async def process_video(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Process a YouTube video for RAG search"""
    try:
        # Start processing with optional frame analysis
        result = await processor.process_video(
            str(request.youtube_url), 
            include_frames=request.extract_frames
        )
        
        return ProcessVideoResponse(
            video_id=result["video_id"],
            status=result["status"],
            message="Video processed successfully",
            video_info=VideoInfo(
                video_id=result["video_id"],
                title=result["video_info"]["title"],
                duration=result["video_info"]["duration"],
                youtube_url=str(request.youtube_url),
                processed_at=result["processed_at"],
                status=result["status"]
            )
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/", response_model=List[VideoInfo])
async def list_videos(processor: VideoRAGProcessor = Depends(get_processor)):
    """List all processed videos"""
    try:
        videos = await processor.list_processed_videos()
        return [
            VideoInfo(
                video_id=video["video_id"],
                title=video["video_info"]["title"],
                duration=video["video_info"]["duration"],
                youtube_url=video["youtube_url"],
                processed_at=video["processed_at"],
                status=video["status"]
            )
            for video in videos
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}", response_model=VideoInfo)
async def get_video(
    video_id: str,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Get video information"""
    try:
        video_info = await processor.get_video_info(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return VideoInfo(
            video_id=video_info["video_id"],
            title=video_info["video_info"]["title"],
            duration=video_info["video_info"]["duration"],
            youtube_url=video_info["youtube_url"],
            processed_at=video_info["processed_at"],
            status=video_info["status"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Delete a processed video"""
    # Implementation would remove from storage and indices
    return {"message": f"Video {video_id} deleted successfully"}

@router.get("/videos/{video_id}/debug")
async def debug_video_data(video_id: str, processor: VideoRAGProcessor = Depends(get_processor)):
    """Debug: Check what data is stored for a video"""
    metadata = await processor.get_video_info(video_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Video not found")
    
    frame_descriptions = metadata.get("frame_descriptions", [])
    
    return {
        "video_id": video_id,
        "frame_count": len(frame_descriptions),
        "frame_descriptions": frame_descriptions[:2],  # Show first 2 for debugging
        "has_frame_data": len(frame_descriptions) > 0
    }