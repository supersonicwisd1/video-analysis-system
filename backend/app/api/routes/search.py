"""Search and chat endpoints"""
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List

from app.api.models.requests import SearchRequest, ChatRequest
from app.api.models.responses import SearchResponse, ChatResponse
from app.core.video_processor import VideoRAGProcessor

router = APIRouter()

def get_processor(request: Request) -> VideoRAGProcessor:
    """Get processor from app state"""
    return request.app.state.processor

@router.post("/", response_model=SearchResponse)
async def search_video(
    request: SearchRequest,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Search within a processed video"""
    start_time = time.time()
    
    try:
        result = await processor.search_video(request.video_id, request.query)
        response_time = time.time() - start_time
        
        return SearchResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            video_id=result["video_id"],
            response_time=response_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat_with_video(
    request: ChatRequest,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Chat with a video using conversational context"""
    try:
        # For now, treat as regular search
        # In production, you'd maintain conversation history
        result = await processor.search_video(request.video_id, request.message)
        
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"
        
        return ChatResponse(
            message=request.message,
            response=result["answer"],
            video_id=request.video_id,
            conversation_id=conversation_id,
            sources=result["sources"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.post("/visual", response_model=SearchResponse)
async def search_visual_content(
    request: SearchRequest,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Search for visual content in video frames"""
    start_time = time.time()
    
    try:
        result = await processor.search_visual_content(request.video_id, request.query)
        response_time = time.time() - start_time
        
        return SearchResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            video_id=result["video_id"],
            response_time=response_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")

@router.get("/videos/{video_id}/suggestions")
async def get_search_suggestions(
    video_id: str,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Get suggested search queries for a video"""
    # Implementation would analyze video content to suggest queries
    suggestions = [
        "What is the main topic of this video?",
        "Summarize the key points",
        "What are the most important takeaways?",
        "What examples are given?",
        "When does the speaker mention specific tools?"
    ]
    
    return {"video_id": video_id, "suggestions": suggestions}