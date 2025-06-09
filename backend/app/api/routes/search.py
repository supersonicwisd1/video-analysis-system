"""Search and chat endpoints"""
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
from datetime import datetime

from app.api.models.requests import SearchRequest, ChatRequest
from app.api.models.responses import SearchResponse, ChatResponse, VideoSegment, SearchResult
from app.core.video_processor import VideoRAGProcessor
from app.api.dependencies import get_video_processor

router = APIRouter()

def get_processor(request: Request) -> VideoRAGProcessor:
    """Get processor from app state"""
    return request.app.state.processor

@router.post("/", response_model=SearchResponse)
async def search_video(
    request: SearchRequest,
    processor: VideoRAGProcessor = Depends(get_video_processor)
):
    """Search within a processed video"""
    start_time = time.time()
    
    try:
        # Get the first (and only) result from the async generator
        async for result in processor.search_video(
            request.video_id,
            request.query,
            search_type=request.search_type
        ):
            response_time = time.time() - start_time
            
            # Convert sources to search results
            search_results = []
            sources = []
            if result.get("sources"):
                for source in result["sources"]:
                    if isinstance(source, dict):  # Ensure source is a dictionary
                        # Get timestamps
                        start_time = float(source.get("timestamp", 0.0))
                        end_time = float(source.get("end_time", start_time + 30.0))  # Default 30s segment
                        
                        # Format timestamp
                        timestamp_formatted = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                        
                        # Create SearchResult object
                        search_result = SearchResult(
                            text=source.get("text", ""),
                            timestamp=start_time,
                            end_time=end_time,
                            youtube_link=source.get("youtube_link", f"https://youtube.com/watch?v={request.video_id}&t={int(start_time)}"),
                            confidence=float(source.get("confidence", 0.0)),
                            source_type=source.get("source_type", "unknown"),
                            timestamp_formatted=timestamp_formatted
                        )
                        search_results.append(search_result)
                        sources.append(search_result)
            
            return SearchResponse(
                query=request.query,
                search_type=request.search_type,
                answer=result.get("answer"),
                results=search_results,
                sources=sources,
                response_time=response_time
            )
        
        # If we get here, no results were found
        return SearchResponse(
            query=request.query,
            search_type=request.search_type,
            answer=None,
            results=[],
            sources=[],
            response_time=time.time() - start_time
        )
        
    except ValueError as e:
        print(f"❌ Search error (ValueError): {str(e)}")  # Add error logging
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ Search error (Exception): {str(e)}")  # Add error logging
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_with_video(
    request: ChatRequest,
    processor: VideoRAGProcessor = Depends(get_processor)
):
    """Chat with a video using conversational context"""
    start_time = time.time()
    
    try:
        # Get the first (and only) result from the async generator
        async for result in processor.search_video(request.video_id, request.message):
            conversation_id = request.conversation_id or f"conv_{int(time.time())}"
            
            # Convert sources to video segments
            references = []
            if result.get("sources"):
                for source in result["sources"]:
                    if isinstance(source, dict):
                        references.append(VideoSegment(
                            start_time=float(source.get("timestamp", 0.0)),
                            end_time=float(source.get("timestamp", 0.0)) + 30.0,  # Assume 30-second segments
                            content=source.get("text", ""),
                            segment_type="transcript",
                            metadata={"confidence": source.get("confidence", 0.0)}
                        ))
            
            return ChatResponse(
                video_id=request.video_id,
                conversation_id=conversation_id,
                message=request.message,
                response=result["answer"],
                references=references,
                processing_time=time.time() - start_time
            )
        
        # If we get here, no results were found
        raise HTTPException(status_code=404, detail="No results found")
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

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