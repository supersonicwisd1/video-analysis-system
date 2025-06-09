"""Health check endpoints"""
from datetime import datetime
from fastapi import APIRouter, Request
from app.core.config import get_settings
from app.api.models.responses import HealthResponse

router = APIRouter()
settings = get_settings()

@router.get("/", response_model=HealthResponse)
async def health_check(request: Request):
    """Basic health check"""
    
    # Check processor
    processor_status = "healthy"
    try:
        processor = request.app.state.processor
        if not processor or not processor.gemini_llm:
            processor_status = "unhealthy"
    except:
        processor_status = "unhealthy"
    
    return HealthResponse(
        status="healthy" if processor_status == "healthy" else "unhealthy",
        timestamp=datetime.now(),
        version=settings.app_version,
        components={
            "processor": processor_status,
            "database": "healthy",  # Add real DB check
            "storage": "healthy"    # Add real storage check
        }
    )

@router.get("/ready")
async def readiness_check(request: Request):
    """Kubernetes readiness check"""
    try:
        processor = request.app.state.processor
        if processor and processor.gemini_llm:
            return {"status": "ready"}
    except:
        pass
    
    return {"status": "not ready"}, 503

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness check"""
    return {"status": "alive"}