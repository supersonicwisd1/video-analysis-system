"""FastAPI Application Entry Point"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.video_processor import VideoRAGProcessor
from app.api.routes import videos, search, health

settings = get_settings()

# Global processor instance
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global processor
    
    # Startup
    print("🚀 Starting Video RAG API...")

    # Initialize video processor
    processor = VideoRAGProcessor(
        gemini_api_key=settings.gemini_api_key,
        storage_path=settings.storage_path,
        openai_api_key=settings.openai_api_key
    )
    await processor.initialize()
    
    # Store in app state
    app.state.processor = processor
    
    print("✅ Video RAG API ready!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Video RAG API...")
    if processor:
        await processor.cleanup()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered video content search and analysis",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "Something went wrong"
        }
    )

# Include routers
app.include_router(health, prefix="/health", tags=["health"])
app.include_router(videos, prefix="/api/v1/videos", tags=["videos"])
app.include_router(search, prefix="/api/v1/search", tags=["search"])

@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }
