"""FastAPI Application Entry Point"""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.core.config import get_settings
from app.core.video_processor import VideoRAGProcessor
from app.core.cache import get_cache
from app.api.routes.videos import router as videos_router
from app.api.routes.search import router as search_router
from app.api.routes.health import router as health_router

# Load environment variables
load_dotenv()

# Global processor instance
processor: Optional[VideoRAGProcessor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global processor
    
    # Startup
    print("🚀 Starting Video RAG API...")

    # Initialize Redis cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache = await get_cache(redis_url)
    
    # Initialize video processor
    processor = VideoRAGProcessor(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        storage_path=os.getenv("STORAGE_PATH", "./storage"),
        cache=cache,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    try:
        await processor.initialize()
        # Store in app state
        app.state.processor = processor
        print("✅ Video RAG API ready!")
    except Exception as e:
        print(f"❌ Failed to initialize Video RAG API: {e}")
        raise
    
    yield
    
    # Shutdown
    print("👋 Shutting down Video RAG API...")
    if processor:
        try:
            await processor.cleanup()
            await processor.cache.close()
        except Exception as e:
            print(f"❌ Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Video Analysis API",
    description="API for analyzing and searching YouTube videos",
    version="1.0.0",
    lifespan=lifespan,
    # Configure trailing slash behavior
    redirect_slashes=False  # Disable automatic trailing slash redirects
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Add specific origins
    allow_credentials=False,  # Set to False since we don't use credentials
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
            "detail": str(exc) if get_settings().debug else "Something went wrong"
        }
    )

# Include routers
app.include_router(health_router, prefix="/api/v1/health", tags=["health"])
app.include_router(videos_router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(search_router, prefix="/api/v1/search", tags=["search"])

@app.get("/")
async def root():
    return {
        "app": "Video Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "processor": "initialized" if processor else "not initialized",
            "redis": "connected" if processor and processor.cache else "not connected"
        }
    }