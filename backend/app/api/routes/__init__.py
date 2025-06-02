"""API route modules"""
from app.api.routes.videos import router as videos
from app.api.routes.search import router as search
from app.api.routes.health import router as health

__all__ = ['videos', 'search', 'health'] 