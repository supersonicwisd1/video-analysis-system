

from typing import Optional, Dict, Any
import json
from redis.asyncio import Redis
from datetime import timedelta

class VideoCache:
    """Redis-based cache manager for video metadata and processing status"""
    
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        self.default_ttl = timedelta(days=7)  # Cache videos for 7 days by default
        
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get cached video metadata"""
        data = await self.redis.get(f"video:{video_id}:metadata")
        return json.loads(data) if data else None
    
    async def set_video_metadata(self, video_id: str, metadata: Dict[str, Any], ttl: Optional[timedelta] = None) -> None:
        """Cache video metadata"""
        await self.redis.set(
            f"video:{video_id}:metadata",
            json.dumps(metadata),
            ex=ttl or self.default_ttl
        )
    
    async def get_processing_status(self, video_id: str) -> Optional[str]:
        """Get video processing status"""
        return await self.redis.get(f"video:{video_id}:status")
    
    async def set_processing_status(self, video_id: str, status: str, ttl: Optional[timedelta] = None) -> None:
        """Update video processing status"""
        await self.redis.set(
            f"video:{video_id}:status",
            status,
            ex=ttl or self.default_ttl
        )
    
    async def get_chat_history(self, video_id: str, conversation_id: str) -> list:
        """Get chat history for a conversation"""
        data = await self.redis.get(f"chat:{video_id}:{conversation_id}")
        return json.loads(data) if data else []
    
    async def append_chat_message(self, video_id: str, conversation_id: str, message: Dict[str, Any]) -> None:
        """Append a message to chat history"""
        history = await self.get_chat_history(video_id, conversation_id)
        history.append(message)
        await self.redis.set(
            f"chat:{video_id}:{conversation_id}",
            json.dumps(history),
            ex=self.default_ttl
        )
    
    async def clear_video_cache(self, video_id: str) -> None:
        """Clear all cached data for a video"""
        keys = await self.redis.keys(f"*{video_id}*")
        if keys:
            await self.redis.delete(*keys)
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch processing job"""
        data = await self.redis.get(f"batch:{batch_id}")
        return json.loads(data) if data else None
    
    async def set_batch_status(self, batch_id: str, status: Dict[str, Any], ttl: Optional[timedelta] = None) -> None:
        """Update batch processing status"""
        await self.redis.set(
            f"batch:{batch_id}",
            json.dumps(status),
            ex=ttl or self.default_ttl
        )
    
    async def close(self) -> None:
        """Close Redis connection"""
        await self.redis.close()

# Create a singleton instance
_cache: Optional[VideoCache] = None

async def get_cache(redis_url: str) -> VideoCache:
    """Get or create cache instance"""
    global _cache
    if _cache is None:
        _cache = VideoCache(redis_url)
    return _cache 