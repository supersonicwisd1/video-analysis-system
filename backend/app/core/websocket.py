from typing import Dict, Set, Any, Optional
import json
import asyncio
from fastapi import WebSocket
from datetime import datetime

class ConnectionManager:
    """Manage WebSocket connections and broadcast messages"""
    
    def __init__(self):
        # Map of video_id to set of active connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of connection to its metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, video_id: str, client_id: str) -> None:
        """Connect a new client"""
        await websocket.accept()
        if video_id not in self.active_connections:
            self.active_connections[video_id] = set()
        self.active_connections[video_id].add(websocket)
        self.connection_metadata[websocket] = {
            "video_id": video_id,
            "client_id": client_id,
            "connected_at": datetime.utcnow().isoformat()
        }
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a client"""
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            video_id = metadata["video_id"]
            self.active_connections[video_id].remove(websocket)
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]
            del self.connection_metadata[websocket]
    
    async def broadcast_to_video(self, video_id: str, message: Dict[str, Any]) -> None:
        """Broadcast message to all clients watching a video"""
        if video_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[video_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.add(connection)
            
            # Clean up disconnected clients
            for connection in disconnected:
                await self.disconnect(connection)
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send message to a specific client"""
        try:
            await websocket.send_json(message)
        except:
            await self.disconnect(websocket)
    
    async def stream_chat_response(self, websocket: WebSocket, response_generator) -> None:
        """Stream chat response chunks to client"""
        try:
            async for chunk in response_generator:
                await websocket.send_json({
                    "type": "chat_chunk",
                    "content": chunk
                })
            await websocket.send_json({
                "type": "chat_complete"
            })
        except:
            await self.disconnect(websocket)

# Create a singleton instance
manager = ConnectionManager() 