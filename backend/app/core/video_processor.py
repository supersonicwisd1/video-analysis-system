"""Video RAG Processor - Integration of your LlamaIndex system"""
import os
import json
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Union, TypedDict, AsyncGenerator
import cv2
import numpy as np
from pathlib import Path
import chromadb
from datetime import datetime
from openai import OpenAI
import base64
import tempfile
import yt_dlp
import re
import whisper
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import asyncio
import uuid
from .cache import VideoCache
from .websocket import manager

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import ImageDocument, Document
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import your existing functions (adapt as needed)
from app.core.video_utils import (
    extract_video_id,
    get_youtube_transcript
)

class ContentType(str, Enum):
    """Content type enum with string values for JSON serialization"""
    TRANSCRIPT = "transcript"
    SPEECH_TO_TEXT = "speech_to_text"
    VISUAL = "visual"
    METADATA = "metadata"

@dataclass
class VideoSegment:
    start_time: float
    end_time: float
    content_type: ContentType
    content: str
    metadata: Dict[str, Any]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "content_type": self.content_type.value,  # Use string value
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence
        }

class ProcessingTask:
    """Represents a video processing task"""
    def __init__(self, video_id: str, task_type: str, status: str = "pending"):
        self.video_id = video_id
        self.task_type = task_type
        self.status = status
        self.progress = 0.0
        self.error = None
        self.result = None

class VideoRAGProcessor:
    """Production video RAG processor with hybrid processing and caching"""
    
    def __init__(self, gemini_api_key: str, storage_path: str, cache: VideoCache, openai_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.cache = cache
        
        # Components (initialized in initialize())
        self.gemini_llm = None
        self.embed_model = None
        self.chroma_client = None
        self.openai_client = None
        self.whisper_model = None
        self.active_indices: Dict[str, Any] = {}  # video_id -> query_engine
        
        # Initialize ChromaDB early
        try:
            print("🔄 Initializing ChromaDB...")
            chroma_path = self.storage_path / "chroma"
            chroma_path.mkdir(exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            self.chroma_client.heartbeat()
            print("✅ ChromaDB initialized")
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            self.chroma_client = None
        
        # Initialize Whisper model early
        try:
            print("🔄 Initializing Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model initialized")
        except Exception as e:
            print(f"❌ Whisper model initialization failed: {e}")
            self.whisper_model = None
        
        # Check for FFmpeg
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is required but not found. Please install FFmpeg first.")
        
        self.processing_tasks: Dict[str, ProcessingTask] = {}
        self.task_semaphore = asyncio.Semaphore(3)  # Limit concurrent tasks
        self.quality_settings = {
            'high': {'height': 1080, 'fps': 30},
            'medium': {'height': 720, 'fps': 30},
            'low': {'height': 480, 'fps': 24},
            'auto': {'height': 720, 'fps': 30}  # Default
        }
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed"""
        try:
            return shutil.which('ffmpeg') is not None
        except Exception:
            return False

    async def initialize(self):
        """Initialize AI models and dependencies"""
        try:
            # Setup Gemini
            self.gemini_llm = GeminiMultiModal(
                model_name="models/gemini-1.5-flash-latest",
                api_key=self.gemini_api_key,
                temperature=0.1,
                max_tokens=1024
            )
            
            # Setup embeddings
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Configure LlamaIndex globally
            Settings.embed_model = self.embed_model
            Settings.llm = self.gemini_llm
            
            # ChromaDB is now initialized in __init__
            if not self.chroma_client:
                print("❌ ChromaDB not available")
                raise RuntimeError("ChromaDB client not initialized")
            
            # Setup OpenAI client for frame analysis
            if self.openai_api_key:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print("✅ OpenAI client initialized")
            
            # Whisper model is now initialized in __init__
            if not self.whisper_model:
                print("⚠️ Whisper model not available")
            
            print("✅ VideoRAGProcessor initialized")
            
        except Exception as e:
            print(f"❌ VideoRAGProcessor initialization failed: {e}")
            # Cleanup any partially initialized components
            await self.cleanup()
            raise
    
    async def process_video(self, youtube_url: str, options: Optional[Dict] = None, batch_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a YouTube video with hybrid approach and caching"""
        if not self.chroma_client:
            raise RuntimeError("ChromaDB client not initialized")
            
        try:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")

            # Parse options
            opts = options or {}
            quality = opts.get('quality', 'auto')
            parallel = opts.get('parallel_processing', True)
            preload = opts.get('preload_segments', True)
            max_tasks = opts.get('max_parallel_tasks', 3)
            include_frames = opts.get('extract_frames', True)
            max_duration = opts.get('max_duration', 300)

            # Update processing status
            await self.cache.set_processing_status(str(video_id), "processing")
            if batch_id:
                await self.cache.set_batch_status(batch_id, {
                    "status": "processing",
                    "videos": {video_id: "processing"}
                })

            # Create processing tasks
            tasks = []
            if parallel:
                # Create parallel tasks
                transcript_task = ProcessingTask(video_id, "transcript")
                visual_task = ProcessingTask(video_id, "visual") if include_frames else None
                metadata_task = ProcessingTask(video_id, "metadata")

                tasks = [t for t in [transcript_task, visual_task, metadata_task] if t]
                self.processing_tasks[video_id] = {t.task_type: t for t in tasks}

                # Process tasks in parallel with semaphore
                async with asyncio.TaskGroup() as tg:
                    for task in tasks:
                        tg.create_task(self._process_task(task, quality, max_duration))

            else:
                # Sequential processing
                await self._process_video_sequential(video_id, quality, include_frames, max_duration)

            # Download video with quality settings
            video_data = await self._download_video_for_frames(youtube_url, quality, max_duration)
            if not video_data:
                raise ValueError("Failed to download video")

            # Get processing results
            segments = []
            if parallel:
                for task in tasks:
                    if task.result:
                        segments.extend(task.result)
            else:
                segments = await self._get_processed_segments(video_id)

            # Create section breakdown
            sections = self._create_section_breakdown(segments)

            # Create RAG index
            query_engine = await self._create_hybrid_index(video_id, segments)
            self.active_indices[video_id] = query_engine

            # Save metadata
            metadata = {
                "video_id": video_id,
                "youtube_url": youtube_url,
                "video_info": video_data,
                "sections": sections,
                "segments": [s.to_dict() for s in segments],
                "processed_at": datetime.now().isoformat(),
                "status": "completed",
                "quality": quality
            }

            # Save to storage and cache
            await self._save_metadata(video_id, metadata)
            await self.cache.set_processing_status(str(video_id), "completed")

            if batch_id:
                await self.cache.set_batch_status(batch_id, {
                    "status": "completed",
                    "videos": {video_id: "completed"}
                })

            # Start preloading if enabled
            if preload:
                asyncio.create_task(self._preload_segments(video_id, segments))

            # Broadcast completion
            await manager.broadcast_to_video(video_id, {
                "type": "processing_complete",
                "video_id": video_id,
                "metadata": metadata
            })

            return metadata

        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            if video_id:
                await self.cache.set_processing_status(str(video_id), "failed")
            if batch_id:
                await self.cache.set_batch_status(batch_id, {
                    "status": "failed",
                    "videos": {video_id: "failed" if video_id else "unknown"},
                    "error": str(e)
                })
            raise
    
    async def search_video(self, video_id: str, query: str, search_type: str = "hybrid", conversation_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Search within a processed video with streaming response"""
        print(f"🔍 Searching video {video_id} for: {query} (type: {search_type})")
        
        # First check if video exists in storage
        metadata = await self.get_video_info(video_id)
        if not metadata:
            print(f"❌ Video {video_id} not found in storage")
            raise ValueError(f"Video {video_id} not found or not processed")
        
        # Check if index exists in memory
        if video_id not in self.active_indices:
            print(f"🔄 Loading index for video {video_id}...")
            # Try to load from storage
            if not await self._load_video_index(video_id):
                print(f"❌ Failed to load index for video {video_id}")
                raise ValueError(f"Video {video_id} index not found")
            print(f"✅ Index loaded for video {video_id}")
        
        query_engine = self.active_indices[video_id]
        if not query_engine:
            print(f"❌ No query engine found for video {video_id}")
            raise ValueError(f"Video {video_id} search engine not initialized")
        
        try:
            print(f"🔄 Executing search query: {query}")
            # Execute search with streaming
            response = await query_engine.aquery(query)
            print("✅ Search query executed")
            print(f"📝 Raw response: {response}")
            
            # Extract sources with enhanced metadata
            sources = []
            if hasattr(response, 'source_nodes'):
                print(f"📚 Found {len(response.source_nodes)} source nodes")
                for i, node in enumerate(response.source_nodes[:3]):
                    print(f"\nSource node {i+1}:")
                    print(f"Text: {node.text[:200]}...")
                    print(f"Metadata: {node.metadata}")
                    
                    if not node.metadata:
                        print("⚠️ Node has no metadata, skipping")
                        continue
                        
                    metadata = node.metadata
                    source_type = metadata.get("source_type", "unknown")
                    
                    # Filter based on search type
                    if search_type == "transcript" and source_type == "metadata":
                        print(f"Skipping source of type {source_type} (search type: {search_type})")
                        continue
                    elif search_type == "visual" and source_type != "video_frame":
                        print(f"Skipping source of type {source_type} (search type: {search_type})")
                        continue
                    elif search_type == "hybrid" and source_type not in ["speech_to_text", "video_frame"]:
                        print(f"Skipping source of type {source_type} (search type: {search_type})")
                        continue
                    
                    # Format timestamp
                    start_time = float(metadata.get("start_time", 0))
                    end_time = float(metadata.get("end_time", 0))
                    start_fmt = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    end_fmt = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                    
                    # Create source info
                    source_info = {
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "timestamp": start_time,
                        "end_time": end_time,
                        "youtube_link": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s",
                        "confidence": float(metadata.get("confidence", 1.0)),
                        "source_type": source_type,
                        "timestamp_formatted": f"{start_fmt} - {end_fmt}"
                    }
                    sources.append(source_info)
                    print(f"Processed source: {source_info}")
            else:
                print("⚠️ No source nodes found in response")
            
            # Store in chat history if conversation_id provided
            if conversation_id:
                message = {
                    "query": query,
                    "response": str(response),
                    "sources": sources,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.cache.append_chat_message(video_id, conversation_id, message)
            
            # Stream response
            result = {
                "type": "search_result",
                "query": query,
                "answer": str(response),
                "sources": sources,
                "video_id": video_id,
                "search_type": search_type
            }
            print(f"📤 Sending search result: {result}")
            yield result
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            error_result = {
                "type": "error",
                "error": str(e)
            }
            print(f"📤 Sending error result: {error_result}")
            yield error_result
            raise
    
    async def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get video metadata"""
        metadata_file = self.storage_path / f"{video_id}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    async def list_processed_videos(self) -> List[Dict]:
        """List all processed videos"""
        videos = []
        for metadata_file in self.storage_path.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    videos.append(json.load(f))
            except:
                continue
        return videos
    
    def _create_transcript_segments(self, transcript: List[Dict[str, Any]], video_info: Dict[str, Any]) -> List[VideoSegment]:
        """Create segments from transcript"""
        segments = []
        current_chunk: Dict[str, Any] = {
            "start_time": 0.0,
            "text": "",
            "end_time": 0.0
        }
        
        for entry in transcript:
            # 30-second chunks
            if float(entry['start']) - float(current_chunk['start_time']) > 30:
                if current_chunk['text'].strip():
                    segment = VideoSegment(
                        start_time=float(current_chunk['start_time']),
                        end_time=float(current_chunk['end_time']),
                        content_type=ContentType.TRANSCRIPT,
                        content=current_chunk['text'].strip(),
                        metadata={
                            "source": "youtube_transcript",
                            "confidence": 1.0
                        }
                    )
                    segments.append(segment)
                
                current_chunk = {
                    "start_time": float(entry['start']),
                    "text": "",
                    "end_time": float(entry['start'])
                }
            
            current_chunk['text'] += " " + entry['text'].strip()
            current_chunk['end_time'] = float(entry['start']) + float(entry.get('duration', 3))
        
        # Add final chunk
        if current_chunk['text'].strip():
            segment = VideoSegment(
                start_time=float(current_chunk['start_time']),
                end_time=float(current_chunk['end_time']),
                content_type=ContentType.TRANSCRIPT,
                content=current_chunk['text'].strip(),
                metadata={
                    "source": "youtube_transcript",
                    "confidence": 1.0
                }
            )
            segments.append(segment)
        
        return segments

    async def _process_speech_to_text(self, video_path: str) -> List[VideoSegment]:
        """Process video using Whisper for speech-to-text"""
        if not self.whisper_model:
            print("❌ Whisper model not initialized")
            return []
            
        try:
            print("🎤 Starting speech-to-text processing...")
            # Transcribe video
            result = self.whisper_model.transcribe(video_path)
            print("✅ Speech-to-text processing completed")
            
            segments = []
            for segment in result['segments']:
                video_segment = VideoSegment(
                    start_time=segment['start'],
                    end_time=segment['end'],
                    content_type=ContentType.SPEECH_TO_TEXT,
                    content=segment['text'],
                    metadata={
                        "source": "whisper",
                        "confidence": segment.get('confidence', 0.8)
                    },
                    confidence=segment.get('confidence', 0.8)
                )
                segments.append(video_segment)
            
            return segments
        except Exception as e:
            print(f"❌ Speech-to-text failed: {e}")
            return []

    def _create_visual_segments(self, frame_descriptions: List[Dict]) -> List[VideoSegment]:
        """Create segments from visual analysis"""
        segments = []
        for frame in frame_descriptions:
            segment = VideoSegment(
                start_time=frame['timestamp'],
                end_time=frame['timestamp'] + 30,  # Assume 30-second segments
                content_type=ContentType.VISUAL,
                content=frame['description'],
                metadata={
                    "source": "visual_analysis",
                    "frame_path": frame['frame_path'],
                    "frame_number": frame['frame_number']
                }
            )
            segments.append(segment)
        return segments

    def _create_metadata_segment(self, video_info: Dict) -> VideoSegment:
        """Create segment from video metadata"""
        return VideoSegment(
            start_time=0,
            end_time=video_info['duration'],
            content_type=ContentType.METADATA,
            content=f"Title: {video_info['title']}\nDuration: {video_info['duration']} seconds",
            metadata={
                "source": "video_metadata",
                "title": video_info['title'],
                "duration": video_info['duration']
            }
        )

    def _create_section_breakdown(self, segments: List[VideoSegment]) -> List[Dict[str, Any]]:
        """Create section breakdown with timestamps"""
        # Group segments into sections based on content type and timing
        sections: List[Dict[str, Any]] = []
        current_section: Optional[Dict[str, Any]] = None
        
        for segment in sorted(segments, key=lambda x: x.start_time):
            if not current_section or segment.start_time - float(current_section['end_time']) > 60:
                # Start new section
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'content_type': segment.content_type.value,
                    'title': self._generate_section_title(segment),
                    'content': segment.content[:200] + "...",
                    'confidence': segment.confidence
                }
            else:
                # Extend current section
                current_section['end_time'] = segment.end_time
                current_section['content'] = str(current_section['content']) + f"\n{segment.content}"
                current_section['confidence'] = min(float(current_section['confidence']), segment.confidence)
        
        if current_section:
            sections.append(current_section)
        
        return sections

    def _generate_section_title(self, segment: VideoSegment) -> str:
        """Generate a title for a section based on its content"""
        if segment.content_type == ContentType.TRANSCRIPT:
            # Use first sentence or first 50 chars
            title = segment.content.split('.')[0][:50]
        elif segment.content_type == ContentType.VISUAL:
            # Use key visual elements
            title = f"Visual: {segment.content.split(',')[0]}"
        else:
            # Use content type and timestamp
            start_fmt = f"{int(segment.start_time//60):02d}:{int(segment.start_time%60):02d}"
            title = f"{segment.content_type.value.title()} at {start_fmt}"
        
        return title

    async def _create_hybrid_index(self, video_id: str, segments: List[VideoSegment]) -> Any:
        """Create a hybrid RAG index from all segments"""
        if not self.chroma_client:
            raise RuntimeError("ChromaDB client not initialized")
        
        print(f"🔄 Creating hybrid index for video {video_id}...")
        
        # Create documents from segments
        documents = []
        for segment in segments:
            # Format timestamp
            start_fmt = f"{int(segment.start_time//60):02d}:{int(segment.start_time%60):02d}"
            end_fmt = f"{int(segment.end_time//60):02d}:{int(segment.end_time%60):02d}"
            
            # Create document with metadata
            doc = Document(
                text=segment.content,
                metadata={
                    "source_type": segment.content_type.value,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "timestamp_range": f"{start_fmt} - {end_fmt}",
                    "confidence": segment.confidence,
                    **segment.metadata
                }
            )
            documents.append(doc)
        
        # Create vector store
        collection_name = f"video_{video_id}"
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            
            # Create new collection
            chroma_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"video_id": video_id}
            )
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Create query engine
            query_engine = index.as_query_engine(
                llm=self.gemini_llm,
                similarity_top_k=5,
                response_mode="tree_summarize"
            )
            
            return query_engine
            
        except Exception as e:
            print(f"❌ Failed to create index for video {video_id}: {e}")
            # Cleanup on failure
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            raise
    
    async def _download_video_for_frames(self, youtube_url: str, quality: str = 'auto', max_duration: int = 300) -> Optional[Dict]:
        """Download video with quality settings"""
        settings = self.quality_settings[quality]
        ydl_opts = {
            'format': f'best[height<={settings["height"]}]',
            'outtmpl': str(self.storage_path / '%(id)s.%(ext)s'),
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                
                # Check duration
                duration = info.get('duration', 0)
                if duration > max_duration:
                    print(f"⚠️ Video duration ({duration}s) exceeds limit ({max_duration}s)")
                    return None
                
                video_path = ydl.prepare_filename(info)
                return {
                    'video_path': video_path,
                    'title': info.get('title', 'Unknown'),
                    'duration': duration,
                    'quality': quality,
                    'height': settings['height'],
                    'fps': settings['fps']
                }
        except Exception as e:
            print(f"❌ Video download failed: {e}")
            return None
    
    async def _extract_frames_for_analysis(self, video_path: str, num_frames: int = 10) -> List[str]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        frame_paths = []
        interval = duration / num_frames
        
        for i in range(num_frames):
            timestamp = i * interval
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frame_filename = f"frame_{i}_{int(timestamp)}.jpg"
                frame_path = self.storage_path / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
        
        cap.release()
        return frame_paths
    
    async def _analyze_frames_with_openai(self, frame_paths: List[str], video_id: str) -> List[Dict]:
        """Analyze frames using OpenAI Vision"""
        if not self.openai_client:
            return []
        
        frame_descriptions = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Read and encode frame
                with open(frame_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Calculate timestamp (assuming equal intervals)
                timestamp = i * 30  # 30 second intervals
                timestamp_formatted = f"{timestamp//60:02d}:{timestamp%60:02d}"
                
                # Analyze with OpenAI Vision
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Analyze this video frame at timestamp {timestamp_formatted}. Describe what you see in detail - people, objects, text, colors, activities, settings, etc. Be specific and comprehensive."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                description = response.choices[0].message.content
                
                frame_descriptions.append({
                    'timestamp': timestamp,
                    'timestamp_formatted': timestamp_formatted,
                    'description': description,
                    'frame_path': frame_path,
                    'frame_number': i
                })
                
                print(f"🔍 Analyzed frame {i+1}/{len(frame_paths)} at {timestamp_formatted}")
                
            except Exception as e:
                print(f"❌ Frame analysis failed for frame {i}: {e}")
                continue
        
        return frame_descriptions
    
    async def _load_video_index(self, video_id: str) -> bool:
        """Load video index from storage"""
        try:
            print(f"🔄 Attempting to load index for video {video_id}...")
            
            # Check if metadata exists
            metadata = await self.get_video_info(video_id)
            if not metadata:
                print(f"❌ No metadata found for video {video_id}")
                return False
            
            # Check if segments exist
            segments = metadata.get("segments", [])
            if not segments:
                print(f"❌ No segments found for video {video_id}")
                return False
            
            # Convert segments back to VideoSegment objects
            video_segments = []
            for segment in segments:
                video_segment = VideoSegment(
                    start_time=segment["start_time"],
                    end_time=segment["end_time"],
                    content_type=ContentType(segment["content_type"]),
                    content=segment["content"],
                    metadata=segment["metadata"],
                    confidence=segment["confidence"]
                )
                video_segments.append(video_segment)
            
            # Create new index
            print(f"🔄 Creating new index for video {video_id}...")
            query_engine = await self._create_hybrid_index(video_id, video_segments)
            
            # Store in active indices
            self.active_indices[video_id] = query_engine
            print(f"✅ Index loaded for video {video_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load index for video {video_id}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.chroma_client:
                # Cleanup any temporary collections
                for collection in self.chroma_client.list_collections():
                    if collection.name.startswith("video_"):
                        try:
                            self.chroma_client.delete_collection(collection.name)
                        except:
                            pass
            self.active_indices.clear()
            print("✅ VideoRAGProcessor cleaned up")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

    async def process_batch(self, youtube_urls: List[str], include_frames: bool = True) -> str:
        """Process multiple videos in batch"""
        batch_id = str(uuid.uuid4())
        
        # Initialize batch status
        await self.cache.set_batch_status(batch_id, {
            "status": "pending",
            "total": len(youtube_urls),
            "completed": 0,
            "failed": 0,
            "videos": {url: "pending" for url in youtube_urls}
        })
        
        # Process videos concurrently
        tasks = []
        for url in youtube_urls:
            task = asyncio.create_task(
                self.process_video(url, include_frames, batch_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update final batch status
        completed = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - completed
        
        await self.cache.set_batch_status(batch_id, {
            "status": "completed",
            "total": len(youtube_urls),
            "completed": completed,
            "failed": failed,
            "videos": {
                url: "completed" if not isinstance(r, Exception) else "failed"
                for url, r in zip(youtube_urls, results)
            }
        })
        
        return batch_id

    async def _process_task(self, task: ProcessingTask, quality: str, max_duration: int):
        """Process a single task with semaphore"""
        async with self.task_semaphore:
            try:
                task.status = "processing"
                if task.task_type == "transcript":
                    task.result = await self._process_transcript_task(task.video_id, quality)
                elif task.task_type == "visual":
                    task.result = await self._process_visual_task(task.video_id, quality)
                elif task.task_type == "metadata":
                    task.result = await self._process_metadata_task(task.video_id)
                task.status = "completed"
                task.progress = 1.0
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                raise

    async def _process_video_sequential(self, video_id: str, quality: str, include_frames: bool, max_duration: int):
        """Process video sequentially"""
        try:
            # Process transcript
            transcript_segments = await self._process_transcript_task(video_id, quality)
            
            # Process visual content if enabled
            visual_segments = []
            if include_frames:
                visual_segments = await self._process_visual_task(video_id, quality)
            
            # Process metadata
            metadata_segment = await self._process_metadata_task(video_id)
            
            return transcript_segments + visual_segments + [metadata_segment]
        except Exception as e:
            print(f"❌ Sequential processing failed: {e}")
            raise

    async def _process_transcript_task(self, video_id: str, quality: str) -> List[VideoSegment]:
        """Process video transcript"""
        try:
            # Try YouTube transcript first
            transcript = get_youtube_transcript(video_id)
            if transcript:
                return self._create_transcript_segments(transcript, {"video_id": video_id})
            
            # Fallback to speech-to-text
            if self.whisper_model:
                video_path = self.storage_path / f"{video_id}.mp4"
                if video_path.exists():
                    return await self._process_speech_to_text(str(video_path))
            
            raise ValueError("No transcript available")
        except Exception as e:
            print(f"❌ Transcript processing failed: {e}")
            raise

    async def _process_visual_task(self, video_id: str, quality: str) -> List[VideoSegment]:
        """Process visual content"""
        try:
            video_path = self.storage_path / f"{video_id}.mp4"
            if not video_path.exists():
                raise ValueError("Video file not found")

            frame_paths = await self._extract_frames_for_analysis(str(video_path), quality)
            if not frame_paths:
                return []

            frame_descriptions = await self._analyze_frames_with_openai(frame_paths, video_id)
            return self._create_visual_segments(frame_descriptions)
        except Exception as e:
            print(f"❌ Visual processing failed: {e}")
            raise

    async def _process_metadata_task(self, video_id: str) -> VideoSegment:
        """Process video metadata"""
        try:
            video_info = await self.get_video_info(video_id)
            if not video_info:
                raise ValueError("Video info not found")
            return self._create_metadata_segment(video_info)
        except Exception as e:
            print(f"❌ Metadata processing failed: {e}")
            raise

    async def _preload_segments(self, video_id: str, segments: List[VideoSegment]):
        """Preload video segments in the background"""
        try:
            # Group segments by time ranges
            time_ranges = []
            current_range = {"start": segments[0].start_time, "segments": [segments[0]]}
            
            for segment in segments[1:]:
                if segment.start_time - current_range["start"] > 30:  # 30-second gap
                    time_ranges.append(current_range)
                    current_range = {"start": segment.start_time, "segments": [segment]}
                else:
                    current_range["segments"].append(segment)
            
            time_ranges.append(current_range)

            # Preload each range
            for i, range_data in enumerate(time_ranges):
                try:
                    # Store segments in cache
                    await self.cache.set_video_segments(
                        video_id,
                        range_data["start"],
                        [s.to_dict() for s in range_data["segments"]]
                    )
                    
                    # Update progress
                    progress = (i + 1) / len(time_ranges)
                    await manager.broadcast_to_video(video_id, {
                        "type": "preload_progress",
                        "video_id": video_id,
                        "progress": progress,
                        "current_time": range_data["start"]
                    })
                    
                except Exception as e:
                    print(f"❌ Failed to preload segment range: {e}")
                    continue

        except Exception as e:
            print(f"❌ Segment preloading failed: {e}")

    async def _save_metadata(self, video_id: str, metadata: Dict[str, Any]):
        """Save video metadata to storage and cache"""
        try:
            # Save to file
            metadata_file = self.storage_path / f"{video_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Cache metadata
            await self.cache.set_video_metadata(video_id, metadata)
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
            raise