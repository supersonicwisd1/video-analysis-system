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
from app.api.models.responses import VideoSection as APIVideoSection
from app.api.models.responses import VideoSection

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

@dataclass
class VideoSection:
    """Video section with JSON serialization support"""
    id: str
    title: str
    start_time: float
    end_time: float
    description: Optional[str]
    type: str
    confidence: float
    metadata: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
            "type": self.type,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }

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
        self.processing_tasks: Dict[str, Dict[str, ProcessingTask]] = {}  # video_id -> {task_type -> task}
        
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
            
        video_id = None
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

            # Download video first
            print(f"📥 Downloading video {video_id}...")
            video_data = await self._download_video_for_frames(youtube_url, quality, max_duration)
            if not video_data:
                raise ValueError("Failed to download video")
            print(f"✅ Video downloaded: {video_data['title']}")

            # Validate video data
            if not video_data.get('title') or not video_data.get('duration'):
                raise ValueError("Invalid video data: missing title or duration")

            # Save video info immediately after download
            initial_metadata = {
                "video_id": video_id,
                "youtube_url": youtube_url,
                "video_info": video_data,
                "status": "downloaded",
                "quality": quality,
                "processed_at": datetime.now().isoformat(),
                "title": video_data['title'],
                "duration": float(video_data['duration'])
            }
            await self._save_metadata(video_id, initial_metadata)

            # Process metadata first
            print("🔄 Processing metadata...")
            try:
                metadata_segment = await self._process_metadata_task(video_id)
                if not metadata_segment:
                    raise ValueError("Failed to process metadata")
            except Exception as e:
                print(f"❌ Metadata processing failed: {e}")
                # Save error state
                error_metadata = {
                    "video_id": video_id,
                    "youtube_url": youtube_url,
                    "video_info": video_data,
                    "status": "failed",
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                    "title": video_data['title'],
                    "duration": float(video_data['duration'])
                }
                await self._save_metadata(video_id, error_metadata)
                await self.cache.set_processing_status(str(video_id), "failed")
                raise

            # Create processing tasks for transcript and visual
            tasks = []
            if parallel:
                # Create parallel tasks for transcript and visual only
                transcript_task = ProcessingTask(video_id, "transcript")
                visual_task = ProcessingTask(video_id, "visual") if include_frames else None

                tasks = [t for t in [transcript_task, visual_task] if t]
                self.processing_tasks[video_id] = {t.task_type: t for t in tasks}

                # Process tasks in parallel with semaphore
                try:
                    async with asyncio.TaskGroup() as tg:
                        for task in tasks:
                            tg.create_task(self._process_task(task, quality, max_duration))
                except* Exception as e:
                    print(f"❌ Task group failed: {e}")
                    # Save error state but continue with any successful tasks
                    error_metadata = {
                        "video_id": video_id,
                        "youtube_url": youtube_url,
                        "video_info": video_data,
                        "status": "failed",
                        "error": f"Task group failed: {str(e)}",
                        "processed_at": datetime.now().isoformat(),
                        "title": video_data['title'],
                        "duration": float(video_data['duration'])
                    }
                    await self._save_metadata(video_id, error_metadata)
                    await self.cache.set_processing_status(str(video_id), "failed")
                    raise

            else:
                # Sequential processing for transcript and visual
                try:
                    transcript_segments = await self._process_transcript_task(video_id, quality)
                    visual_segments = []
                    if include_frames:
                        visual_segments = await self._process_visual_task(video_id, quality)
                    segments = transcript_segments + visual_segments
                except Exception as e:
                    print(f"❌ Sequential processing failed: {e}")
                    error_metadata = {
                        "video_id": video_id,
                        "youtube_url": youtube_url,
                        "video_info": video_data,
                        "status": "failed",
                        "error": str(e),
                        "processed_at": datetime.now().isoformat(),
                        "title": video_data['title'],
                        "duration": float(video_data['duration'])
                    }
                    await self._save_metadata(video_id, error_metadata)
                    await self.cache.set_processing_status(str(video_id), "failed")
                    raise

            # Get processing results
            segments = [metadata_segment]  # Start with metadata segment
            if parallel:
                for task in tasks:
                    if task.result:
                        if isinstance(task.result, list):
                            segments.extend(task.result)
                        else:
                            segments.append(task.result)
            else:
                segments.extend(segments)  # Add segments from sequential processing

            # Create section breakdown
            sections = self._create_section_breakdown(segments)

            # Create RAG index
            query_engine = await self._create_hybrid_index(video_id, segments)
            self.active_indices[video_id] = query_engine

            # Update metadata with processing results
            final_metadata = {
                "video_id": video_id,
                "youtube_url": youtube_url,
                "video_info": video_data,
                "sections": sections,
                "segments": [s.to_dict() for s in segments],
                "processed_at": datetime.now().isoformat(),
                "status": "completed",
                "quality": quality,
                "title": video_data['title'],
                "duration": float(video_data['duration'])
            }

            # Save to storage and cache
            await self._save_metadata(video_id, final_metadata)
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
                "metadata": final_metadata
            })

            return final_metadata

        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            if video_id:
                await self.cache.set_processing_status(str(video_id), "failed")
                # Save error metadata
                error_metadata = {
                    "video_id": video_id,
                    "youtube_url": youtube_url,
                    "status": "failed",
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                }
                # Try to get video info if available
                try:
                    video_info = await self.get_video_info(video_id)
                    if video_info and 'video_info' in video_info:
                        error_metadata.update({
                            "video_info": video_info['video_info'],
                            "title": video_info['video_info'].get('title', 'Unknown'),
                            "duration": float(video_info['video_info'].get('duration', 0))
                        })
                except:
                    pass
                await self._save_metadata(video_id, error_metadata)
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
        
        try:
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
            
            query_engine = self.active_indices.get(video_id)
            if not query_engine:
                print(f"❌ No query engine found for video {video_id}")
                raise ValueError(f"Video {video_id} search engine not initialized")
            
            try:
                print(f"🔄 Executing search query: {query}")
                # Execute search with streaming
                response = await query_engine.aquery(query)
                print("✅ Search query executed")
                
                # Extract sources with enhanced metadata
                sources = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"📚 Found {len(response.source_nodes)} source nodes")
                    for i, node in enumerate(response.source_nodes[:3]):
                        if not node or not node.metadata:
                            continue
                            
                        print(f"\nSource node {i+1}:")
                        print(f"Text: {node.text[:200]}...")
                        print(f"Metadata: {node.metadata}")
                        
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
                        
                        try:
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
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ Error processing source node {i+1}: {e}")
                            continue
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
                print(f"❌ Search query failed: {e}")
                error_result = {
                    "type": "error",
                    "error": str(e)
                }
                print(f"📤 Sending error result: {error_result}")
                yield error_result
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            error_result = {
                "type": "error",
                "error": str(e)
            }
            print(f"📤 Sending error result: {error_result}")
            yield error_result
    
    async def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get video metadata with validation"""
        try:
            metadata_file = self.storage_path / f"{video_id}_metadata.json"
            if not metadata_file.exists():
                print(f"❌ Metadata file not found for video {video_id}")
                return None

            # Read and validate JSON
            try:
                with open(metadata_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"❌ Empty metadata file for video {video_id}")
                        return None
                    
                    # Try to parse JSON
                    try:
                        metadata = json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON in metadata file for video {video_id}: {e}")
                        # Try to fix truncated JSON
                        if content.endswith(','):
                            content = content[:-1]
                        if not content.endswith('}'):
                            content += '}'
                        try:
                            metadata = json.loads(content)
                        except:
                            print(f"❌ Could not fix truncated JSON for video {video_id}")
                            return None
            except Exception as e:
                print(f"❌ Error reading metadata file for video {video_id}: {e}")
                return None

            # Validate required fields
            required_fields = ['video_id', 'youtube_url', 'video_info']
            if not all(field in metadata for field in required_fields):
                print(f"❌ Missing required fields in metadata for video {video_id}")
                return None

            # Validate video_info
            video_info = metadata.get('video_info', {})
            if not isinstance(video_info, dict):
                print(f"❌ Invalid video_info in metadata for video {video_id}")
                return None

            # Ensure numeric fields are proper types
            if 'duration' in video_info:
                try:
                    video_info['duration'] = float(video_info['duration'])
                except (ValueError, TypeError):
                    print(f"❌ Invalid duration in metadata for video {video_id}")
                    return None

            return metadata

        except Exception as e:
            print(f"❌ Error getting video info for {video_id}: {e}")
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

    def _create_section_breakdown(self, segments: List[VideoSegment]) -> List[VideoSection]:
        """Create video sections from segments using AI analysis"""
        try:
            if not segments:
                return []

            # Get video duration from first segment's metadata
            video_id = segments[0].metadata.get("video_id")
            if not video_id:
                print("⚠️ No video ID found in segments")
                return []

            # First try to get YouTube chapters if available
            youtube_sections = self._get_youtube_chapters(video_id)
            if youtube_sections:
                return youtube_sections

            # Group segments into potential sections
            section_candidates = self._group_segments_into_sections(segments)
            
            # Use AI to analyze and create sections
            sections = []
            for i, candidate in enumerate(section_candidates):
                # Analyze content to generate title and description
                title, description = self._analyze_section_content(candidate)
                
                section = VideoSection(
                    id=str(uuid.uuid4()),
                    title=title,
                    start_time=float(candidate["start_time"]),
                    end_time=float(candidate["end_time"]),
                    description=description or "",  # Ensure description is never None
                    type="auto",
                    confidence=float(candidate["confidence"]),
                    metadata={
                        "segment_count": len(candidate["segments"]),
                        "content_types": list(set(s.content_type.value for s in candidate["segments"]))
                    }
                )
                sections.append(section)

            # Merge overlapping or very close sections
            sections = self._merge_close_sections(sections)
            
            # Get total duration from video info
            video_info = self.get_video_info(video_id)
            total_duration = float(video_info.get("video_info", {}).get("duration", 0)) if video_info else 0
            
            # Ensure sections cover the entire video
            sections = self._fill_section_gaps(sections, total_duration)
            
            return sections

        except Exception as e:
            print(f"❌ Section creation failed: {e}")
            return []

    def _get_youtube_chapters(self, video_id: str) -> Optional[List[VideoSection]]:
        """Extract chapters from YouTube video metadata"""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(video_id, download=False)
                chapters = info.get('chapters', [])
                
                if not chapters:
                    return None
                
                return [
                    VideoSection(
                        id=str(uuid.uuid4()),
                        title=chapter['title'],
                        start_time=chapter['start_time'],
                        end_time=chapter['end_time'],
                        description=chapter.get('description'),
                        type="youtube",
                        confidence=1.0,
                        metadata={"chapter_index": i}
                    )
                    for i, chapter in enumerate(chapters)
                ]
        except Exception as e:
            print(f"❌ Failed to get YouTube chapters: {e}")
            return None

    def _group_segments_into_sections(self, segments: List[VideoSegment]) -> List[Dict[str, Any]]:
        """Group segments into potential sections based on content and timing"""
        if not segments:
            return []

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        candidates = []
        current_candidate = {
            "start_time": sorted_segments[0].start_time,
            "end_time": sorted_segments[0].end_time,
            "segments": [sorted_segments[0]],
            "confidence": 1.0
        }

        for segment in sorted_segments[1:]:
            # Check if segment should be part of current section
            time_gap = segment.start_time - current_candidate["end_time"]
            content_similarity = self._calculate_content_similarity(
                current_candidate["segments"][-1],
                segment
            )

            if time_gap < 30 and content_similarity > 0.5:  # Adjust thresholds as needed
                # Extend current section
                current_candidate["end_time"] = segment.end_time
                current_candidate["segments"].append(segment)
                current_candidate["confidence"] = min(
                    current_candidate["confidence"],
                    content_similarity
                )
            else:
                # Start new section
                candidates.append(current_candidate)
                current_candidate = {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "segments": [segment],
                    "confidence": 1.0
                }

        # Add last candidate
        candidates.append(current_candidate)
        return candidates

    def _calculate_content_similarity(self, seg1: VideoSegment, seg2: VideoSegment) -> float:
        """Calculate similarity between two segments"""
        # For now, use a simple approach based on content type
        if seg1.content_type != seg2.content_type:
            return 0.3  # Different content types are less likely to be in same section
        
        # TODO: Implement more sophisticated similarity calculation
        # Could use embeddings or other NLP techniques
        return 0.7  # Default similarity for same content type

    def _analyze_section_content(self, candidate: Dict[str, Any]) -> tuple[str, str]:
        """Use AI to analyze section content and generate title and description"""
        try:
            # Combine segment content
            content = " ".join(s.content for s in candidate["segments"])
            
            # Use OpenAI to generate title and description
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, descriptive titles and summaries for video sections."},
                    {"role": "user", "content": f"Create a title and brief description for this video section:\n\n{content}"}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            # Parse response
            result = response.choices[0].message.content.split("\n", 1)
            title = result[0].strip()
            description = result[1].strip() if len(result) > 1 else None
            
            return title, description

        except Exception as e:
            print(f"❌ Section analysis failed: {e}")
            # Fallback to simple title
            return f"Section {candidate['start_time']:.0f}s", None

    def _merge_close_sections(self, sections: List[VideoSection]) -> List[VideoSection]:
        """Merge sections that are too close together or overlapping"""
        if not sections:
            return []

        merged = []
        current = sections[0]

        for next_section in sections[1:]:
            # Check if sections should be merged
            if (next_section.start_time - current.end_time < 10 or  # Less than 10s gap
                next_section.start_time < current.end_time):  # Overlapping
                
                # Merge sections
                current = VideoSection(
                    id=current.id,
                    title=f"{current.title} / {next_section.title}",
                    start_time=current.start_time,
                    end_time=max(current.end_time, next_section.end_time),
                    description=f"{current.description or ''}\n{next_section.description or ''}".strip() or None,
                    type="auto",
                    confidence=min(current.confidence, next_section.confidence),
                    metadata={
                        **current.metadata,
                        "merged_sections": current.metadata.get("merged_sections", 0) + 1
                    }
                )
            else:
                merged.append(current)
                current = next_section

        merged.append(current)
        return merged

    def _fill_section_gaps(self, sections: List[VideoSection], total_duration: float) -> List[VideoSection]:
        """Ensure sections cover the entire video duration"""
        if not sections:
            return []

        filled = []
        current_time = 0.0

        for section in sections:
            # Add gap section if needed
            if section.start_time - current_time > 30:  # Gap larger than 30s
                filled.append(VideoSection(
                    id=str(uuid.uuid4()),
                    title=f"Gap {current_time:.0f}s - {section.start_time:.0f}s",
                    start_time=current_time,
                    end_time=section.start_time,
                    description="",  # Empty description for gap sections
                    type="gap",
                    confidence=1.0,
                    metadata={"is_gap": True}
                ))
            
            filled.append(section)
            current_time = section.end_time

        # Add final gap if needed
        if total_duration - current_time > 30:
            filled.append(VideoSection(
                id=str(uuid.uuid4()),
                title=f"Gap {current_time:.0f}s - {total_duration:.0f}s",
                start_time=current_time,
                end_time=total_duration,
                description="",  # Empty description for gap sections
                type="gap",
                confidence=1.0,
                metadata={"is_gap": True}
            ))

        return filled

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
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                
                # Check duration
                duration = float(info.get('duration', 0))
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
                    'fps': settings['fps'],
                    'video_id': info.get('id'),
                    'upload_date': info.get('upload_date'),
                    'channel': info.get('channel'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'description': info.get('description'),
                    'thumbnail': info.get('thumbnail'),
                    'has_subtitles': bool(info.get('subtitles') or info.get('automatic_captions'))
                }
        except Exception as e:
            print(f"❌ Video download failed: {e}")
            return None
    
    async def _extract_frames_for_analysis(self, video_path: str, num_frames: int = 10) -> List[str]:
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            if duration == 0:
                raise ValueError("Invalid video duration")

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
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            return []
    
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
        """Load video index from storage with validation"""
        try:
            print(f"🔄 Attempting to load index for video {video_id}...")
            
            # Check if metadata exists and is valid
            metadata = await self.get_video_info(video_id)
            if not metadata:
                print(f"❌ No valid metadata found for video {video_id}")
                return False
            
            # Check if segments exist and are valid
            segments = metadata.get("segments", [])
            if not segments or not isinstance(segments, list):
                print(f"❌ No valid segments found for video {video_id}")
                return False
            
            # Convert segments back to VideoSegment objects with validation
            video_segments = []
            for segment in segments:
                try:
                    if not isinstance(segment, dict):
                        print(f"⚠️ Invalid segment format: {segment}")
                        continue
                        
                    # Validate required fields
                    required_fields = ['start_time', 'end_time', 'content_type', 'content']
                    if not all(field in segment for field in required_fields):
                        print(f"⚠️ Missing required fields in segment: {segment}")
                        continue
                    
                    # Convert numeric fields
                    try:
                        start_time = float(segment['start_time'])
                        end_time = float(segment['end_time'])
                        confidence = float(segment.get('confidence', 1.0))
                    except (ValueError, TypeError):
                        print(f"⚠️ Invalid numeric values in segment: {segment}")
                        continue
                    
                    # Create VideoSegment
                    video_segment = VideoSegment(
                        start_time=start_time,
                        end_time=end_time,
                        content_type=ContentType(segment['content_type']),
                        content=str(segment['content']),
                        metadata=segment.get('metadata', {}),
                        confidence=confidence
                    )
                    video_segments.append(video_segment)
                except Exception as e:
                    print(f"⚠️ Error processing segment: {e}")
                    continue
            
            if not video_segments:
                print(f"❌ No valid segments could be loaded for video {video_id}")
                return False
            
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
                    segments = await self._process_transcript_task(task.video_id, quality)
                    task.result = segments if segments else []
                elif task.task_type == "visual":
                    segments = await self._process_visual_task(task.video_id, quality)
                    task.result = segments if segments else []
                elif task.task_type == "metadata":
                    segment = await self._process_metadata_task(task.video_id)
                    task.result = segment if segment else None
                task.status = "completed"
                task.progress = 1.0
            except Exception as e:
                print(f"❌ Task {task.task_type} failed: {e}")
                task.status = "failed"
                task.error = str(e)
                task.result = [] if task.task_type in ["transcript", "visual"] else None
                # Don't raise to allow other tasks to continue

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

            # Convert quality to number of frames
            num_frames = {
                'high': 20,
                'medium': 15,
                'low': 10,
                'auto': 15
            }.get(quality, 15)

            frame_paths = await self._extract_frames_for_analysis(str(video_path), num_frames)
            if not frame_paths:
                return []

            frame_descriptions = await self._analyze_frames_with_openai(frame_paths, video_id)
            return self._create_visual_segments(frame_descriptions)
        except Exception as e:
            print(f"❌ Visual processing failed: {e}")
            return []  # Return empty list instead of raising to allow other tasks to continue

    async def _process_metadata_task(self, video_id: str) -> VideoSegment:
        """Process video metadata"""
        try:
            video_info = await self.get_video_info(video_id)
            if not video_info or 'video_info' not in video_info:
                raise ValueError("Video info not found")
            
            info = video_info['video_info']
            duration = float(info.get('duration', 0))
            if not duration:
                raise ValueError("Video duration not found")
            
            return self._create_metadata_segment(info)
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

            # Get existing metadata
            existing_metadata = await self.cache.get_video_metadata(video_id) or {}
            
            # Preload each range
            for i, range_data in enumerate(time_ranges):
                try:
                    # Update metadata with new segments
                    updated_metadata = {
                        **existing_metadata,
                        "segments": [s.to_dict() for s in range_data["segments"]],
                        "last_updated": datetime.now().isoformat(),
                        "preload_progress": {
                            "current": i + 1,
                            "total": len(time_ranges),
                            "timestamp": range_data["start"]
                        }
                    }
                    
                    # Store in cache
                    await self.cache.set_video_metadata(video_id, updated_metadata)
                    
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
        """Save video metadata to storage and cache with validation"""
        try:
            # Create a copy of metadata to avoid modifying the original
            metadata_copy = metadata.copy()
            
            # Convert VideoSection objects to dictionaries
            if 'sections' in metadata_copy:
                metadata_copy['sections'] = [
                    s.to_dict() if hasattr(s, 'to_dict') else s 
                    for s in metadata_copy['sections']
                ]
            
            # Convert VideoSegment objects to dictionaries
            if 'segments' in metadata_copy:
                metadata_copy['segments'] = [
                    s.to_dict() if hasattr(s, 'to_dict') else s 
                    for s in metadata_copy['segments']
                ]
            
            # Ensure all values are JSON serializable
            def make_serializable(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                else:
                    return str(obj)
            
            metadata_copy = make_serializable(metadata_copy)
            
            # Validate metadata structure
            if not isinstance(metadata_copy, dict):
                raise ValueError("Metadata must be a dictionary")
            
            required_fields = ['video_id', 'youtube_url', 'video_info']
            if not all(field in metadata_copy for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")
            
            # Validate video_info
            video_info = metadata_copy.get('video_info', {})
            if not isinstance(video_info, dict):
                raise ValueError("video_info must be a dictionary")
            
            # Ensure numeric fields are proper types
            if 'duration' in video_info:
                try:
                    video_info['duration'] = float(video_info['duration'])
                except (ValueError, TypeError):
                    raise ValueError("Invalid duration value")
            
            # Save to file with atomic write
            metadata_file = self.storage_path / f"{video_id}_metadata.json"
            temp_file = metadata_file.with_suffix('.tmp')
            
            try:
                # Write to temporary file first
                with open(temp_file, 'w') as f:
                    json.dump(metadata_copy, f, indent=2, ensure_ascii=False)
                
                # Validate the written JSON
                with open(temp_file, 'r') as f:
                    json.load(f)  # This will raise if JSON is invalid
                
                # If validation passes, rename temp file to actual file
                temp_file.replace(metadata_file)
                print(f"✅ Metadata saved for video {video_id}")
                
                # Cache metadata
                await self.cache.set_video_metadata(video_id, metadata_copy)
                
            except Exception as e:
                # Clean up temp file if something goes wrong
                if temp_file.exists():
                    temp_file.unlink()
                raise ValueError(f"Failed to save metadata: {e}")
            
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
            raise