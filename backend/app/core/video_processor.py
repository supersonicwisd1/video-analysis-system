"""Video RAG Processor - Integration of your LlamaIndex system"""
import os
import json
import tempfile
from typing import List, Dict, Any, Optional
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

class VideoRAGProcessor:
    """Production video RAG processor"""
    
    def __init__(self, gemini_api_key: str, storage_path: str, openai_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Components (initialized in initialize())
        self.gemini_llm = None
        self.embed_model = None
        self.chroma_client = None
        self.openai_client = None
        self.active_indices = {}  # video_id -> query_engine
        
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
            
            # Setup ChromaDB
            self.chroma_client = chromadb.EphemeralClient()

            # Setup OpenAI client for frame analysis (optional)
            if self.openai_api_key:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            
            print("✅ VideoRAGProcessor initialized")
            
        except Exception as e:
            print(f"❌ VideoRAGProcessor initialization failed: {e}")
            raise
    
    async def process_video(self, youtube_url: str, include_frames: bool = False) -> Dict[str, Any]:
        """Process a YouTube video for RAG"""
        try:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            # Check if already processed
            if video_id in self.active_indices:
                return {"video_id": video_id, "status": "already_processed"}
            
            print(f"🎬 Processing video: {video_id}")
            
            # Get transcript
            transcript = get_youtube_transcript(video_id)
            if not transcript:
                raise ValueError("No transcript available")
            
            # Process video (optional frame extraction)
            video_info = {"title": f"Video {video_id}", "duration": 0,  "video_id": video_id}
            frame_paths = []
            frame_descriptions = []
            
            if include_frames:
                try:
                    # Attempt video download and frame extraction
                    video_data = await self._download_video_for_frames(youtube_url, max_duration=300)
                    if video_data:
                        video_info = video_data
                        video_info["video_id"] = video_id
                        frame_paths = await self._extract_frames_for_analysis(
                            video_data['video_path'], num_frames=10
                        )
                        if self.openai_client:
                            frame_descriptions = await self._analyze_frames_with_openai(frame_paths, video_id)
                except Exception as e:
                    print(f"⚠️  Frame extraction skipped: {e}")
            
            # Create documents
            transcript_docs = self._create_transcript_documents(transcript, video_info)
            image_docs = self._create_image_documents(frame_paths, video_info) if frame_paths else []
            
            # Create RAG index
            query_engine = await self._create_rag_index(video_id, transcript_docs, image_docs)
            
            # Store for future queries
            self.active_indices[video_id] = query_engine
            
            # Save metadata
            metadata = {
                "video_id": video_id,
                "youtube_url": youtube_url,
                "video_info": video_info,
                "transcript_segments": len(transcript),
                "frames_extracted": len(frame_paths),
                "frame_descriptions": frame_descriptions,
                "processed_at": datetime.now().isoformat(),
                "status": "completed"
            }
            
            metadata_file = self.storage_path / f"{video_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Video processed: {video_id}")
            return metadata
            
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            raise
    
    async def search_video(self, video_id: str, query: str) -> Dict[str, Any]:
        """Search within a processed video"""
        if video_id not in self.active_indices:
            # Try to load from storage
            if not await self._load_video_index(video_id):
                raise ValueError(f"Video {video_id} not found or not processed")
        
        query_engine = self.active_indices[video_id]
        
        try:
            # Execute search
            response = query_engine.query(query)
            
            # Extract sources/citations with enhanced metadata
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:3]:
                    # Get timestamp information
                    metadata = node.metadata
                    source_type = metadata.get("source_type", "transcript")
                    
                    # Handle different source types
                    if source_type == "transcript":
                        start_time = metadata.get("start_time", 0)
                        end_time = metadata.get("end_time", 0)
                        timestamp_range = metadata.get("timestamp_range", "")
                        youtube_link = metadata.get("youtube_link", "")
                        content = node.text[:200] + "..." if len(node.text) > 200 else node.text
                    else:  # video_frame
                        timestamp = metadata.get("timestamp", 0)
                        timestamp_formatted = metadata.get("timestamp_formatted", "")
                        youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
                        timestamp_range = timestamp_formatted
                        content = metadata.get("description", "")[:200] + "..." if len(metadata.get("description", "")) > 200 else metadata.get("description", "")
                    
                    # Format the content to include timestamp link
                    content_with_timestamp = f"{content}\n\n🔗 [Watch at {timestamp_range}]({youtube_link})"
                    
                    source_info = {
                        "content": content,
                        "metadata": metadata,
                        "timestamp_range": timestamp_range,
                        "youtube_link": youtube_link,
                        "source_type": source_type,
                        "start_time": metadata.get("start_time", timestamp if source_type == "video_frame" else 0),
                        "end_time": metadata.get("end_time", timestamp + 30 if source_type == "video_frame" else 0)
                    }
                    sources.append(source_info)
            
            return {
                "query": query,
                "answer": str(response),
                "sources": sources,
                "video_id": video_id,
                "search_type": "chat"
            }
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
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
    
    def _create_transcript_documents(self, transcript: List[Dict], video_info: Dict) -> List[Document]:
        """Create LlamaIndex documents from transcript"""
        documents = []
        current_chunk = {"start_time": 0, "text": "", "end_time": 0}
        
        for entry in transcript:
            # 30-second chunks
            if entry['start'] - current_chunk['start_time'] > 30:
                if current_chunk['text'].strip():
                    start_fmt = f"{int(current_chunk['start_time']//60):02d}:{int(current_chunk['start_time']%60):02d}"
                    end_fmt = f"{int(current_chunk['end_time']//60):02d}:{int(current_chunk['end_time']%60):02d}"
                    
                    doc = Document(
                        text=current_chunk['text'].strip(),
                        metadata={
                            "source_type": "transcript",
                            "video_title": video_info.get('title', 'Unknown'),
                            "start_time": current_chunk['start_time'],
                            "end_time": current_chunk['end_time'],
                            "timestamp_formatted": start_fmt,
                            "timestamp_range": f"{start_fmt} - {end_fmt}",
                            "youtube_link": f"https://www.youtube.com/watch?v={video_info.get('video_id', '')}&t={int(current_chunk['start_time'])}s"
                        }
                    )
                    documents.append(doc)
                
                current_chunk = {"start_time": entry['start'], "text": "", "end_time": entry['start']}
            
            current_chunk['text'] += " " + entry['text'].strip()
            current_chunk['end_time'] = entry['start'] + entry.get('duration', 3)
        
        # Add final chunk
        if current_chunk['text'].strip():
            start_fmt = f"{int(current_chunk['start_time']//60):02d}:{int(current_chunk['start_time']%60):02d}"
            end_fmt = f"{int(current_chunk['end_time']//60):02d}:{int(current_chunk['end_time']%60):02d}"
            
            doc = Document(
                text=current_chunk['text'].strip(),
                metadata={
                    "source_type": "transcript",
                    "video_title": video_info.get('title', 'Unknown'),
                    "start_time": current_chunk['start_time'],
                    "end_time": current_chunk['end_time'],
                    "timestamp_formatted": start_fmt,
                    "timestamp_range": f"{start_fmt} - {end_fmt}",
                    "youtube_link": f"https://www.youtube.com/watch?v={video_info.get('video_id', '')}&t={int(current_chunk['start_time'])}s"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_image_documents(self, frame_paths: List[str], video_info: Dict) -> List[ImageDocument]:
        """Create image documents from frames"""
        image_documents = []
        for i, frame_path in enumerate(frame_paths):
            timestamp = i * (video_info.get('duration', 300) // len(frame_paths))
            img_doc = ImageDocument(
                image_path=frame_path,
                metadata={
                    "source_type": "video_frame",
                    "video_title": video_info.get('title', 'Unknown'),
                    "frame_number": i,
                    "timestamp": timestamp,
                    "timestamp_formatted": f"{timestamp//60:02d}:{timestamp%60:02d}",
                }
            )
            image_documents.append(img_doc)
        return image_documents
    
    async def _create_rag_index(self, video_id: str, transcript_docs: List[Document], image_docs: List[ImageDocument]):
        """Create RAG index for video"""
        # Setup collection
        collection_name = f"video_{video_id}"
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        chroma_collection = self.chroma_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Combine documents
        all_documents = transcript_docs.copy()
        if image_docs:
            all_documents.extend(image_docs)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            all_documents,
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
    
    def _get_fallback_answer(self, frames: List[Dict], limit: int = 3) -> str:
        """Generate a fallback answer from frame descriptions"""
        try:
            descriptions = []
            for frame in frames[:limit]:
                if isinstance(frame, dict) and 'description' in frame:
                    descriptions.append(f"{frame['description'][:100]}...")
                else:
                    print(f"⚠️ Invalid frame data structure: {frame}")
            return f"Video contains: " + "; ".join(descriptions) if descriptions else "No frame descriptions available."
        except Exception as e:
            print(f"⚠️ Error generating fallback answer: {e}")
            return "Unable to generate frame descriptions."

    async def search_visual_content(self, video_id: str, query: str) -> Dict[str, Any]:
        """Search for visual content in video frames"""
        metadata = await self.get_video_info(video_id)
        if not metadata:
            return {
                "query": query,
                "answer": f"Video {video_id} has not been processed yet. Please process the video first.",
                "sources": [],
                "video_id": video_id,
                "search_type": "visual",
                "status": "not_processed"
            }
        
        frame_descriptions = metadata.get("frame_descriptions", [])
        
        # Handle case where no frame analysis is available
        if not frame_descriptions:
            return {
                "query": query,
                "answer": "Frame analysis is not available for this video. The video was processed without visual frame extraction. Please re-process the video with 'include_frames=true' to enable visual search.",
                "sources": [],
                "video_id": video_id,
                "search_type": "visual",
                "status": "processed_no_frames",
                "video_info": {
                    "title": metadata.get("video_info", {}).get("title", "Unknown"),
                    "duration": metadata.get("video_info", {}).get("duration", 0),
                    "processed_at": metadata.get("processed_at", "")
                }
            }
        
        try:
            # Search through frame descriptions with flexible matching
            matching_frames = []
            query_lower = query.lower()
            query_words = [word.strip() for word in query_lower.split()]
            
            for frame in frame_descriptions:
                description_lower = frame['description'].lower()
                matches = 0
                relevance_score = 0
                
                # More flexible matching
                for word in query_words:
                    # Direct word match
                    if word in description_lower:
                        matches += 1
                        relevance_score += description_lower.count(word)
                    
                    # Synonym matching for common visual terms
                    elif word in ["image", "picture", "photo"] and any(term in description_lower for term in ["shows", "displays", "contains", "features", "visible", "scene"]):
                        matches += 1
                        relevance_score += 1
                    elif word in ["video", "content"] and any(term in description_lower for term in ["screen", "frame", "scene", "background", "foreground"]):
                        matches += 1
                        relevance_score += 1
                    elif word in ["text", "writing"] and any(term in description_lower for term in ["text", "words", "letters", "writing", "title", "caption"]):
                        matches += 2
                        relevance_score += 2
                    elif word in ["person", "people"] and any(term in description_lower for term in ["person", "people", "man", "woman", "individual", "human"]):
                        matches += 2
                        relevance_score += 2
                    elif word in ["code", "programming"] and any(term in description_lower for term in ["code", "programming", "terminal", "computer", "keyboard", "screen"]):
                        matches += 2
                        relevance_score += 2
                
                # Handle very general queries by showing all frames
                general_queries = ["what", "describe", "show", "see", "image", "visual", "content"]
                if any(gen_query in query_lower for gen_query in general_queries) and len(query_words) <= 3:
                    matches = max(matches, 1)  # Ensure general queries get results
                    relevance_score = max(relevance_score, 1)
                
                # Include frame if it has matches or for very general queries
                if matches > 0 or len(description_lower) > 20:  # Include substantial descriptions
                    timestamp = frame['timestamp']
                    youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
                    
                    start_fmt = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                    end_fmt = f"{int((timestamp+23)//60):02d}:{int((timestamp+23)%60):02d}"
                    
                    matching_frames.append({
                        "content": frame['description'],
                        "metadata": frame,
                        "timestamp_range": f"{start_fmt} - {end_fmt}",
                        "youtube_link": youtube_link,
                        "source_type": "video_frame",
                        "relevance_score": relevance_score,
                        "match_count": matches
                    })
            
            # Sort by relevance score, then by match count
            matching_frames.sort(key=lambda x: (x['relevance_score'], x['match_count']), reverse=True)
            
            # Generate response using Gemini
            if matching_frames:
                # Limit context to top 3 most relevant frames
                frame_context = "\n".join([
                    f"[{frame['timestamp_range']}]: {frame['content']}" 
                    for frame in matching_frames[:3]
                ])
                
                prompt = f"""
                Based on visual analysis of video frames, answer: "{query}"
                
                Visual content found in frames:
                {frame_context}
                
                Instructions:
                - Provide a helpful answer about what was seen in the video frames
                - Include specific timestamps when relevant
                - If the query is very general, summarize the key visual elements across frames
                - Be descriptive and specific about visual details
                """
                
                if not self.gemini_llm:
                    answer = self._get_fallback_answer(matching_frames)
                else:
                    try:
                        response = self.gemini_llm.complete(prompt)
                        answer = str(response)
                    except Exception as e:
                        answer = self._get_fallback_answer(matching_frames)
            else:
                # Fallback: show first few frames for very general queries
                if any(word in query_lower for word in ["what", "describe", "show", "see"]):
                    fallback_frames = frame_descriptions[:3]  # Show first 3 frames
                    if not self.gemini_llm:
                        answer = self._get_fallback_answer(fallback_frames)
                    else:
                        try:
                            frame_context = "\n".join([
                                f"[{int(frame['timestamp']//60):02d}:{int(frame['timestamp']%60):02d}]: {frame['description']}" 
                                for frame in fallback_frames
                            ])
                            
                            prompt = f"""
                            Based on visual analysis of video frames, provide an overview of what this video contains:
                            
                            Video frames analyzed:
                            {frame_context}
                            
                            Summarize the visual content and key elements seen across these frames.
                            """
                            
                            response = self.gemini_llm.complete(prompt)
                            answer = str(response)
                        except Exception as e:
                            answer = self._get_fallback_answer(fallback_frames)
                    
                    # Create sources for fallback frames
                    for frame in fallback_frames:
                        timestamp = frame['timestamp']
                        youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
                        start_fmt = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                        end_fmt = f"{int((timestamp+23)//60):02d}:{int((timestamp+23)%60):02d}"
                        
                        matching_frames.append({
                            "content": frame['description'],
                            "metadata": frame,
                            "timestamp_range": f"{start_fmt} - {end_fmt}",
                            "youtube_link": youtube_link,
                            "source_type": "video_frame",
                            "relevance_score": 1
                        })
                else:
                    answer = f"No visual content matching '{query}' was found in the analyzed frames. Try more general terms like 'person', 'text', 'screen', or 'what does this show'."
            
            return {
                "query": query,
                "answer": answer,
                "sources": matching_frames[:5],
                "video_id": video_id,
                "search_type": "visual",
                "status": "processed_with_frames",
                "video_info": {
                    "title": metadata.get("video_info", {}).get("title", "Unknown"),
                    "duration": metadata.get("video_info", {}).get("duration", 0),
                    "processed_at": metadata.get("processed_at", ""),
                    "frames_analyzed": len(frame_descriptions)
                }
            }
            
        except Exception as e:
            print(f"❌ Visual search failed: {e}")
            return {
                "query": query,
                "answer": f"Error during visual search: {str(e)}",
                "sources": [],
                "video_id": video_id,
                "search_type": "visual",
                "status": "error",
                "error": str(e)
            }
    
    async def _download_video_for_frames(self, youtube_url: str, max_duration: int = 300) -> Optional[Dict]:
        """Download video for frame extraction"""
        ydl_opts = {
            'format': 'best[height<=720]',
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
                    'duration': duration
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
        """Load video index from storage (implement if needed)"""
        # For now, return False - requires re-processing
        return False
    
    async def cleanup(self):
        """Cleanup resources"""
        self.active_indices.clear()
        print("✅ VideoRAGProcessor cleaned up")