from youtube_transcript_api import YouTubeTranscriptApi
import re
from typing import Optional, List, Dict


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id: str) -> Optional[List[Dict]]:
    """Get transcript with timestamps"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"❌ Error getting transcript: {e}")
        return None
