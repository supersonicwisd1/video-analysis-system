export enum ContentType {
  TRANSCRIPT = 'transcript',
  SPEECH_TO_TEXT = 'speech_to_text',
  VISUAL = 'visual',
  METADATA = 'metadata',
}

export interface VideoSegment {
  id: string;
  start_time: number;
  end_time: number;
  content_type: ContentType;
  content: string;
  metadata: {
    source: string;
    confidence: number;
    [key: string]: any;
  };
  confidence: number;
}

export interface VideoSection {
  id: string;
  start_time: number;
  end_time: number;
  title: string;
  content: string;
  content_type: ContentType;
  confidence: number;
  segments: VideoSegment[];
}

export interface VideoMetadata {
  video_id: string;
  youtube_url: string;
  title: string;
  duration: number;
  thumbnail_url: string;
  processed_at: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  sections: VideoSection[];
  segments: VideoSegment[];
  frames_extracted: number;
  frame_descriptions: Array<{
    timestamp: number;
    description: string;
    frame_path: string;
  }>;
}

export interface SearchResult {
  query: string;
  answer: string;
  sources: Array<{
    content: string;
    metadata: {
      source_type: ContentType;
      start_time: number;
      end_time: number;
      timestamp: {
        start: number;
        end: number;
        formatted: string;
        youtube_link: string;
      };
      confidence: number;
      [key: string]: any;
    };
    source_type: string;
    confidence: number;
  }>;
  video_id: string;
  response_time: number;
}

export interface ChatMessage {
  id: string;
  message: string;
  response?: string;
  timestamp: number;
  conversation_id: string;
  video_id: string;
  sources?: SearchResult['sources'];
  status: 'sent' | 'delivered' | 'failed';
  error?: string;
}

export interface ProcessingStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  current_step?: string;
  error?: string;
  video_id: string;
  batch_id?: string;
}

export interface BatchProcessingStatus {
  batch_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  total: number;
  completed: number;
  failed: number;
  videos: {
    [videoId: string]: 'pending' | 'processing' | 'completed' | 'failed';
  };
  error?: string;
} 