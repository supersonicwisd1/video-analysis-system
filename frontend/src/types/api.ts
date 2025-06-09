export interface VideoInfo {
  video_id: string;
  title: string;
  duration: number;
  youtube_url: string;
  processed_at: string;
  status: string;
}

export interface SearchSource {
  text: string;
  timestamp: number;
  end_time?: number;
  youtube_link: string;
  confidence: number;
  source_type: string;
  timestamp_formatted?: string;
}

export interface SearchResult {
  text: string;
  timestamp: number;
  end_time?: number;
  youtube_link: string;
  confidence: number;
  source_type?: string;
  timestamp_formatted?: string;
}

export interface SearchResponse {
  query: string;
  search_type: 'transcript' | 'visual' | 'hybrid';
  results?: SearchSource[];
  sources?: SearchSource[];
  answer?: string;
  response_time: number;
  video_id?: string;
}

export interface ProcessingOptions {
  quality?: 'auto' | 'high' | 'medium' | 'low';
  parallel_processing?: boolean;
  preload_segments?: boolean;
  max_parallel_tasks?: number;
  extract_frames?: boolean;
  max_duration?: number;
}

export interface ProcessVideoRequest {
  youtube_url: string;
  options?: ProcessingOptions;
}

export interface ProcessVideoResponse {
  video_id: string;
  status: string;
  message: string;
  metadata: VideoInfo | null;
}

export interface SearchRequest {
  video_id: string;
  query: string;
  top_k?: number;
  search_type?: 'transcript' | 'visual' | 'hybrid';
}

export interface ApiError {
  detail: string;
  status_code?: number;
}

export interface VideoSection {
  id: string;
  title: string;
  start_time: number;
  end_time: number;
  description?: string;
  type: 'auto' | 'manual' | 'youtube' | 'gap';
  confidence: number;
  metadata?: {
    segment_count?: number;
    content_types?: string[];
    is_gap?: boolean;
    merged_sections?: number;
    chapter_index?: number;
    [key: string]: any;
  };
}