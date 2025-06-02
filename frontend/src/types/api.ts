export interface VideoInfo {
  video_id: string;
  title: string;
  duration: number;
  youtube_url: string;
  processed_at: string;
  status: string;
}

export interface SearchSource {
  content: string;
  metadata: Record<string, unknown>;
  timestamp_range?: string;
  youtube_link?: string;
  source_type: string;
  relevance_score?: number;
}

export interface SearchResponse {
  query: string;
  answer: string;
  sources: SearchSource[];
  video_id: string;
  response_time: number;
  search_type?: string;
}

export interface ProcessVideoRequest {
  youtube_url: string;
  extract_frames: boolean;
  max_duration: number;
}

export interface ProcessVideoResponse {
  video_id: string;
  status: string;
  message: string;
  video_info: VideoInfo;
}

export interface SearchRequest {
  video_id: string;
  query: string;
  top_k?: number;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}