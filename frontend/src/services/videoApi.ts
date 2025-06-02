import axios, { type AxiosResponse, type AxiosError, type InternalAxiosRequestConfig } from 'axios';
import {
  type ProcessVideoRequest,
  type ProcessVideoResponse,
  type SearchRequest,
  type SearchResponse,
  type VideoInfo,
  type ApiError
} from '../types/api';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

// Create separate axios instances for different operations
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add CORS headers
  withCredentials: true
});

// Special instance for video processing with longer timeout
const videoProcessingApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 5 minutes for video processing
  // Add CORS headers
  withCredentials: true
});

// Request/Response interceptors for error handling
const errorInterceptor = (error: AxiosError<ApiError>) => {
  console.error('API Error:', {
    status: error.response?.status,
    statusText: error.response?.statusText,
    data: error.response?.data,
    message: error.message,
    code: error.code
  });

  if (error.code === 'ECONNABORTED') {
    throw new Error('Request timed out. The server is taking too long to respond.');
  }
  
  if (error.code === 'ERR_NETWORK') {
    throw new Error('Network error: Could not connect to the server. Please check if the backend is running at ' + API_BASE_URL);
  }

  if (error.response?.data?.detail) {
    throw new Error(error.response.data.detail);
  }

  if (error.response?.status === 404) {
    throw new Error('API endpoint not found. Please check the server configuration.');
  }

  if (error.response?.status === 500) {
    throw new Error('Server error. Please check the backend logs.');
  }

  throw new Error(error.message || 'An unexpected error occurred');
};

// Add request interceptor for debugging
const requestInterceptor = (config: InternalAxiosRequestConfig) => {
  console.log('API Request:', {
    method: config.method,
    url: config.url,
    data: config.data,
    headers: config.headers
  });
  return config;
};

api.interceptors.request.use(requestInterceptor);
videoProcessingApi.interceptors.request.use(requestInterceptor);

api.interceptors.response.use(
  (response) => {
    console.log('API Response:', {
      status: response.status,
      data: response.data
    });
    return response;
  },
  errorInterceptor
);

videoProcessingApi.interceptors.response.use(
  (response) => {
    console.log('API Response:', {
      status: response.status,
      data: response.data
    });
    return response;
  },
  errorInterceptor
);

export class VideoApi {
  /**
   * Process a YouTube video for RAG search
   */
  static async processVideo(request: ProcessVideoRequest): Promise<ProcessVideoResponse> {
    const response: AxiosResponse<ProcessVideoResponse> = await videoProcessingApi.post('/videos/process', request);
    return response.data;
  }

  /**
   * Search within a processed video (transcript)
   */
  static async searchVideo(request: SearchRequest): Promise<SearchResponse> {
    const response: AxiosResponse<SearchResponse> = await api.post('/search/', request);
    return response.data;
  }

  /**
   * Search visual content in video frames
   */
  static async searchVisualContent(request: SearchRequest): Promise<SearchResponse> {
    const response: AxiosResponse<SearchResponse> = await api.post('/search/visual', request);
    return response.data;
  }

  /**
   * Get video information
   */
  static async getVideoInfo(videoId: string): Promise<VideoInfo> {
    const response: AxiosResponse<VideoInfo> = await api.get(`/videos/${videoId}`);
    return response.data;
  }

  /**
   * Get processed videos list
   */
  static async getProcessedVideos(): Promise<VideoInfo[]> {
    const response: AxiosResponse<VideoInfo[]> = await api.get('/videos/');
    return response.data;
  }

  /**
   * Check if API is healthy
   */
  static async healthCheck(): Promise<{ status: string }> {
    const response = await api.get('/health');
    return response.data;
  }
}

// Utility functions
export const extractVideoId = (url: string): string | null => {
  const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/;
  const match = url.match(regex);
  return match ? match[1] : null;
};

export const isValidYouTubeUrl = (url: string): boolean => {
  return extractVideoId(url) !== null;
};

export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};