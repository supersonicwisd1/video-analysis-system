import { useState, useCallback, useEffect } from 'react';
import { VideoApi } from '../services/videoApi';
import {
  type ProcessVideoRequest,
  type ProcessVideoResponse,
  type SearchRequest,
  type SearchResponse,
  type ApiError
} from '../types/api';

interface UseVideoSearchReturn {
  // State
  isProcessing: boolean;
  isSearching: boolean;
  processedVideo: ProcessVideoResponse | null;
  searchResults: SearchResponse | null;
  error: string | null;
  isSearchEnabled: boolean;
  
  // Actions
  processVideo: (request: ProcessVideoRequest) => Promise<ProcessVideoResponse>;
  searchVideo: (request: SearchRequest) => Promise<void>;
  searchVisualContent: (request: SearchRequest) => Promise<void>;
  clearError: () => void;
  reset: () => void;
  enableSearch: () => void;
}

export const useVideoSearch = (): UseVideoSearchReturn => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [processedVideo, setProcessedVideo] = useState<ProcessVideoResponse | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSearchEnabled, setIsSearchEnabled] = useState(false);

  // Debug state changes
  useEffect(() => {
    console.log('useVideoSearch state updated:', {
      isProcessing,
      isSearching,
      processedVideo: processedVideo ? {
        video_id: processedVideo.video_id,
        status: processedVideo.status,
        hasVideoInfo: !!processedVideo.video_info
      } : null,
      isSearchEnabled,
      hasError: !!error
    });
  }, [isProcessing, isSearching, processedVideo, error, isSearchEnabled]);

  const processVideo = useCallback(async (request: ProcessVideoRequest) => {
    console.log('Starting video processing with request:', request);
    
    try {
      // Reset states
      setIsProcessing(true);
      setError(null);
      setIsSearchEnabled(false);
      
      // Make API call
      console.log('Making API call to process video...');
      const result = await VideoApi.processVideo(request);
      console.log('API response received:', result);
      
      // Validate response
      if (!result.video_id) {
        throw new Error('Invalid response: Missing video_id');
      }
      if (!result.video_info) {
        throw new Error('Invalid response: Missing video_info');
      }
      
      // Update state with validated response
      console.log('Setting processed video state with:', {
        video_id: result.video_id,
        status: result.status,
        hasVideoInfo: !!result.video_info
      });
      setProcessedVideo(result);
      
      // Enable search interface after successful processing
      if (result.status === 'completed') {
        console.log('Enabling search interface for video:', result.video_id);
        setIsSearchEnabled(true);
      }
      
      return result;
    } catch (err) {
      console.error('Video processing failed:', err);
      let errorMessage = 'Failed to process video';
      
      if (err instanceof Error) {
        errorMessage = err.message;
      } else if (typeof err === 'object' && err !== null && 'detail' in err) {
        const apiError = err as ApiError;
        errorMessage = apiError.detail || 'Server error occurred';
      }
      
      setError(errorMessage);
      setProcessedVideo(null);
      setIsSearchEnabled(false);
      throw err;
    } finally {
      console.log('Processing complete, setting isProcessing to false');
      setIsProcessing(false);
    }
  }, []);

  const searchVideo = useCallback(async (request: SearchRequest) => {
    console.log('Starting video search with request:', request);
    setIsSearching(true);
    setError(null);
    
    try {
      console.log('Making API call to search video...');
      const result = await VideoApi.searchVideo(request);
      console.log('Search API response received:', result);
      setSearchResults(result);
      console.log('Search results state updated:', result);
    } catch (err) {
      console.error('Search failed:', err);
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsSearching(false);
      console.log('Search complete, setting isSearching to false');
    }
  }, []);

  const searchVisualContent = useCallback(async (request: SearchRequest) => {
    console.log('Starting visual search with request:', request);
    setIsSearching(true);
    setError(null);
    
    try {
      console.log('Making API call to search visual content...');
      const result = await VideoApi.searchVisualContent(request);
      console.log('Visual search API response received:', result);
      setSearchResults(result);
      console.log('Visual search results state updated:', result);
    } catch (err) {
      console.error('Visual search failed:', err);
      setError(err instanceof Error ? err.message : 'Visual search failed');
    } finally {
      setIsSearching(false);
      console.log('Visual search complete, setting isSearching to false');
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const reset = useCallback(() => {
    setProcessedVideo(null);
    setSearchResults(null);
    setError(null);
  }, []);

  const enableSearch = useCallback(() => {
    if (processedVideo?.video_id && !isProcessing) {
      console.log('Manually enabling search for video:', processedVideo.video_id);
      setIsSearchEnabled(true);
    }
  }, [processedVideo, isProcessing]);

  // Add a cleanup effect to prevent state resets
  useEffect(() => {
    return () => {
      // Don't reset state on unmount
      // This prevents state from being reset when components re-render
    };
  }, []);

  return {
    isProcessing,
    isSearching,
    processedVideo,
    searchResults,
    error,
    isSearchEnabled,
    processVideo,
    searchVideo,
    searchVisualContent,
    clearError,
    reset,
    enableSearch,
  };
};