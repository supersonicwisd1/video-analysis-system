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
        hasMetadata: !!processedVideo.metadata
      } : null,
      isSearchEnabled,
      hasError: !!error
    });
  }, [isProcessing, isSearching, processedVideo, error, isSearchEnabled]);

  // Add more detailed logging for search results
  useEffect(() => {
    if (searchResults) {
      console.log('useVideoSearch: Search results updated:', {
        query: searchResults.query,
        searchType: searchResults.search_type,
        resultsCount: searchResults.results?.length || 0,
        hasResults: !!searchResults.results?.length,
        responseTime: searchResults.response_time
      });
    }
  }, [searchResults]);

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
      if (!result.metadata) {
        throw new Error('Invalid response: Missing metadata');
      }
      
      // Update state with validated response
      console.log('Setting processed video state with:', {
        video_id: result.video_id,
        status: result.status,
        hasMetadata: !!result.metadata
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
    console.log('useVideoSearch: Starting video search with request:', request);
    setIsSearching(true);
    setError(null);
    setSearchResults(null); // Clear previous results
    
    try {
      console.log('useVideoSearch: Making API call to search video...');
      const result = await VideoApi.searchVideo(request);
      console.log('useVideoSearch: Raw API response:', result);
      
      // Transform the response to match our expected format
      const transformedResult = {
        ...result,
        sources: result.sources || result.results || [],
        answer: result.answer || result.text || ''
      };
      console.log('useVideoSearch: Transformed response:', transformedResult);
      
      setSearchResults(transformedResult);
    } catch (err) {
      console.error('useVideoSearch: Search failed:', err);
      setError(err instanceof Error ? err.message : 'Search failed');
      setSearchResults(null);
    } finally {
      setIsSearching(false);
      console.log('useVideoSearch: Search complete, setting isSearching to false');
    }
  }, []);

  const searchVisualContent = useCallback(async (request: SearchRequest) => {
    console.log('useVideoSearch: Starting visual search with request:', request);
    setIsSearching(true);
    setError(null);
    setSearchResults(null); // Clear previous results
    
    try {
      // Ensure search_type is set to 'visual'
      const visualRequest = {
        ...request,
        search_type: 'visual' as const
      };
      
      console.log('useVideoSearch: Making API call to search visual content...', visualRequest);
      const result = await VideoApi.searchVisualContent(visualRequest);
      console.log('useVideoSearch: Raw visual search response:', result);
      
      // Transform the response to match our expected format
      const transformedResult = {
        ...result,
        sources: result.sources || result.results || [],
        answer: result.answer || result.text || ''
      };
      console.log('useVideoSearch: Transformed visual response:', transformedResult);
      
      setSearchResults(transformedResult);
    } catch (err) {
      console.error('useVideoSearch: Visual search failed:', err);
      setError(err instanceof Error ? err.message : 'Visual search failed');
      setSearchResults(null);
    } finally {
      setIsSearching(false);
      console.log('useVideoSearch: Visual search complete, setting isSearching to false');
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