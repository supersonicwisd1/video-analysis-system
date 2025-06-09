import { useState, useEffect, useCallback, useRef } from 'react';
import { cacheService } from '../services/cache';
import { VideoSegment, ContentType } from '../types/video';

interface UseVideoSegmentsOptions {
  videoId: string;
  initialBatchSize?: number;
  batchSize?: number;
  preloadThreshold?: number;
}

interface VideoSegmentState {
  segments: VideoSegment[];
  sections: any[];
  isLoading: boolean;
  hasMore: boolean;
  error: Error | null;
}

export function useVideoSegments({
  videoId,
  initialBatchSize = 10,
  batchSize = 20,
  preloadThreshold = 0.8,
}: UseVideoSegmentsOptions) {
  const [state, setState] = useState<VideoSegmentState>({
    segments: [],
    sections: [],
    isLoading: true,
    hasMore: true,
    error: null,
  });

  const observer = useRef<IntersectionObserver | null>(null);
  const loadingRef = useRef<HTMLDivElement>(null);
  const processedSegments = useRef<Set<string>>(new Set());
  const abortController = useRef<AbortController | null>(null);

  // Load segments from cache or API
  const loadSegments = useCallback(async (startIndex: number, count: number) => {
    if (abortController.current) {
      abortController.current.abort();
    }
    abortController.current = new AbortController();

    try {
      // Try cache first
      const cached = await cacheService.getVideoMetadata(videoId);
      if (cached) {
        const { segments, sections } = cached;
        const newSegments = segments.slice(startIndex, startIndex + count);
        
        // Filter out already processed segments
        const uniqueSegments = newSegments.filter(
          segment => !processedSegments.current.has(segment.id)
        );

        if (uniqueSegments.length > 0) {
          setState(prev => ({
            ...prev,
            segments: [...prev.segments, ...uniqueSegments],
            sections: sections,
            isLoading: false,
            hasMore: startIndex + count < segments.length,
          }));

          // Mark segments as processed
          uniqueSegments.forEach(segment => {
            processedSegments.current.add(segment.id);
          });
        }

        return;
      }

      // If not in cache, fetch from API
      const response = await fetch(
        `/api/v1/videos/${videoId}/segments?start=${startIndex}&count=${count}`,
        { signal: abortController.current.signal }
      );

      if (!response.ok) {
        throw new Error(`Failed to load segments: ${response.statusText}`);
      }

      const data = await response.json();
      const { segments, sections, hasMore } = data;

      // Cache the results
      if (startIndex === 0) {
        await cacheService.setVideoMetadata(videoId, {
          segments,
          sections,
          lastUpdated: Date.now(),
        });
      }

      // Filter out already processed segments
      const uniqueSegments = segments.filter(
        segment => !processedSegments.current.has(segment.id)
      );

      if (uniqueSegments.length > 0) {
        setState(prev => ({
          ...prev,
          segments: [...prev.segments, ...uniqueSegments],
          sections: sections,
          isLoading: false,
          hasMore,
        }));

        // Mark segments as processed
        uniqueSegments.forEach(segment => {
          processedSegments.current.add(segment.id);
        });
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error : new Error('Unknown error'),
        isLoading: false,
      }));
    }
  }, [videoId]);

  // Initial load
  useEffect(() => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    loadSegments(0, initialBatchSize);

    return () => {
      if (abortController.current) {
        abortController.current.abort();
      }
    };
  }, [videoId, initialBatchSize, loadSegments]);

  // Setup intersection observer for infinite scroll
  useEffect(() => {
    if (!loadingRef.current) return;

    observer.current = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && !state.isLoading && state.hasMore) {
          const currentLength = state.segments.length;
          loadSegments(currentLength, batchSize);
        }
      },
      {
        root: null,
        rootMargin: '100px',
        threshold: preloadThreshold,
      }
    );

    observer.current.observe(loadingRef.current);

    return () => {
      if (observer.current) {
        observer.current.disconnect();
      }
    };
  }, [state.isLoading, state.hasMore, batchSize, preloadThreshold, loadSegments]);

  // Preload next batch when approaching end
  useEffect(() => {
    if (!state.hasMore || state.isLoading) return;

    const preloadIndex = Math.floor(state.segments.length * preloadThreshold);
    if (preloadIndex > 0 && preloadIndex % batchSize === 0) {
      loadSegments(preloadIndex, batchSize);
    }
  }, [state.segments.length, state.hasMore, state.isLoading, batchSize, preloadThreshold, loadSegments]);

  // Get segments by time range
  const getSegmentsByTimeRange = useCallback((startTime: number, endTime: number) => {
    return state.segments.filter(
      segment => segment.start_time >= startTime && segment.end_time <= endTime
    );
  }, [state.segments]);

  // Get segments by content type
  const getSegmentsByType = useCallback((contentType: ContentType) => {
    return state.segments.filter(segment => segment.content_type === contentType);
  }, [state.segments]);

  // Get section by time
  const getSectionByTime = useCallback((time: number) => {
    return state.sections.find(
      section => time >= section.start_time && time <= section.end_time
    );
  }, [state.sections]);

  return {
    ...state,
    loadingRef,
    getSegmentsByTimeRange,
    getSegmentsByType,
    getSectionByTime,
    refresh: () => {
      processedSegments.current.clear();
      setState(prev => ({ ...prev, segments: [], isLoading: true, error: null }));
      loadSegments(0, initialBatchSize);
    },
  };
} 