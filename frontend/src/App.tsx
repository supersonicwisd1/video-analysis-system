import React, { useState, useEffect, useMemo } from 'react';
import { Layout } from './components/Layout';
import { VideoProcessor } from './components/VideoProcessor';
import { SearchInterface } from './components/SearchInterface';
import { ResultsDisplay } from './components/ResultsDisplay';
import { useVideoSearch } from './hooks/useVideoSearch';
import { MessageSquare, Eye, Clock } from 'lucide-react';
import { SearchResponse, ProcessVideoRequest, ProcessVideoResponse } from './types/api';
import { VideoAnalysisLayout } from './components/VideoAnalysisLayout';

function App() {
  // 1. All useState hooks first
  const [currentVideoId, setCurrentVideoId] = useState<string>('');
  
  // 2. All custom hooks next
  const {
    processedVideo,
    isProcessing,
    isSearching,
    searchResults,
    error,
    processVideo,
    searchVideo,
    searchVisualContent,
    clearError
  } = useVideoSearch();

  // Wrap processVideo to update currentVideoId
  const processVideoWrapper = async (request: ProcessVideoRequest): Promise<ProcessVideoResponse> => {
    const result = await processVideo(request);
    if (result.status === 'completed' && result.video_id) {
      setCurrentVideoId(result.video_id);
    }
    return result;
  };

  // Debug logging
  useEffect(() => {
    console.log('App: State update:', {
      currentVideoId,
      processedVideoId: processedVideo?.video_id,
      processedVideoStatus: processedVideo?.status,
      isSearching,
      hasSearchResults: !!searchResults,
      error,
      isProcessing,
      isSearchEnabled: true // Assuming isSearchEnabled is always true in useVideoSearch
    });
  }, [currentVideoId, processedVideo, isSearching, searchResults, error, isProcessing]);

  // Determine if we should show the search interface
  const shouldShowSearch = Boolean(
    processedVideo?.video_id && 
    processedVideo.status === 'completed' && 
    currentVideoId === processedVideo.video_id &&
    true // Assuming isSearchEnabled is always true in useVideoSearch
  );

  // Log render state
  console.log('App: Render state:', {
    shouldShowSearch,
    hasVideoId: !!processedVideo?.video_id,
    videoStatus: processedVideo?.status,
    currentVideoId,
    isSearchEnabled: true // Assuming isSearchEnabled is always true in useVideoSearch
  });

  // Transform video sections for the layout
  const videoSections = useMemo(() => {
    if (!processedVideo?.sections) return [];
    return processedVideo.sections.map((section: any) => ({
      title: section.title,
      startTime: section.start_time,
      endTime: section.end_time,
      description: section.description
    }));
  }, [processedVideo?.sections]);

  // Add search handler functions
  const handleSearch = async (query: string) => {
    if (!currentVideoId) return;
    console.log('App: Starting search with query:', query);
    try {
      const request = {
        video_id: currentVideoId,
        query: query,
        search_type: 'transcript' as const
      };
      await searchVideo(request);
    } catch (err) {
      console.error('App: Search failed:', err);
    }
  };

  const handleVisualSearch = async (query: string) => {
    if (!currentVideoId) return;
    console.log('App: Starting visual search with query:', query);
    try {
      const request = {
        video_id: currentVideoId,
        query: query,
        search_type: 'visual' as const
      };
      await searchVisualContent(request);
    } catch (err) {
      console.error('App: Visual search failed:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Layout>
        <main className="container mx-auto px-4 py-8">
          {!shouldShowSearch ? (
            <VideoProcessor
              onProcessVideo={processVideoWrapper}
              isProcessing={isProcessing}
              error={error}
              onClearError={clearError}
            />
          ) : (
            <VideoAnalysisLayout
              videoId={currentVideoId}
              videoTitle={processedVideo?.video_info?.title || ''}
              onSearch={handleSearch}
              onVisualSearch={handleVisualSearch}
              searchResults={searchResults}
              isSearching={isSearching}
              error={error}
              onClearError={clearError}
              sections={videoSections}
            />
          )}
        </main>
      </Layout>
    </div>
  );
}

export default App;