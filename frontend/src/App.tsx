import React, { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { VideoProcessor } from './components/VideoProcessor';
import { SearchInterface } from './components/SearchInterface';
import { ResultsDisplay } from './components/ResultsDisplay';
import { useVideoSearch } from './hooks/useVideoSearch';
import { MessageSquare, Eye, Clock } from 'lucide-react';
import { SearchResponse, ProcessVideoRequest, ProcessVideoResponse } from './types/api';

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

  return (
    <Layout>
      <div className="space-y-8">
        {/* Header Section */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🧠 Video RAG Search
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Process YouTube videos and search through their content using AI. 
            Find specific topics, moments, and visual elements with semantic search.
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Processing */}
          <div className="space-y-6">
            <VideoProcessor 
              isProcessing={isProcessing}
              processedVideo={processedVideo}
              error={error}
              onProcessVideo={processVideoWrapper}
              onClearError={clearError}
            />
          </div>

          {/* Right Column: Chat Interface */}
          <div className="space-y-6">
            <div className="sticky top-8">
              {shouldShowSearch && processedVideo ? (
                <SearchInterface 
                  key={`search-${processedVideo.video_id}`}
                  videoId={processedVideo.video_id}
                  searchResults={searchResults}
                  isSearching={isSearching}
                  error={error}
                  onSearch={async (query: string) => {
                    await searchVideo({
                      video_id: processedVideo.video_id,
                      query,
                      top_k: 5
                    });
                  }}
                  onVisualSearch={async (query: string) => {
                    await searchVisualContent({
                      video_id: processedVideo.video_id,
                      query,
                      top_k: 5
                    });
                  }}
                  onClearError={clearError}
                />
              ) : (
                <div className="card text-center py-12">
                  <div className="text-gray-400 mb-4">
                    <MessageSquare className="w-16 h-16 mx-auto" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Ready to Chat
                  </h3>
                  <p className="text-gray-600">
                    Process a video to start chatting about its content
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center space-x-3 mb-3">
              <MessageSquare className="h-8 w-8 text-blue-600" />
              <h3 className="font-semibold text-gray-900">Transcript Search</h3>
            </div>
            <p className="text-gray-600 text-sm">
              Search through spoken content, captions, and dialogue using natural language queries.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center space-x-3 mb-3">
              <Eye className="h-8 w-8 text-purple-600" />
              <h3 className="font-semibold text-gray-900">Visual Search</h3>
            </div>
            <p className="text-gray-600 text-sm">
              Find specific visual elements, scenes, and on-screen content using AI vision analysis.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center space-x-3 mb-3">
              <Clock className="h-8 w-8 text-green-600" />
              <h3 className="font-semibold text-gray-900">Timestamp Links</h3>
            </div>
            <p className="text-gray-600 text-sm">
              Jump directly to relevant moments in the video with clickable timestamp links.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default App;