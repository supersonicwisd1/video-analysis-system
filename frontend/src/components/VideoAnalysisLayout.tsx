import React, { useState, useCallback, useRef } from 'react';
import VideoPlayer from './VideoPlayer';
import { SearchInterface } from './SearchInterface';
import { MessageSquare, List, X, Maximize2, Minimize2, Clock } from 'lucide-react';
import { SearchResponse } from '../types/api';

interface VideoAnalysisLayoutProps {
  videoId: string;
  videoTitle: string;
  onSearch: (query: string) => Promise<void>;
  onVisualSearch: (query: string) => Promise<void>;
  searchResults: SearchResponse | null;
  isSearching: boolean;
  error: string | null;
  onClearError: () => void;
  sections: Array<{
    title: string;
    startTime: number;
    endTime: number;
    description?: string;
  }>;
}

export const VideoAnalysisLayout: React.FC<VideoAnalysisLayoutProps> = ({
  videoId,
  videoTitle,
  onSearch,
  onVisualSearch,
  searchResults,
  isSearching,
  error,
  onClearError,
  sections
}) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showSections, setShowSections] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const playerRef = useRef<any>(null);

  // Format time for display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle time updates from video player
  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  // Handle player state changes
  const handleStateChange = useCallback((isPlaying: boolean) => {
    setIsPlaying(isPlaying);
  }, []);

  // Handle seeking to timestamp
  const handleSeekToTimestamp = useCallback((timestamp: number) => {
    if (playerRef.current) {
      playerRef.current.seekTo(timestamp);
      playerRef.current.playVideo();
    }
  }, []);

  // Find current section
  const currentSection = sections.find(
    section => currentTime >= section.startTime && currentTime <= section.endTime
  );

  // Toggle fullscreen mode
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
  }, [isFullscreen]);

  return (
    <div className={`min-h-screen flex flex-col bg-white dark:bg-gray-900 transition-colors duration-200 ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Main Content - Vertical Layout */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Video Player Section */}
        <div className={`relative bg-black ${isFullscreen ? 'flex-1' : 'h-[60vh]'}`}>
          <VideoPlayer
            ref={playerRef}
            videoId={videoId}
            onTimeUpdate={handleTimeUpdate}
            onStateChange={handleStateChange}
            className="absolute inset-0"
          />
          
          {/* Video Controls Overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <button
                  onClick={toggleFullscreen}
                  className="p-2 hover:bg-white/10 rounded-full transition-colors"
                  aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                  {isFullscreen ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
                </button>
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4" />
                  <span className="text-sm font-medium">{formatTime(currentTime)}</span>
                </div>
              </div>
              {currentSection && (
                <div className="text-sm bg-white/10 px-3 py-1 rounded-full">
                  {currentSection.title}
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Video Info */}
        <div className="bg-white dark:bg-gray-800 p-4 shadow-sm border-t dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{videoTitle}</h2>
          {currentSection && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Current section: {currentSection.title} ({formatTime(currentSection.startTime)} - {formatTime(currentSection.endTime)})
            </p>
          )}
        </div>

        {/* Chat & Sections Section */}
        {!isFullscreen && (
          <div className="flex-1 flex flex-col border-t dark:border-gray-700 bg-white dark:bg-gray-800">
            {/* Toggle Buttons */}
            <div className="flex border-b dark:border-gray-700">
              <button
                onClick={() => setShowSections(true)}
                className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                  showSections
                    ? 'text-primary border-b-2 border-primary dark:text-primary-light dark:border-primary-light'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <List className="h-4 w-4 inline-block mr-2" />
                Sections
              </button>
              <button
                onClick={() => setShowSections(false)}
                className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                  !showSections
                    ? 'text-primary border-b-2 border-primary dark:text-primary-light dark:border-primary-light'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <MessageSquare className="h-4 w-4 inline-block mr-2" />
                Chat
              </button>
            </div>

            <div className="flex-1 overflow-hidden">
              <div className="h-full flex">
                {/* Sections Panel */}
                {showSections && (
                  <div className="w-80 border-r dark:border-gray-700 overflow-y-auto p-4 space-y-2">
                    {sections.map((section, index) => (
                      <button
                        key={index}
                        onClick={() => handleSeekToTimestamp(section.startTime)}
                        className={`w-full text-left p-3 rounded-lg transition-all ${
                          currentSection === section
                            ? 'bg-primary/10 text-primary dark:bg-primary-light/10 dark:text-primary-light shadow-sm'
                            : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium dark:text-white">{section.title}</span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {formatTime(section.startTime)}
                          </span>
                        </div>
                        {section.description && (
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{section.description}</p>
                        )}
                      </button>
                    ))}
                  </div>
                )}
                
                {/* Chat Interface - Always mounted */}
                <div className={`flex-1 ${showSections ? 'hidden' : 'block'}`}>
                  <SearchInterface
                    videoId={videoId}
                    searchResults={searchResults}
                    isSearching={isSearching}
                    error={error}
                    onSearch={onSearch}
                    onVisualSearch={onVisualSearch}
                    onClearError={onClearError}
                    seekToTimestamp={handleSeekToTimestamp}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 