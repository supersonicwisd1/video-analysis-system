import React, { useState } from 'react';
import { Upload, Youtube, AlertCircle, Settings, Loader2 } from 'lucide-react';
import { isValidYouTubeUrl, extractVideoId } from '../services/videoApi';
import { LoadingSpinner } from './LoadingSpinner';
import { type ProcessVideoRequest, type ProcessVideoResponse } from '../types/api';

// Create a simple event emitter for video processing
const videoProcessedEvent = new Event('videoProcessed');

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    if (message.includes('no transcript available')) {
      return 'This video does not have captions/subtitles available. Please try a different video with captions enabled.';
    }
    if (message.includes('timeout')) {
      return 'The video processing took too long. Please try a shorter video or try again later.';
    }
    if (message.includes('invalid url')) {
      return 'Please enter a valid YouTube URL.';
    }
    if (message.includes('private video')) {
      return 'This video is private or unavailable. Please try a public video.';
    }
    return error.message;
  }
  return 'An unexpected error occurred. Please try again.';
};

interface VideoProcessorProps {
  isProcessing: boolean;
  processedVideo: ProcessVideoResponse | null;
  error: string | null;
  onProcessVideo: (request: ProcessVideoRequest) => Promise<ProcessVideoResponse>;
  onClearError: () => void;
}

export const VideoProcessor: React.FC<VideoProcessorProps> = ({
  isProcessing,
  processedVideo,
  error: propError,
  onProcessVideo,
  onClearError
}) => {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [extractFrames, setExtractFrames] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const error = propError || localError;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    onClearError();
    setLocalError(null);

    if (!youtubeUrl.trim()) {
      console.log('Submit prevented: Empty URL');
      return;
    }

    if (!isValidYouTubeUrl(youtubeUrl)) {
      console.log('Submit prevented: Invalid YouTube URL');
      setLocalError('Please enter a valid YouTube URL');
      return;
    }

    console.log('Starting video processing:', {
      url: youtubeUrl,
      extractFrames
    });

    try {
      const result = await onProcessVideo({
        youtube_url: youtubeUrl,
        extract_frames: extractFrames
      });
      
      console.log('Video processing completed:', result);
      
      // Clear the URL input after successful processing
      if (result.status === 'completed' && result.video_id) {
        setYoutubeUrl('');
      } else {
        throw new Error('Video processing did not complete successfully');
      }
    } catch (err) {
      console.error('Video processing failed:', err);
      setLocalError(err instanceof Error ? err.message : 'Failed to process video');
    }
  };

  const isValidUrl = youtubeUrl ? isValidYouTubeUrl(youtubeUrl) : true;
  const videoId = extractVideoId(youtubeUrl);

  return (
    <div className="card bg-white dark:bg-gray-800 shadow-lg rounded-xl overflow-hidden">
      <div className="p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 bg-primary/10 dark:bg-primary-light/10 rounded-lg">
            <Upload className="h-6 w-6 text-primary dark:text-primary-light" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Process Video</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Enter a YouTube URL to start analyzing the video
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* YouTube URL Input */}
          <div>
            <label htmlFor="youtube-url" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              YouTube URL
            </label>
            <div className="relative">
              <Youtube className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                id="youtube-url"
                type="url"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=..."
                className={`w-full pl-10 pr-4 py-2.5 rounded-lg border ${
                  !isValidUrl 
                    ? 'border-red-300 dark:border-red-500 focus:ring-red-500 dark:focus:ring-red-400' 
                    : 'border-gray-300 dark:border-gray-600 focus:ring-primary dark:focus:ring-primary-light'
                } bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:border-transparent transition-colors`}
                disabled={isProcessing}
              />
            </div>
            {!isValidUrl && (
              <p className="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center">
                <AlertCircle className="h-4 w-4 mr-1.5" />
                Please enter a valid YouTube URL
              </p>
            )}
            {videoId && (
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                Video ID: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">{videoId}</code>
              </p>
            )}
          </div>

          {/* Advanced Options */}
          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <Settings className="h-4 w-4" />
              <span>Advanced Options</span>
            </button>

            {showAdvanced && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg space-y-4">
                {/* Frame Extraction */}
                <div className="flex items-center space-x-3">
                  <input
                    id="extract-frames"
                    type="checkbox"
                    checked={extractFrames}
                    onChange={(e) => setExtractFrames(e.target.checked)}
                    className="h-4 w-4 text-primary dark:text-primary-light border-gray-300 dark:border-gray-600 rounded focus:ring-primary dark:focus:ring-primary-light bg-white dark:bg-gray-700"
                    disabled={isProcessing}
                  />
                  <label htmlFor="extract-frames" className="text-sm text-gray-700 dark:text-gray-300">
                    Extract video frames for visual search
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isProcessing || !youtubeUrl.trim() || !isValidUrl}
            className="w-full flex items-center justify-center px-4 py-2.5 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-primary dark:bg-primary-light hover:bg-primary-dark dark:hover:bg-primary-light/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary dark:focus:ring-primary-light disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isProcessing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Processing Video...
              </>
            ) : (
              'Process Video'
            )}
          </button>
        </form>

        {/* Success Message */}
        {(processedVideo) && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <h3 className="font-medium text-green-900">Video Processed Successfully!</h3>
            </div>
            <div className="mt-2 text-sm text-green-700">
              <p><strong>Video ID:</strong> {processedVideo.video_id}</p>
              <p><strong>Title:</strong> {processedVideo.video_info.title}</p>
              {(() => {
                const duration = processedVideo.video_info.duration;
                if (typeof duration === 'number') {
                  const minutes = Math.floor(duration / 60);
                  const seconds = duration % 60;
                  return (
                    <p><strong>Duration:</strong> {minutes}:{seconds.toString().padStart(2, '0')}</p>
                  );
                }
                return null;
              })()}
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-700 dark:text-red-400 flex items-center">
              <AlertCircle className="h-4 w-4 mr-1.5" />
              {error}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};