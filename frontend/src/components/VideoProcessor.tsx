import React, { useState } from 'react';
import { Upload, Youtube, Settings, AlertCircle, CheckCircle } from 'lucide-react';
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
  const [maxDuration, setMaxDuration] = useState(300);
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
      extractFrames,
      maxDuration
    });

    try {
      const result = await onProcessVideo({
        youtube_url: youtubeUrl,
        extract_frames: extractFrames,
        max_duration: maxDuration
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
    <div className="card">
      <div className="flex items-center space-x-2 mb-6">
        <Upload className="h-6 w-6 text-primary" />
        <h2 className="text-xl font-semibold text-gray-900">Process Video</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* YouTube URL Input */}
        <div>
          <label htmlFor="youtube-url" className="block text-sm font-medium text-gray-700 mb-2">
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
              className={`input pl-10 ${!isValidUrl ? 'border-red-300 focus:ring-red-500' : ''}`}
              disabled={isProcessing}
            />
          </div>
          {!isValidUrl && (
            <p className="mt-1 text-sm text-red-600 flex items-center">
              <AlertCircle className="h-4 w-4 mr-1" />
              Please enter a valid YouTube URL
            </p>
          )}
          {videoId && (
            <p className="mt-1 text-sm text-gray-600">
              Video ID: <code className="bg-gray-100 px-2 py-1 rounded">{videoId}</code>
            </p>
          )}
        </div>

        {/* Advanced Options */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900"
          >
            <Settings className="h-4 w-4" />
            <span>Advanced Options</span>
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4 bg-gray-50 p-4 rounded-lg">
              {/* Frame Extraction */}
              <div className="flex items-center space-x-3">
                <input
                  id="extract-frames"
                  type="checkbox"
                  checked={extractFrames}
                  onChange={(e) => setExtractFrames(e.target.checked)}
                  className="h-4 w-4 text-primary border-gray-300 rounded focus:ring-primary"
                  disabled={isProcessing}
                />
                <label htmlFor="extract-frames" className="text-sm text-gray-700">
                  Extract video frames for visual search
                </label>
              </div>

              {/* Max Duration */}
              <div>
                <label htmlFor="max-duration" className="block text-sm font-medium text-gray-700 mb-1">
                  Max Duration (seconds): {maxDuration}
                </label>
                <input
                  id="max-duration"
                  type="range"
                  min="60"
                  max="600"
                  step="30"
                  value={maxDuration}
                  onChange={(e) => setMaxDuration(Number(e.target.value))}
                  className="w-full"
                  disabled={isProcessing}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1 min</span>
                  <span>10 min</span>
                </div>
              </div>

              {extractFrames && (
                <div className="text-sm text-amber-600 bg-amber-50 p-3 rounded">
                  ⚠️ Frame extraction requires OpenAI API key and will use additional API credits
                </div>
              )}
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isProcessing || !youtubeUrl || !isValidUrl}
          className="btn btn-primary w-full"
        >
          {isProcessing ? (
            <div className="flex items-center justify-center space-x-2">
              <LoadingSpinner size="sm" message="" />
              <span>Processing Video...</span>
            </div>
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
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <h3 className="font-medium text-red-900">Processing Failed</h3>
          </div>
          <p className="mt-1 text-sm text-red-700">{error}</p>
          {error.includes('no transcript available') && (
            <div className="mt-2 text-sm text-red-600">
              <p>Tips:</p>
              <ul className="list-disc list-inside mt-1">
                <li>Make sure the video has captions/subtitles enabled</li>
                <li>Try a different video with captions</li>
                <li>Check if the video is public and accessible</li>
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};