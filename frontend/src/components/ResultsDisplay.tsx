import React from 'react';
import { 
  MessageSquare, 
  Eye, 
  Clock, 
  ExternalLink, 
  Copy, 
  Star,
  Youtube
} from 'lucide-react';
import { SearchResponse } from '../types/api';

interface ResultsDisplayProps {
  results: SearchResponse | null;
  isLoading?: boolean;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, isLoading }) => {
  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const openInYouTube = (link: string) => {
    window.open(link, '_blank', 'noopener,noreferrer');
  };

  if (isLoading) {
    return (
      <div className="card">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  const searchTypeIcon = results.search_type === 'visual' ? Eye : MessageSquare;
  const SearchIcon = searchTypeIcon;

  return (
    <div className="space-y-6">
      {/* Answer Section */}
      <div className="card">
        <div className="flex items-start space-x-3 mb-4">
          <div className="flex-shrink-0">
            <SearchIcon className="h-6 w-6 text-primary" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 mb-2">
              AI Answer
            </h3>
            <div className="bg-blue-50 border-l-4 border-primary p-4 rounded-r-lg">
              <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                {results.answer}
              </p>
            </div>
            
            <div className="flex items-center justify-between mt-4 text-sm text-gray-500">
              <div className="flex items-center space-x-4">
                <span className="flex items-center space-x-1">
                  <Clock className="h-4 w-4" />
                  <span>{(results.response_time * 1000).toFixed(0)}ms</span>
                </span>
                <span className="capitalize">
                  {results.search_type || 'transcript'} search
                </span>
              </div>
              <button
                onClick={() => copyToClipboard(results.answer)}
                className="flex items-center space-x-1 text-primary hover:text-blue-600 transition-colors"
              >
                <Copy className="h-4 w-4" />
                <span>Copy</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Sources Section */}
      {results.sources && results.sources.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <Star className="h-5 w-5 text-yellow-500" />
            <span>Sources & Timestamps</span>
          </h3>
          
          <div className="space-y-4">
            {results.sources.map((source, index) => (
              <div
                key={index}
                className="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="inline-flex items-center justify-center w-6 h-6 bg-primary text-white text-xs font-medium rounded-full">
                      {index + 1}
                    </span>
                    {source.timestamp_range && (
                      <span className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">
                        <Clock className="h-3 w-3 mr-1" />
                        {source.timestamp_range}
                      </span>
                    )}
                    <span className="inline-flex items-center px-2 py-1 bg-gray-200 text-gray-700 text-xs font-medium rounded">
                      {source.source_type === 'video_frame' ? (
                        <><Eye className="h-3 w-3 mr-1" />Visual</>
                      ) : (
                        <><MessageSquare className="h-3 w-3 mr-1" />Transcript</>
                      )}
                    </span>
                  </div>

                  {source.youtube_link && (
                    <button
                      onClick={() => openInYouTube(source.youtube_link!)}
                      className="flex items-center space-x-1 text-red-600 hover:text-red-700 text-sm font-medium transition-colors"
                    >
                      <Youtube className="h-4 w-4" />
                      <span>Watch</span>
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  )}
                </div>

                <div className="text-gray-700 text-sm leading-relaxed">
                  {source.content}
                </div>

                {source.relevance_score && source.relevance_score > 1 && (
                  <div className="mt-2 text-xs text-gray-500">
                    Relevance: {source.relevance_score} matches
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Query Info */}
      <div className="text-center text-sm text-gray-500">
        <p>
          Searched for: <span className="font-medium">"{results.query}"</span> in video{' '}
          <code className="bg-gray-100 px-2 py-1 rounded font-mono">
            {results.video_id}
          </code>
        </p>
      </div>
    </div>
  );
};