// src/components/SearchInterface.tsx - Enhanced Chat Version
import React, { useState, useEffect, useRef } from 'react';
import { Eye, MessageSquare, Sparkles, Send, Bot, Clock, Copy, ExternalLink, Youtube } from 'lucide-react';
import { SearchResponse } from '../types/api';
import { LoadingSpinner } from './LoadingSpinner';

interface ChatMessage {
  type: 'user' | 'assistant';
  query: string;
  timestamp: Date;
  response?: SearchResponse;
}

interface SearchInterfaceProps {
  videoId: string;
  searchResults: SearchResponse | null;
  isSearching: boolean;
  error: string | null;
  onSearch: (query: string) => Promise<void>;
  onVisualSearch: (query: string) => Promise<void>;
  onClearError: () => void;
}

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  videoId,
  searchResults,
  isSearching,
  error,
  onSearch,
  onVisualSearch,
  onClearError
}) => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<'transcript' | 'visual'>('transcript');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Reset chat history when video changes
  useEffect(() => {
    setChatHistory([]);
    setQuery('');
    onClearError();
  }, [videoId, onClearError]);

  // Scroll to bottom when chat updates
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Update chat history when search results arrive
  useEffect(() => {
    if (searchResults && chatHistory.length > 0) {
      const lastMessage = chatHistory[chatHistory.length - 1];
      if (lastMessage.type === 'user' && !lastMessage.response) {
        setChatHistory(prev => prev.map((msg, idx) => 
          idx === prev.length - 1 ? { ...msg, response: searchResults } : msg
        ));
      }
    }
  }, [searchResults, chatHistory]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !videoId) return;

    // Add user message to chat
    const userMessage: ChatMessage = {
      type: 'user',
      query: query.trim(),
      timestamp: new Date()
    };
    setChatHistory(prev => [...prev, userMessage]);

    try {
      if (searchType === 'visual') {
        await onVisualSearch(query);
      } else {
        await onSearch(query);
      }
      setQuery('');
    } catch (err) {
      console.error('Search error:', err);
    }
  };

  const handleQuickQuery = (quickQuery: string) => {
    setQuery(quickQuery);
  };

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

  const conversationStarters = {
    transcript: [
      "What is this video about?",
      "Summarize the key points",
      "What problems are discussed?",
      "What solutions are mentioned?",
      "Who is speaking in this video?",
      "What topics are covered?"
    ],
    visual: [
      "What do I see on screen?",
      "Show me when there's code",
      "When do people appear?",
      "What images are displayed?",
      "Find screenshots or diagrams",
      "Show me the visual content"
    ]
  };

  return (
    <div className="card h-[calc(100vh-12rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b pb-4 mb-4">
        <div className="flex items-center space-x-2">
          <Bot className="h-6 w-6 text-primary" />
          <h2 className="text-xl font-semibold text-gray-900">Chat with Your Video</h2>
        </div>
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
          <button
            type="button"
            onClick={() => setSearchType('transcript')}
            className={`flex items-center justify-center space-x-2 py-1 px-3 rounded-md text-sm font-medium transition-colors ${
              searchType === 'transcript'
                ? 'bg-white text-primary shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <MessageSquare className="h-4 w-4" />
            <span>Content</span>
          </button>
          <button
            type="button"
            onClick={() => setSearchType('visual')}
            className={`flex items-center justify-center space-x-2 py-1 px-3 rounded-md text-sm font-medium transition-colors ${
              searchType === 'visual'
                ? 'bg-white text-primary shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Eye className="h-4 w-4" />
            <span>Visuals</span>
          </button>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 px-4">
        {chatHistory.map((message, index) => (
          <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div 
              className={`w-full max-w-[90%] ${
                message.type === 'user' 
                  ? 'bg-indigo-600 text-white rounded-2xl rounded-tr-none shadow-sm' 
                  : 'bg-white border border-gray-200 rounded-2xl rounded-tl-none shadow-sm'
              } p-4`}
            >
              {/* Message Header */}
              <div className="flex items-center justify-between mb-2">
                <span className={`text-xs ${message.type === 'user' ? 'text-indigo-100' : 'text-gray-500'}`}>
                  {message.timestamp.toLocaleTimeString()}
                </span>
                {message.type === 'user' && message.response?.answer && (
                  <button
                    onClick={() => message.response && copyToClipboard(message.response.answer)}
                    className={`text-xs ${message.type === 'user' ? 'text-indigo-100' : 'text-gray-500'} hover:opacity-100`}
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                )}
              </div>

              {/* Message Content */}
              <p className={`text-sm mb-2 ${message.type === 'user' ? 'text-white' : 'text-gray-800'}`}>
                {message.query}
              </p>

              {/* Assistant Response */}
              {message.type === 'user' && message.response && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <p className="text-sm mb-3 text-[#D97707] whitespace-pre-wrap">{message.response.answer}</p>
                  
                  {/* Sources */}
                  {message.response.sources && message.response.sources.length > 0 && (
                    <div className="mt-3 space-y-3">
                      {message.response.sources.map((source, idx) => (
                        <div key={idx} className="text-sm bg-gray-50 p-3 rounded-lg border border-gray-100">
                          <div className="flex items-center justify-between mb-2">
                            <a
                              href={source.youtube_link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center space-x-2 text-blue-600 hover:text-blue-700 font-medium"
                            >
                              <Clock className="h-4 w-4" />
                              <span>{String(source.metadata.timestamp_formatted)}</span>
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          </div>
                          {source.content && (
                            <div className="text-gray-700 mt-2">
                              {(source.content as string).split('\n\n').map((part: string, i: number) => {
                                // Check if the part contains a markdown link
                                const linkMatch = part.match(/\[Watch at (\d{2}:\d{2})\]\((.*?)\)/);
                                if (linkMatch && typeof linkMatch[1] === 'string' && typeof linkMatch[2] === 'string') {
                                  return (
                                    <a
                                      key={i}
                                      href={linkMatch[2]}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-blue-600 hover:text-blue-700 flex items-center space-x-1"
                                    >
                                      <Clock className="h-3 w-3" />
                                      <span>{linkMatch[1]}</span>
                                      <ExternalLink className="h-2 w-2" />
                                    </a>
                                  );
                                }
                                return <p key={i} className="whitespace-pre-wrap text-gray-700">{part}</p>;
                              })}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Response Footer */}
                  <div className="mt-3 text-xs text-gray-600 flex items-center justify-between">
                    <span className="flex items-center space-x-1">
                      <Clock className="h-3 w-3" />
                      <span>{(message.response.response_time * 1000).toFixed(0)}ms</span>
                    </span>
                    <span className="capitalize">
                      {message.response.search_type || 'transcript'} search
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        {isSearching && (
          <div className="flex justify-start">
            <div className="w-full max-w-[90%] bg-white border border-gray-200 rounded-2xl rounded-tl-none p-4 shadow-sm">
              <div className="flex items-center space-x-2 text-indigo-600">
                <LoadingSpinner size="sm" message="" />
                <span className="text-sm">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Chat Input */}
      <div className="border-t pt-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <MessageSquare className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                searchType === 'transcript'
                  ? 'Ask me anything about what was said...'
                  : 'Ask me about what you see in the video...'
              }
              className="input pl-10 pr-12"
              disabled={isSearching}
            />
            <button
              type="submit"
              disabled={isSearching || !query.trim()}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-primary hover:text-blue-600 disabled:text-gray-400"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>

          {/* Quick Starters */}
          <div className="grid grid-cols-2 gap-2">
            {conversationStarters[searchType].map((starter, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleQuickQuery(starter)}
                className="text-left text-xs bg-gray-50 hover:bg-gray-100 text-gray-700 px-3 py-2 rounded-lg transition-colors border border-gray-200 hover:border-gray-300"
                disabled={isSearching}
              >
                "{starter}"
              </button>
            ))}
          </div>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-700">❌ {error}</p>
        </div>
      )}
    </div>
  );
};