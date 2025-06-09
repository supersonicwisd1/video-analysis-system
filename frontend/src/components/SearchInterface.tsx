// src/components/SearchInterface.tsx - Enhanced Chat Version
import React, { useState, useEffect, useRef } from 'react';
import { Bot, MessageSquare, Eye, Copy, ExternalLink, Send, Loader2, Clock } from 'lucide-react';
import { SearchResponse, SearchSource } from '../types/api';

interface ChatMessage {
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: SearchSource[];
}

interface SearchInterfaceProps {
  videoId: string;
  searchResults: SearchResponse | null;
  isSearching: boolean;
  error: string | null;
  onSearch: (query: string) => Promise<void>;
  onVisualSearch: (query: string) => Promise<void>;
  onClearError: () => void;
  seekToTimestamp?: (timestamp: number) => void;
}

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  videoId,
  searchResults,
  isSearching,
  error,
  onSearch,
  onVisualSearch,
  onClearError,
  seekToTimestamp
}) => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<'transcript' | 'visual'>('transcript');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
    if (searchResults) {
      console.log('Raw search results received:', searchResults);
      
      // Add user message to chat history
      const userMessage: ChatMessage = {
        type: 'user',
        content: searchResults.query,
        timestamp: `${new Date().toISOString()}-user`  // Make timestamp unique
      };
      console.log('Adding user message:', userMessage);
      
      // Add assistant message with answer and sources
      const assistantMessage: ChatMessage = {
        type: 'assistant',
        content: searchResults.answer || 'No answer available.',
        timestamp: `${new Date().toISOString()}-assistant`,  // Make timestamp unique
        sources: searchResults.sources || searchResults.results || []
      };
      console.log('Adding assistant message:', assistantMessage);

      // Update chat history with both messages at once to avoid duplicate renders
      setChatHistory(prev => {
        // Remove any existing messages with the same query to prevent duplicates
        const filteredHistory = prev.filter(msg => 
          msg.type !== 'user' || msg.content !== searchResults.query
        );
        return [...filteredHistory, userMessage, assistantMessage];
      });
    }
  }, [searchResults]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 150)}px`;
    }
  }, [query]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !videoId) return;

    // Add user message to chat
    const userMessage: ChatMessage = {
      type: 'user',
      content: query.trim(),
      timestamp: new Date().toISOString()
    };
    setChatHistory(prev => [...prev, userMessage]);

    try {
      if (searchType === 'visual') {
        await onVisualSearch(query.trim());
      } else {
        await onSearch(query.trim());
      }
      setQuery('');
      if (inputRef.current) {
        inputRef.current.style.height = 'auto';
      }
    } catch (err) {
      console.error('Search error:', err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
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

  const renderChatMessage = (message: ChatMessage) => {
    console.log('Rendering message:', message);
    const isUser = message.type === 'user';
    const hasSources = message.sources && message.sources.length > 0;
    console.log('Message has sources:', hasSources, message.sources);

    return (
      <div key={message.timestamp} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
        <div className={`max-w-[80%] ${isUser ? 'bg-blue-500 text-white' : 'bg-gray-100'} rounded-lg p-4`}>
          {/* Message content */}
          <div className="whitespace-pre-wrap">
            {isUser ? message.content : (
              <>
                <div className="font-medium mb-2">Answer:</div>
                <div>{message.content}</div>
              </>
            )}
          </div>

          {/* Sources section */}
          {!isUser && hasSources && message.sources && (
            <div className="mt-4 space-y-2">
              <div className="text-sm font-semibold text-gray-600">Sources:</div>
              {message.sources.map((source, index) => {
                const timestamp = source.timestamp || 0;
                const minutes = Math.floor(timestamp / 60);
                const seconds = Math.floor(timestamp % 60);
                const formattedTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;

                return (
                  <div key={`${message.timestamp}-source-${index}`} className="bg-white rounded p-3 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">
                        {formattedTime}
                      </span>
                      <button
                        onClick={() => seekToTimestamp?.(timestamp)}
                        className="text-blue-500 hover:text-blue-700 text-sm flex items-center space-x-1"
                      >
                        <Clock className="h-4 w-4" />
                        <span>Jump to {formattedTime}</span>
                      </button>
                    </div>
                    <p className="text-sm text-gray-600">{source.text}</p>
                    {source.confidence && (
                      <div className="text-xs text-gray-500 mt-1">
                        Confidence: {(source.confidence * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <Bot className="h-6 w-6 text-primary dark:text-primary-light" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Chat with Your Video</h2>
        </div>
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
          <button
            type="button"
            onClick={() => setSearchType('transcript')}
            className={`flex items-center justify-center space-x-2 py-1.5 px-3 rounded-md text-sm font-medium transition-all ${
              searchType === 'transcript'
                ? 'bg-white dark:bg-gray-600 text-primary dark:text-primary-light shadow-sm'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <MessageSquare className="h-4 w-4" />
            <span>Content</span>
          </button>
          <button
            type="button"
            onClick={() => setSearchType('visual')}
            className={`flex items-center justify-center space-x-2 py-1.5 px-3 rounded-md text-sm font-medium transition-all ${
              searchType === 'visual'
                ? 'bg-white dark:bg-gray-600 text-primary dark:text-primary-light shadow-sm'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Eye className="h-4 w-4" />
            <span>Visuals</span>
          </button>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Quick Starters - Moved back to chat area */}
        {chatHistory.length === 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
              Try asking about:
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {conversationStarters[searchType].map((starter, index) => (
                <button
                  key={index}
                  onClick={() => setQuery(starter)}
                  className="text-left p-3 text-sm bg-gray-50 dark:bg-gray-700/50 text-gray-700 dark:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  {starter}
                </button>
              ))}
            </div>
          </div>
        )}

        {chatHistory.map(renderChatMessage)}
        {isSearching && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-gray-100 dark:bg-gray-700 rounded-2xl rounded-tl-none p-4 shadow-sm">
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-gray-500 dark:text-gray-400" />
                <span className="text-sm text-gray-500 dark:text-gray-400">Searching...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t dark:border-gray-700 bg-white dark:bg-gray-800">
        <form onSubmit={handleSubmit} className="flex items-end space-x-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask about the video's ${searchType === 'transcript' ? 'content' : 'visuals'}...`}
              className="w-full resize-none rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-primary dark:focus:ring-primary-light focus:border-transparent p-3 pr-12 min-h-[44px] max-h-[150px]"
              rows={1}
            />
            <button
              type="submit"
              disabled={!query.trim() || isSearching}
              className="absolute right-2 bottom-2 p-1.5 text-primary dark:text-primary-light hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};