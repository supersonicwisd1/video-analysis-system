import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import { VideoProcessingService } from '../services/videoProcessingService';
import { VideoApi } from '../services/videoApi';

declare global {
  interface Window {
    YT: any;
    onYouTubeIframeAPIReady: () => void;
  }
}

export interface VideoPlayerRef {
  seekTo: (seconds: number) => void;
  playVideo: () => void;
  pauseVideo: () => void;
  getCurrentTime: () => number;
  setQuality: (quality: 'auto' | 'high' | 'medium' | 'low') => void;
}

interface VideoPlayerProps {
  videoId: string;
  onTimeUpdate?: (currentTime: number) => void;
  onStateChange?: (isPlaying: boolean) => void;
  onQualityChange?: (quality: string) => void;
  onError?: (error: Error) => void;
  initialQuality?: 'auto' | 'high' | 'medium' | 'low';
  autoPlay?: boolean;
  startTime?: number;
  className?: string;
}

const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(
  ({ videoId, onTimeUpdate, onStateChange, onQualityChange, onError, initialQuality = 'auto', autoPlay = false, startTime = 0, className }, ref) => {
    const playerRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isPlayerReady, setIsPlayerReady] = useState(false);
    const [currentQuality, setCurrentQuality] = useState(initialQuality);
    const [isLoading, setIsLoading] = useState(true);
    const [initError, setInitError] = useState<string | null>(null);
    const [isContainerMounted, setIsContainerMounted] = useState(false);
    const processingService = useRef(VideoProcessingService.getInstance());
    const playerContainerId = useRef(`youtube-player-${Math.random().toString(36).substr(2, 9)}`);

    // Log component mount and props
    useEffect(() => {
      console.log('VideoPlayer mounted with props:', {
        videoId,
        initialQuality,
        autoPlay,
        startTime,
        className,
        playerContainerId: playerContainerId.current
      });
    }, []);

    // Handle container mount
    useEffect(() => {
      console.log('Setting up container mount check');
      
      const checkContainer = () => {
        const element = document.getElementById(playerContainerId.current);
        console.log('Checking container:', {
          elementExists: !!element,
          elementType: element?.tagName,
          elementClasses: element?.className,
          isContainerMounted
        });
        
        if (element && !isContainerMounted) {
          console.log('Container element found and not mounted, setting mounted state');
          setIsContainerMounted(true);
        }
      };

      // Initial check
      checkContainer();

      // Set up a mutation observer on the parent element
      const parentElement = document.querySelector('.relative.w-full.h-full.bg-black');
      if (parentElement) {
        const observer = new MutationObserver((mutations) => {
          let shouldCheck = false;
          mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
              mutation.addedNodes.forEach((node) => {
                if (node instanceof Element && node.id === playerContainerId.current) {
                  shouldCheck = true;
                }
              });
            }
          });
          if (shouldCheck) {
            checkContainer();
          }
        });

        observer.observe(parentElement, {
          childList: true,
          subtree: true
        });

        return () => {
          console.log('Cleaning up container observer');
          observer.disconnect();
        };
      }
    }, [isContainerMounted]);

    // Initialize YouTube IFrame API
    useEffect(() => {
      console.log('YouTube initialization effect running:', {
        isContainerMounted,
        videoId,
        hasYT: !!window.YT,
        playerReady: isPlayerReady,
        playerContainerId: playerContainerId.current
      });

      if (!isContainerMounted) {
        console.log('Container not mounted yet, waiting...');
        return;
      }

      if (!videoId) {
        console.log('No video ID provided, waiting...');
        return;
      }

      let isComponentMounted = true;
      let playerInstance: any = null;

      const initializePlayer = () => {
        console.log('Initializing player for video:', videoId, {
          containerId: playerContainerId.current,
          containerExists: !!document.getElementById(playerContainerId.current)
        });

        const container = document.getElementById(playerContainerId.current);
        if (!container) {
          console.error('Container not found');
          if (isComponentMounted) {
            setInitError('Player container not found');
            setIsLoading(false);
          }
          return;
        }

        try {
          console.log('Creating YouTube player instance');
          playerInstance = new window.YT.Player(playerContainerId.current, {
            height: '100%',
            width: '100%',
            videoId: videoId,
            playerVars: {
              autoplay: autoPlay ? 1 : 0,
              start: startTime,
              modestbranding: 1,
              rel: 0,
              showinfo: 0,
              controls: 1,
              playsinline: 1,
              enablejsapi: 1,
              origin: window.location.origin,
              widget_referrer: window.location.href,
            },
            events: {
              onReady: (event: any) => {
                if (!isComponentMounted) return;
                console.log('YouTube player ready event triggered', {
                  playerState: event.target.getPlayerState(),
                  videoData: event.target.getVideoData()
                });
                playerRef.current = event.target;
                setIsPlayerReady(true);
                setIsLoading(false);
                if (autoPlay) {
                  console.log('Auto-playing video');
                  event.target.playVideo();
                }
              },
              onStateChange: (event: any) => {
                if (!isComponentMounted) return;
                console.log('Player state changed:', {
                  state: event.data,
                  stateName: getStateName(event.data)
                });
                onStateChange?.(event.data === window.YT.PlayerState.PLAYING);
              },
              onError: (event: any) => {
                if (!isComponentMounted) return;
                console.error('YouTube Player Error:', {
                  errorCode: event.data,
                  errorName: getErrorName(event.data)
                });
                setInitError(`YouTube Player Error: ${getErrorName(event.data)}`);
                setIsLoading(false);
                onError?.(new Error(`YouTube Player Error: ${getErrorName(event.data)}`));
              },
            },
          });

          // Set up interval for time updates
          const timeUpdateInterval = setInterval(() => {
            if (playerRef.current && isPlayerReady && isComponentMounted) {
              try {
                const currentTime = playerRef.current.getCurrentTime();
                onTimeUpdate?.(currentTime);
              } catch (error) {
                console.error('Error getting current time:', error);
                clearInterval(timeUpdateInterval);
              }
            }
          }, 1000);

          return () => {
            clearInterval(timeUpdateInterval);
            if (playerInstance) {
              try {
                playerInstance.destroy();
                playerRef.current = null;
                setIsPlayerReady(false);
              } catch (error) {
                console.error('Error destroying player:', error);
              }
            }
          };
        } catch (error) {
          console.error('Error initializing YouTube player:', error);
          if (isComponentMounted) {
            setInitError('Failed to initialize YouTube player');
            setIsLoading(false);
            onError?.(error instanceof Error ? error : new Error('Failed to initialize YouTube player'));
          }
        }
      };

      let cleanup: (() => void) | undefined;

      if (!window.YT) {
        console.log('YouTube IFrame API not loaded, loading script...');
        const tag = document.createElement('script');
        tag.src = 'https://www.youtube.com/iframe_api';
        const firstScriptTag = document.getElementsByTagName('script')[0];
        firstScriptTag.parentNode?.insertBefore(tag, firstScriptTag);

        window.onYouTubeIframeAPIReady = () => {
          console.log('YouTube IFrame API ready callback triggered');
          cleanup = initializePlayer();
        };
      } else {
        console.log('YouTube IFrame API already loaded, initializing player directly');
        cleanup = initializePlayer();
      }

      return () => {
        console.log('Cleaning up YouTube player');
        isComponentMounted = false;
        if (cleanup) {
          cleanup();
        }
      };
    }, [videoId, isContainerMounted]);

    // Handle quality changes
    useEffect(() => {
      if (isPlayerReady && playerRef.current) {
        try {
          const qualityMap = {
            'high': 'hd1080',
            'medium': 'hd720',
            'low': 'large',
            'auto': 'default'
          };
          // Only set quality if player is still valid
          if (playerRef.current.getPlayerState) {
            playerRef.current.setPlaybackQuality(qualityMap[currentQuality]);
            onQualityChange?.(currentQuality);
          }
        } catch (error) {
          console.error('Error setting playback quality:', error);
        }
      }
    }, [currentQuality, isPlayerReady]);

    // Handle network changes for quality
    useEffect(() => {
      const handleNetworkChange = () => {
        const quality = processingService.current.getOptimalQuality();
        setCurrentQuality(quality);
      };

      window.addEventListener('online', handleNetworkChange);
      window.addEventListener('offline', handleNetworkChange);

      return () => {
        window.removeEventListener('online', handleNetworkChange);
        window.removeEventListener('offline', handleNetworkChange);
      };
    }, []);

    // Helper functions for logging
    const getStateName = (state: number): string => {
      const states: { [key: number]: string } = {
        [-1]: 'unstarted',
        0: 'ended',
        1: 'playing',
        2: 'paused',
        3: 'buffering',
        5: 'video cued'
      };
      return states[state] || `unknown (${state})`;
    };

    const getErrorName = (errorCode: number): string => {
      const errors: { [key: number]: string } = {
        2: 'invalid parameter',
        5: 'HTML5 player error',
        100: 'video not found',
        101: 'embedding not allowed',
        150: 'embedding not allowed'
      };
      return errors[errorCode] || `unknown error (${errorCode})`;
    };

    useImperativeHandle(ref, () => ({
      seekTo: (seconds: number) => {
        if (playerRef.current && isPlayerReady) {
          playerRef.current.seekTo(seconds, true);
        }
      },
      playVideo: () => {
        if (playerRef.current && isPlayerReady) {
          playerRef.current.playVideo();
        }
      },
      pauseVideo: () => {
        if (playerRef.current && isPlayerReady) {
          playerRef.current.pauseVideo();
        }
      },
      getCurrentTime: () => {
        if (playerRef.current && isPlayerReady) {
          return playerRef.current.getCurrentTime();
        }
        return 0;
      },
      setQuality: (quality: 'auto' | 'high' | 'medium' | 'low') => {
        setCurrentQuality(quality);
      },
    }));

    return (
      <div className={`relative w-full h-full bg-black ${className || ''}`}>
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-white">Loading video player...</div>
          </div>
        )}
        {initError && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-red-500">Error: {initError}</div>
          </div>
        )}
        <div 
          id={playerContainerId.current}
          className="w-full h-full"
          data-testid="youtube-player-container"
        />
      </div>
    );
  }
);

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer; 