import { VideoApi } from './videoApi';
import { VideoSegment, ContentType, ProcessingStatus } from '../types/video';

interface ProcessingOptions {
  quality?: 'auto' | 'high' | 'medium' | 'low';
  parallelProcessing?: boolean;
  preloadSegments?: boolean;
  maxParallelTasks?: number;
}

interface NetworkConditions {
  effectiveType: 'slow-2g' | '2g' | '3g' | '4g';
  downlink: number;
  rtt: number;
  saveData: boolean;
}

export class VideoProcessingService {
  private static instance: VideoProcessingService;
  private processingQueue: Map<string, Promise<any>> = new Map();
  private preloadedSegments: Map<string, VideoSegment[]> = new Map();
  private networkMonitor: NetworkInformation | null = null;
  private currentQuality: string = 'auto';
  private readonly maxParallelTasks = 2;

  private constructor() {
    // Initialize network monitoring if available
    if ('connection' in navigator) {
      this.networkMonitor = (navigator as any).connection;
      this.networkMonitor?.addEventListener('change', this.handleNetworkChange);
    }
  }

  public static getInstance(): VideoProcessingService {
    if (!VideoProcessingService.instance) {
      VideoProcessingService.instance = new VideoProcessingService();
    }
    return VideoProcessingService.instance;
  }

  private handleNetworkChange = () => {
    if (!this.networkMonitor) return;
    
    const conditions: NetworkConditions = {
      effectiveType: this.networkMonitor.effectiveType,
      downlink: this.networkMonitor.downlink,
      rtt: this.networkMonitor.rtt,
      saveData: this.networkMonitor.saveData
    };

    this.updateVideoQuality(conditions);
  };

  private updateVideoQuality(conditions: NetworkConditions) {
    if (conditions.saveData || conditions.effectiveType === 'slow-2g') {
      this.currentQuality = 'low';
    } else if (conditions.effectiveType === '2g' || conditions.downlink < 1) {
      this.currentQuality = 'low';
    } else if (conditions.effectiveType === '3g' || conditions.downlink < 3) {
      this.currentQuality = 'medium';
    } else {
      this.currentQuality = 'high';
    }
  }

  async processVideo(videoId: string, options: ProcessingOptions = {}): Promise<ProcessingStatus> {
    const {
      quality = 'auto',
      parallelProcessing = true,
      preloadSegments = true,
      maxParallelTasks = 3
    } = options;

    // Set initial quality based on network conditions
    if (quality === 'auto' && this.networkMonitor) {
      this.updateVideoQuality({
        effectiveType: this.networkMonitor.effectiveType,
        downlink: this.networkMonitor.downlink,
        rtt: this.networkMonitor.rtt,
        saveData: this.networkMonitor.saveData
      });
    } else {
      this.currentQuality = quality;
    }

    // Start parallel processing if enabled
    if (parallelProcessing) {
      return this.processVideoParallel(videoId, maxParallelTasks, preloadSegments);
    }

    // Fallback to sequential processing
    return this.processVideoSequential(videoId, preloadSegments);
  }

  private async processVideoParallel(
    videoId: string,
    maxParallelTasks: number,
    preloadSegments: boolean
  ): Promise<ProcessingStatus> {
    const tasks = [
      this.processTranscript(videoId),
      this.processVisualContent(videoId),
      this.processMetadata(videoId)
    ];

    // Process tasks in parallel with a limit
    const results = await Promise.allSettled(
      tasks.map(task => this.executeWithQueueLimit(task, maxParallelTasks))
    );

    // Handle results and start preloading if enabled
    if (preloadSegments) {
      this.startBackgroundPreloading(videoId);
    }

    // Combine results and return status
    return this.combineProcessingResults(results, videoId);
  }

  private async processVideoSequential(
    videoId: string,
    preloadSegments: boolean
  ): Promise<ProcessingStatus> {
    try {
      // Process transcript
      const transcriptResult = await this.processTranscript(videoId);
      
      // Process visual content
      const visualResult = await this.processVisualContent(videoId);
      
      // Process metadata
      const metadataResult = await this.processMetadata(videoId);

      // Start preloading if enabled
      if (preloadSegments) {
        this.startBackgroundPreloading(videoId);
      }

      return {
        status: 'completed',
        video_id: videoId,
        progress: 1,
        current_step: 'completed',
        message: 'Video processing completed successfully'
      };
    } catch (error) {
      return {
        status: 'failed',
        video_id: videoId,
        progress: 0,
        current_step: 'failed',
        message: error instanceof Error ? error.message : 'Processing failed'
      };
    }
  }

  private async executeWithQueueLimit<T>(
    task: Promise<T>,
    maxParallelTasks: number
  ): Promise<T> {
    // Wait if we've reached the parallel task limit
    while (this.processingQueue.size >= maxParallelTasks) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Add task to queue
    const taskPromise = task.finally(() => {
      this.processingQueue.delete(task.toString());
    });
    this.processingQueue.set(task.toString(), taskPromise);

    return taskPromise;
  }

  private async processTranscript(videoId: string): Promise<any> {
    return VideoApi.processTranscript(videoId, { quality: this.currentQuality });
  }

  private async processVisualContent(videoId: string): Promise<any> {
    return VideoApi.processVisualContent(videoId, { quality: this.currentQuality });
  }

  private async processMetadata(videoId: string): Promise<any> {
    return VideoApi.getVideoInfo(videoId);
  }

  private async startBackgroundPreloading(videoId: string) {
    // Get initial segments
    const segments = await this.getInitialSegments(videoId);
    if (!segments.length) return;

    // Start preloading in the background
    this.preloadNextSegments(videoId, segments);
  }

  private async getInitialSegments(videoId: string): Promise<VideoSegment[]> {
    try {
      const response = await VideoApi.getVideoSegments(videoId, {
        limit: 5,
        quality: this.currentQuality
      });
      return response.segments;
    } catch (error) {
      console.error('Failed to get initial segments:', error);
      return [];
    }
  }

  private async preloadNextSegments(videoId: string, currentSegments: VideoSegment[]) {
    if (!currentSegments.length) return;

    // Store current segments
    this.preloadedSegments.set(videoId, currentSegments);

    // Get the last segment's end time
    const lastSegment = currentSegments[currentSegments.length - 1];
    const nextStartTime = lastSegment.end_time;

    try {
      // Preload next batch of segments
      const nextSegments = await VideoApi.getVideoSegments(videoId, {
        start_time: nextStartTime,
        limit: 5,
        quality: this.currentQuality
      });

      // Update preloaded segments
      this.preloadedSegments.set(videoId, [
        ...currentSegments,
        ...nextSegments.segments
      ]);

      // Continue preloading if there are more segments
      if (nextSegments.has_more) {
        this.preloadNextSegments(videoId, nextSegments.segments);
      }
    } catch (error) {
      console.error('Failed to preload next segments:', error);
    }
  }

  private combineProcessingResults(
    results: PromiseSettledResult<any>[],
    videoId: string
  ): ProcessingStatus {
    const errors = results
      .filter((result): result is PromiseRejectedResult => result.status === 'rejected')
      .map(result => result.reason);

    if (errors.length > 0) {
      return {
        status: 'failed',
        video_id: videoId,
        progress: 0,
        current_step: 'failed',
        message: errors.map(e => e.message).join(', ')
      };
    }

    return {
      status: 'completed',
      video_id: videoId,
      progress: 1,
      current_step: 'completed',
      message: 'Video processing completed successfully'
    };
  }

  getPreloadedSegments(videoId: string): VideoSegment[] {
    return this.preloadedSegments.get(videoId) || [];
  }

  clearPreloadedSegments(videoId: string) {
    this.preloadedSegments.delete(videoId);
  }

  cleanup() {
    if (this.networkMonitor) {
      this.networkMonitor.removeEventListener('change', this.handleNetworkChange);
    }
    this.processingQueue.clear();
    this.preloadedSegments.clear();
  }

  /**
   * Get the optimal video quality based on network conditions
   */
  public getOptimalQuality(): 'auto' | 'high' | 'medium' | 'low' {
    if (!this.networkMonitor) {
      return 'auto';
    }

    const { effectiveType, downlink } = this.networkMonitor;

    // If we're on a slow connection or offline, use low quality
    if (effectiveType === 'slow-2g' || effectiveType === '2g' || downlink < 1) {
      return 'low';
    }

    // If we're on 3G or have moderate bandwidth, use medium quality
    if (effectiveType === '3g' || downlink < 3) {
      return 'medium';
    }

    // For 4G and better connections, use high quality
    if (effectiveType === '4g' || downlink >= 3) {
      return 'high';
    }

    // Default to auto if we can't determine
    return 'auto';
  }
} 