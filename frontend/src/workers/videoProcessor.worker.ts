// Video processing worker
import { VideoSegment, ContentType } from '../types/video';

// Message types
type WorkerMessage = {
  type: 'processFrames' | 'analyzeSegments' | 'generateThumbnails';
  payload: any;
};

type WorkerResponse = {
  type: string;
  success: boolean;
  data?: any;
  error?: string;
};

// Frame processing
async function processFrames(videoElement: HTMLVideoElement, numFrames: number): Promise<{ timestamp: number; imageData: ImageData }[]> {
  const frames: { timestamp: number; imageData: ImageData }[] = [];
  const canvas = new OffscreenCanvas(videoElement.videoWidth, videoElement.videoHeight);
  const ctx = canvas.getContext('2d');
  
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  const interval = videoElement.duration / numFrames;
  
  for (let i = 0; i < numFrames; i++) {
    const timestamp = i * interval;
    videoElement.currentTime = timestamp;
    
    // Wait for seek to complete
    await new Promise(resolve => {
      videoElement.onseeked = resolve;
    });
    
    // Draw frame to canvas
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    frames.push({
      timestamp,
      imageData,
    });
  }
  
  return frames;
}

// Segment analysis
function analyzeSegments(segments: VideoSegment[]): { 
  sections: any[];
  keywords: string[];
  summary: string;
} {
  // Group segments into sections based on timing and content
  const sections = [];
  let currentSection = null;
  
  for (const segment of segments) {
    if (!currentSection || segment.start_time - currentSection.end_time > 60) {
      if (currentSection) {
        sections.push(currentSection);
      }
      currentSection = {
        start_time: segment.start_time,
        end_time: segment.end_time,
        content: segment.content,
        type: segment.content_type,
        segments: [segment],
      };
    } else {
      currentSection.end_time = segment.end_time;
      currentSection.content += '\n' + segment.content;
      currentSection.segments.push(segment);
    }
  }
  
  if (currentSection) {
    sections.push(currentSection);
  }
  
  // Extract keywords (simple implementation)
  const keywords = new Set<string>();
  const text = segments
    .filter(s => s.content_type === ContentType.TRANSCRIPT)
    .map(s => s.content)
    .join(' ')
    .toLowerCase();
  
  const words = text.split(/\W+/);
  const wordFreq: { [key: string]: number } = {};
  
  for (const word of words) {
    if (word.length > 3) {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    }
  }
  
  // Get top 10 keywords
  const sortedWords = Object.entries(wordFreq)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([word]) => word);
  
  sortedWords.forEach(word => keywords.add(word));
  
  // Generate simple summary
  const summary = sections
    .slice(0, 3)
    .map(s => s.content.split('.')[0])
    .join('. ');
  
  return {
    sections,
    keywords: Array.from(keywords),
    summary,
  };
}

// Thumbnail generation
async function generateThumbnails(frames: { timestamp: number; imageData: ImageData }[]): Promise<string[]> {
  const thumbnails: string[] = [];
  const canvas = new OffscreenCanvas(320, 180); // 16:9 aspect ratio
  const ctx = canvas.getContext('2d');
  
  if (!ctx) {
    throw new Error('Could not get canvas context');
  }
  
  for (const frame of frames) {
    // Resize frame to thumbnail size
    ctx.putImageData(frame.imageData, 0, 0);
    
    // Convert to blob
    const blob = await canvas.convertToBlob({
      type: 'image/jpeg',
      quality: 0.8,
    });
    
    // Convert to base64
    const reader = new FileReader();
    const base64 = await new Promise<string>((resolve) => {
      reader.onloadend = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
    
    thumbnails.push(base64);
  }
  
  return thumbnails;
}

// Message handler
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;
  
  try {
    let response: WorkerResponse;
    
    switch (type) {
      case 'processFrames':
        const frames = await processFrames(payload.videoElement, payload.numFrames);
        response = {
          type,
          success: true,
          data: frames,
        };
        break;
        
      case 'analyzeSegments':
        const analysis = analyzeSegments(payload.segments);
        response = {
          type,
          success: true,
          data: analysis,
        };
        break;
        
      case 'generateThumbnails':
        const thumbnails = await generateThumbnails(payload.frames);
        response = {
          type,
          success: true,
          data: thumbnails,
        };
        break;
        
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
    
    self.postMessage(response);
  } catch (error) {
    self.postMessage({
      type,
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
};

// Export worker
export {}; 