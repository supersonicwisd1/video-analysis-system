import { cacheService } from './cache';

interface ChatMessage {
  message: string;
  timestamp: number;
  conversationId: string;
  videoId: string;
}

interface BatchedRequest {
  messages: ChatMessage[];
  timestamp: number;
  timeout: NodeJS.Timeout;
}

class ChatService {
  private ws: WebSocket | null = null;
  private messageQueue: ChatMessage[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;
  private readonly BATCH_INTERVAL = 1000; // 1 second
  private readonly MAX_BATCH_SIZE = 5;
  private reconnectAttempts = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private readonly RECONNECT_DELAY = 1000; // 1 second

  constructor(private wsUrl: string) {}

  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.handleReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onmessage = this.handleMessage.bind(this);
      } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        reject(error);
      }
    });
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts - 1);

    setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        console.error('Reconnection failed:', error);
        this.handleReconnect();
      }
    }, delay);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'chat_response':
          this.handleChatResponse(data);
          break;
        case 'error':
          console.error('WebSocket error:', data.error);
          break;
        default:
          console.warn('Unknown message type:', data.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  private async handleChatResponse(data: any): Promise<void> {
    const { videoId, conversationId, response, sources } = data;
    
    // Cache the response
    await cacheService.appendChatMessage(videoId, conversationId, {
      response,
      sources,
      timestamp: Date.now(),
    });

    // Dispatch event for UI update
    window.dispatchEvent(new CustomEvent('chatResponse', {
      detail: { videoId, conversationId, response, sources }
    }));
  }

  async sendMessage(message: ChatMessage): Promise<void> {
    // Add message to queue
    this.messageQueue.push(message);

    // Start batch timeout if not already running
    if (!this.batchTimeout) {
      this.batchTimeout = setTimeout(() => {
        this.processBatch();
      }, this.BATCH_INTERVAL);
    }

    // Process batch immediately if max size reached
    if (this.messageQueue.length >= this.MAX_BATCH_SIZE) {
      this.processBatch();
    }
  }

  private async processBatch(): Promise<void> {
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }

    if (this.messageQueue.length === 0) return;

    // Ensure WebSocket connection
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      try {
        await this.connect();
      } catch (error) {
        console.error('Failed to connect for batch processing:', error);
        return;
      }
    }

    // Group messages by conversation
    const batches = new Map<string, ChatMessage[]>();
    for (const message of this.messageQueue) {
      const key = `${message.videoId}:${message.conversationId}`;
      if (!batches.has(key)) {
        batches.set(key, []);
      }
      batches.get(key)!.push(message);
    }

    // Send each batch
    for (const [key, messages] of batches) {
      const [videoId, conversationId] = key.split(':');
      
      try {
        // Send batch request
        this.ws!.send(JSON.stringify({
          type: 'batch_chat',
          payload: {
            videoId,
            conversationId,
            messages: messages.map(m => ({
              message: m.message,
              timestamp: m.timestamp
            }))
          }
        }));

        // Cache messages
        for (const message of messages) {
          await cacheService.appendChatMessage(
            message.videoId,
            message.conversationId,
            {
              message: message.message,
              timestamp: message.timestamp,
              status: 'sent'
            }
          );
        }
      } catch (error) {
        console.error('Error sending batch:', error);
        // Cache failed messages
        for (const message of messages) {
          await cacheService.appendChatMessage(
            message.videoId,
            message.conversationId,
            {
              message: message.message,
              timestamp: message.timestamp,
              status: 'failed',
              error: error instanceof Error ? error.message : 'Unknown error'
            }
          );
        }
      }
    }

    // Clear queue
    this.messageQueue = [];
  }

  async getChatHistory(videoId: string, conversationId: string): Promise<any[]> {
    // Try cache first
    const cached = await cacheService.getChatHistory(videoId, conversationId);
    if (cached) return cached;

    // If not in cache, request from server
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      await this.connect();
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Timeout waiting for chat history'));
      }, 5000);

      const handler = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'chat_history' && 
              data.videoId === videoId && 
              data.conversationId === conversationId) {
            this.ws!.removeEventListener('message', handler);
            clearTimeout(timeout);
            resolve(data.messages);
          }
        } catch (error) {
          console.error('Error handling chat history response:', error);
        }
      };

      this.ws!.addEventListener('message', handler);
      this.ws!.send(JSON.stringify({
        type: 'get_chat_history',
        payload: { videoId, conversationId }
      }));
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }
  }
}

// Export singleton instance
export const chatService = new ChatService(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'); 