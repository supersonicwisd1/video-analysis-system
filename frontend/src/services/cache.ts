import { openDB, DBSchema, IDBPDatabase } from 'idb';

interface VideoCacheDB extends DBSchema {
  videos: {
    key: string;  // video_id
    value: {
      metadata: any;
      lastAccessed: number;
      segments: any[];
      sections: any[];
    };
  };
  chatHistory: {
    key: string;  // `${video_id}:${conversation_id}`
    value: {
      messages: any[];
      lastUpdated: number;
    };
  };
  searchResults: {
    key: string;  // `${video_id}:${queryHash}`
    value: {
      results: any;
      timestamp: number;
      query: string;
    };
  };
}

class CacheService {
  private db: IDBPDatabase<VideoCacheDB> | null = null;
  private readonly DB_NAME = 'video-analysis-cache';
  private readonly VERSION = 1;
  private readonly CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days

  async initialize(): Promise<void> {
    if (this.db) return;

    this.db = await openDB<VideoCacheDB>(this.DB_NAME, this.VERSION, {
      upgrade(db) {
        // Create video store
        if (!db.objectStoreNames.contains('videos')) {
          const videoStore = db.createObjectStore('videos', { keyPath: 'key' });
          videoStore.createIndex('lastAccessed', 'lastAccessed');
        }

        // Create chat history store
        if (!db.objectStoreNames.contains('chatHistory')) {
          const chatStore = db.createObjectStore('chatHistory', { keyPath: 'key' });
          chatStore.createIndex('lastUpdated', 'lastUpdated');
        }

        // Create search results store
        if (!db.objectStoreNames.contains('searchResults')) {
          const searchStore = db.createObjectStore('searchResults', { keyPath: 'key' });
          searchStore.createIndex('timestamp', 'timestamp');
        }
      },
    });

    // Clean up old entries
    await this.cleanup();
  }

  private async cleanup(): Promise<void> {
    if (!this.db) return;

    const now = Date.now();
    const cutoff = now - this.CACHE_DURATION;

    // Clean up old videos
    const videoTx = this.db.transaction('videos', 'readwrite');
    const videoIndex = videoTx.store.index('lastAccessed');
    let cursor = await videoIndex.openCursor();
    while (cursor) {
      if (cursor.value.lastAccessed < cutoff) {
        await cursor.delete();
      }
      cursor = await cursor.continue();
    }

    // Clean up old chat history
    const chatTx = this.db.transaction('chatHistory', 'readwrite');
    const chatIndex = chatTx.store.index('lastUpdated');
    cursor = await chatIndex.openCursor();
    while (cursor) {
      if (cursor.value.lastUpdated < cutoff) {
        await cursor.delete();
      }
      cursor = await cursor.continue();
    }

    // Clean up old search results
    const searchTx = this.db.transaction('searchResults', 'readwrite');
    const searchIndex = searchTx.store.index('timestamp');
    cursor = await searchIndex.openCursor();
    while (cursor) {
      if (cursor.value.timestamp < cutoff) {
        await cursor.delete();
      }
      cursor = await cursor.continue();
    }
  }

  // Video metadata caching
  async getVideoMetadata(videoId: string): Promise<any | null> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('videos', 'readonly');
    const store = tx.objectStore('videos');
    const result = await store.get(videoId);
    if (result) {
      // Update last accessed time
      await this.updateVideoLastAccessed(videoId);
      return result.metadata;
    }
    return null;
  }

  async setVideoMetadata(videoId: string, metadata: any, segments: any[], sections: any[]): Promise<void> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('videos', 'readwrite');
    const store = tx.objectStore('videos');
    await store.put({
      key: videoId,
      value: {
        metadata,
        segments,
        sections,
        lastAccessed: Date.now(),
      },
    });
  }

  private async updateVideoLastAccessed(videoId: string): Promise<void> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('videos', 'readwrite');
    const store = tx.objectStore('videos');
    const data = await store.get(videoId);
    if (data) {
      data.value.lastAccessed = Date.now();
      await store.put(data);
    }
  }

  // Chat history caching
  async getChatHistory(videoId: string, conversationId: string): Promise<any[] | null> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('chatHistory', 'readonly');
    const store = tx.objectStore('chatHistory');
    const key = `${videoId}:${conversationId}`;
    const result = await store.get(key);
    return result?.messages || null;
  }

  async appendChatMessage(videoId: string, conversationId: string, message: any): Promise<void> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('chatHistory', 'readwrite');
    const store = tx.objectStore('chatHistory');
    const key = `${videoId}:${conversationId}`;
    
    const existing = await store.get(key);
    const messages = existing?.messages || [];
    messages.push(message);

    await store.put({
      key,
      value: {
        messages,
        lastUpdated: Date.now(),
      },
    });
  }

  // Search results caching
  async getSearchResults(videoId: string, query: string): Promise<any | null> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('searchResults', 'readonly');
    const store = tx.objectStore('searchResults');
    const key = `${videoId}:${this.hashQuery(query)}`;
    const result = await store.get(key);
    
    // Check if result is still valid (within 1 hour)
    if (result && Date.now() - result.timestamp < 60 * 60 * 1000) {
      return result.results;
    }
    return null;
  }

  async setSearchResults(videoId: string, query: string, results: any): Promise<void> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction('searchResults', 'readwrite');
    const store = tx.objectStore('searchResults');
    const key = `${videoId}:${this.hashQuery(query)}`;
    
    await store.put({
      key,
      value: {
        results,
        query,
        timestamp: Date.now(),
      },
    });
  }

  private hashQuery(query: string): string {
    // Simple hash function for queries
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  // Utility methods
  async clearVideoCache(videoId: string): Promise<void> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction(['videos', 'chatHistory', 'searchResults'], 'readwrite');
    
    // Clear video metadata
    await tx.objectStore('videos').delete(videoId);
    
    // Clear related chat history
    const chatStore = tx.objectStore('chatHistory');
    let cursor = await chatStore.openCursor();
    while (cursor) {
      if (cursor.key.startsWith(`${videoId}:`)) {
        await cursor.delete();
      }
      cursor = await cursor.continue();
    }
    
    // Clear related search results
    const searchStore = tx.objectStore('searchResults');
    cursor = await searchStore.openCursor();
    while (cursor) {
      if (cursor.key.startsWith(`${videoId}:`)) {
        await cursor.delete();
      }
      cursor = await cursor.continue();
    }
  }

  async clearAll(): Promise<void> {
    if (!this.db) await this.initialize();
    const tx = this.db!.transaction(['videos', 'chatHistory', 'searchResults'], 'readwrite');
    await Promise.all([
      tx.objectStore('videos').clear(),
      tx.objectStore('chatHistory').clear(),
      tx.objectStore('searchResults').clear(),
    ]);
  }
}

// Export singleton instance
export const cacheService = new CacheService(); 