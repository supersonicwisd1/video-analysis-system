import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import { dirname, resolve } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src')
    }
  },
  base: '/', // Ensure we're not using chrome-extension:// URLs
  optimizeDeps: {
    force: true
  },
  build: {
    sourcemap: true,
    outDir: 'dist',
    assetsDir: 'assets'
  }
})