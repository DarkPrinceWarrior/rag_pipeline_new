import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // Forward API calls to FastAPI during dev to avoid CORS
      '/ask': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/healthz': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/docs': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: resolve(__dirname, '../app/static'),
    emptyOutDir: true
  }
})

