import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config: run dev server on port 8080 and proxy API calls under /api to backend at 127.0.0.1:8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
