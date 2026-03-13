import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API requests to Python backend
      '/api': {
        target: 'http://localhost:8443',  // Updated to actual backend port
        changeOrigin: true,
        secure: false,
        ws: true,
      },
      '/static': {
        target: 'http://localhost:8443',  // Updated to actual backend port
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
