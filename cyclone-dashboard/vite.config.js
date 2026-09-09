import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    proxy: {
      // Route ./api/* to the local integration API backend (phase6 + /api).
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
    fs: {
      allow: [
        // Allow serving files from the workspace root (one level up)
        path.resolve(import.meta.dirname, '..'),
        path.resolve(import.meta.dirname),
      ]
    }
  },
})
