import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8899',
    },
  },
})
