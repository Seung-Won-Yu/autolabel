import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    port: 5173,
    proxy: {
      // e2e는 실제 DB를 안 건드리게 별도 포트의 백엔드로 붙는다
      '/api': `http://127.0.0.1:${process.env.VITE_API_PORT || 8899}`,
    },
  },
})
