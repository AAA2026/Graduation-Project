import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/styles': path.resolve(__dirname, './src/styles'),
    },
  },
  server: {
    port: 3000,
    open: true,
    host: true,
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    cssMinify: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'chart-vendor': ['recharts'],
          'motion-vendor': ['motion'],
        },
      },
    },
  },
  css: {
    devSourcemap: true,
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'lucide-react', 'recharts', 'motion', 'sonner'],
  },
})
