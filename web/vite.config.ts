import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Everything the SPA calls in dev is proxied to a locally running `DV_Service`; in production the
// service itself serves the build at /ui, so the app always talks same-origin.
const API = ['/schema', '/validate', '/describe', '/prepare', '/execute', '/results', '/book'];

export default defineConfig({
  base: '/ui/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          echarts: ['echarts/core', 'echarts/charts', 'echarts/components', 'echarts/renderers'],
        },
      },
    },
  },
  server: {
    proxy: Object.fromEntries(
      API.map((p) => [p, { target: 'http://127.0.0.1:8000', changeOrigin: false }])),
  },
});
